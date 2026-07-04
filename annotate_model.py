"""Torchvision image classifier for the plankton annotation tool.

The single in-loop model: any torchvision classification backbone (default
MobileNetV3-Small), trained on raw crops with **aggressive RandAugment**. Used to

* ``predict_proba`` — score grid candidates and rank the pool,
* ``fit`` — (re)train on the labels recorded while annotating,
* ``features`` — penultimate embeddings for active-learning diversity sampling.

CPU-only. Training hyper-parameters come from ``annotate_settings`` via the
Session; ``RECIPE`` here is only the fallback default.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tvm
from torchvision import transforms

DEVICE = torch.device("cpu")
CKPT_DIR = Path("models")
DEFAULT_BACKBONE = "mobilenet_v3_small"

# Fallback training defaults (the Settings page overrides these).
RECIPE = {
    "img_size": 160,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "freeze_backbone": True,   # train only the head -> fast on CPU
    "randaug_ops": 3,          # RandAugment: number of ops per image
    "randaug_magnitude": 9,    # RandAugment: strength 0-30
    "finetune_epochs": 4,
    "base_epochs": 3,
    "base_max_samples": 1500,
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def available_backbones() -> list[str]:
    """torchvision classification model names (the backbone dropdown)."""
    return sorted(tvm.list_models(module=tvm))


def model_ckpt_path(backbone: str) -> Path:
    return CKPT_DIR / f"model_{backbone}.pt"


def _to_pil(arr) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return Image.fromarray(a).convert("RGB")


def _build_transform(img_size: int, train: bool = False,
                     randaug_ops: int = 0, randaug_magnitude: int = 9):
    """Eval = resize + normalise. Train = aggressive augmentation on top.

    RandAugment samples from ~14 operations (rotate, shear, translate, colour,
    contrast, brightness, sharpness, posterize, solarize, equalize, …); with
    ``num_ops`` of them per image it is the bulk of the augmentation, plus
    random-resized-crop and flips for scale/orientation invariance.
    """
    if not train:
        steps = [transforms.Resize((img_size, img_size))]
    else:
        steps = [transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0),
                                              ratio=(0.75, 1.33))]
        if randaug_ops > 0:
            steps.append(transforms.RandAugment(
                num_ops=randaug_ops, magnitude=randaug_magnitude))
        steps += [transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
    return transforms.Compose(steps)


def _replace_classifier(model: nn.Module, num_classes: int):
    """Swap a torchvision model's final head for ``num_classes``.

    Handles the final ``nn.Linear`` of most families (resnet ``fc``,
    mobilenet/efficientnet/convnext/vgg ``classifier[-1]``, densenet
    ``classifier``, vit ``heads.head``) and SqueezeNet's ``Conv2d`` head.
    Returns ``(qualified_name, new_module)``.
    """
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_name, last_mod = name, mod

    def _set(path: str, new: nn.Module):
        parts = path.split(".")
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        key = parts[-1]
        if key.isdigit():
            parent[int(key)] = new
        else:
            setattr(parent, key, new)

    if last_name is not None:
        new = nn.Linear(last_mod.in_features, num_classes)
        _set(last_name, new)
        return last_name, new
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and name.startswith("classifier"):
            new = nn.Conv2d(mod.in_channels, num_classes, kernel_size=1)
            _set(name, new)
            if hasattr(model, "num_classes"):
                model.num_classes = num_classes
            return name, new
    raise ValueError("no replaceable classification head found for this backbone")


class _CropDataset(torch.utils.data.Dataset):
    def __init__(self, crops, labels_idx, tf):
        self.crops, self.y, self.tf = crops, labels_idx, tf

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        return self.tf(_to_pil(self.crops[i])), self.y[i]


class PlanktonClassifier:
    """A torchvision backbone over a fixed class list, trained on raw crops."""

    def __init__(self, classes, backbone: str = DEFAULT_BACKBONE,
                 pretrained: bool = True, img_size: int = 160):
        self.classes = list(classes)
        self.backbone = backbone
        self.img_size = img_size
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.model = self._make_model(backbone, len(self.classes), pretrained)
        self.model.to(DEVICE).eval()
        self.trained = False
        self.lock = threading.Lock()

    def _make_model(self, backbone, num_classes, pretrained) -> nn.Module:
        try:
            m = tvm.get_model(backbone, weights="DEFAULT" if pretrained else None)
        except Exception:
            m = tvm.get_model(backbone, weights=None)   # offline / no weights
        self.head_name, self.head_module = _replace_classifier(m, num_classes)
        return m

    # ── inference ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_proba(self, crops) -> np.ndarray:
        if not crops:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        tf = _build_transform(self.img_size, train=False)
        self.model.eval()
        out, bs = [], RECIPE["batch_size"]
        for i in range(0, len(crops), bs):
            x = torch.stack([tf(_to_pil(c)) for c in crops[i:i + bs]]).to(DEVICE)
            out.append(torch.softmax(self.model(x), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def features(self, crops) -> np.ndarray:
        """Penultimate features (the vector fed to the final Linear head).

        Used for active-learning diversity (farthest-point sampling) so the
        queue isn't full of near-duplicates. Captured via a forward hook.
        """
        if not crops or not isinstance(self.head_module, nn.Linear):
            # fall back to probabilities as a coarse feature
            return self.predict_proba(crops)
        feats: list[np.ndarray] = []
        h = self.head_module.register_forward_hook(
            lambda m, inp, out: feats.append(inp[0].detach().cpu().numpy()))
        try:
            self.predict_proba(crops)
        finally:
            h.remove()
        return np.concatenate(feats, axis=0)

    # ── training ─────────────────────────────────────────────────────────────--
    def _set_trainable(self, freeze: bool):
        head = getattr(self, "head_name", "classifier")
        for name, p in self.model.named_parameters():
            p.requires_grad = (not freeze) or name.startswith(head)

    def fit(self, crops, label_names, epochs: int, *, lr=None, batch_size=None,
            weight_decay=None, freeze=None, randaug_ops=None,
            randaug_magnitude=None, progress=None) -> dict:
        """Train on crops with string class labels using the given recipe.

        ``progress`` (optional) is called as ``progress(frac, epoch, loss)`` after
        each batch so the UI can show a live progress bar.
        """
        if not crops:
            return {"n": 0, "loss": None, "epochs": 0}
        lr = RECIPE["lr"] if lr is None else lr
        batch_size = RECIPE["batch_size"] if batch_size is None else batch_size
        weight_decay = RECIPE["weight_decay"] if weight_decay is None else weight_decay
        freeze = RECIPE["freeze_backbone"] if freeze is None else freeze
        randaug_ops = RECIPE["randaug_ops"] if randaug_ops is None else randaug_ops
        randaug_magnitude = (RECIPE["randaug_magnitude"] if randaug_magnitude is None
                             else randaug_magnitude)

        y = torch.tensor([self.class_to_idx[l] for l in label_names], dtype=torch.long)
        tf = _build_transform(self.img_size, train=True,
                              randaug_ops=randaug_ops, randaug_magnitude=randaug_magnitude)
        loader = torch.utils.data.DataLoader(
            _CropDataset(crops, y, tf), batch_size=batch_size, shuffle=True,
            num_workers=0)
        self._set_trainable(freeze)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        crit = nn.CrossEntropyLoss()
        self.model.train()
        last = None
        total_steps = epochs * max(len(loader), 1)
        step = 0
        for ep in range(epochs):
            running, seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                seen += xb.size(0)
                step += 1
                if progress is not None:
                    progress(step / total_steps, ep + 1, loss.item())
            last = running / max(seen, 1)
        self.model.eval()
        self.trained = True
        return {"n": len(crops), "loss": last, "epochs": epochs}

    # ── checkpoint ───────────────────────────────────────────────────────────--
    def _default_path(self) -> Path:
        return model_ckpt_path(self.backbone)

    def save(self, path: Path | None = None):
        path = path or self._default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "classes": self.classes,
                    "backbone": self.backbone, "img_size": self.img_size,
                    "trained": self.trained}, path)

    def load(self, path: Path | None = None) -> bool:
        path = path or self._default_path()
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=DEVICE)
        if (list(ckpt.get("classes", [])) != self.classes
                or ckpt.get("backbone", self.backbone) != self.backbone):
            return False
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(DEVICE).eval()
        self.trained = bool(ckpt.get("trained", True))
        return True
