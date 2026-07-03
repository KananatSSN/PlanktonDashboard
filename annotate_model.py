"""MobileNetV3-Small image classifier for the grid annotation tool.

CPU-only proof of concept. The model is fine-tuned (classifier head by default)
on plankton crops. It is used in two places by ``annotate.py``:

* ``predict_proba`` — score grid candidates so the grid can *minimise* the
  number of images predicted as the target class.
* ``fit`` — fine-tune on the labels the user records while annotating, plus a
  one-off ``Train Base Model`` bootstrap on the existing ``LabelPredicted``
  column so predictions are meaningful from the very first round.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

DEVICE = torch.device("cpu")
CKPT_DIR = Path("models")
CKPT_PATH = CKPT_DIR / "annotate_mobilenetv3.pt"

# ── fine-tune recipe (PoC, CPU) ────────────────────────────────────────────────
# Tweak these to change the training behaviour of both buttons.
RECIPE = {
    "img_size": 160,        # crops are resized to img_size x img_size
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "freeze_backbone": True,  # train only the classifier head -> fast on CPU
    "finetune_epochs": 4,     # "Update Model" — fine-tune on recorded labels
    "base_epochs": 3,         # "Train Base Model" — bootstrap on LabelPredicted
    "base_max_samples": 1200,  # subsample bootstrap set so it finishes in minutes
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _to_pil(arr) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return Image.fromarray(a).convert("RGB")


def _build_transform(img_size: int, train: bool = False) -> transforms.Compose:
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
    return transforms.Compose(steps)


class _CropDataset(torch.utils.data.Dataset):
    """Holds raw crop arrays (cheap) and applies the transform lazily."""

    def __init__(self, crops, labels_idx, tf):
        self.crops = crops
        self.y = labels_idx
        self.tf = tf

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        return self.tf(_to_pil(self.crops[i])), self.y[i]


class PlanktonClassifier:
    """MobileNetV3-Small over a fixed class list, held in memory by annotate.py."""

    def __init__(self, classes, pretrained: bool = True):
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.model = self._make_model(len(self.classes), pretrained)
        self.model.to(DEVICE).eval()
        self.trained = False
        self.lock = threading.Lock()  # callbacks must not train concurrently

    @staticmethod
    def _make_model(num_classes: int, pretrained: bool) -> nn.Module:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, num_classes)
        return m

    # ── inference ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_proba(self, crops) -> np.ndarray:
        """Return softmax probabilities, shape (len(crops), num_classes)."""
        if not crops:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        tf = _build_transform(RECIPE["img_size"], train=False)
        self.model.eval()
        out = []
        bs = RECIPE["batch_size"]
        for i in range(0, len(crops), bs):
            batch = crops[i:i + bs]
            x = torch.stack([tf(_to_pil(c)) for c in batch]).to(DEVICE)
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()
            out.append(probs)
        return np.concatenate(out, axis=0)

    # ── training ─────────────────────────────────────────────────────────────--
    def _set_trainable(self):
        freeze = RECIPE["freeze_backbone"]
        for name, p in self.model.named_parameters():
            # classifier head is always trainable; backbone optional
            p.requires_grad = (not freeze) or name.startswith("classifier")

    def fit(self, crops, label_names, epochs: int) -> dict:
        """Fine-tune (warm-start) on crops with string class labels."""
        if not crops:
            return {"n": 0, "loss": None}
        y = torch.tensor([self.class_to_idx[l] for l in label_names], dtype=torch.long)
        tf = _build_transform(RECIPE["img_size"], train=True)
        loader = torch.utils.data.DataLoader(
            _CropDataset(crops, y, tf),
            batch_size=RECIPE["batch_size"], shuffle=True, num_workers=0,
        )
        self._set_trainable()
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=RECIPE["lr"],
                               weight_decay=RECIPE["weight_decay"])
        crit = nn.CrossEntropyLoss()
        self.model.train()
        last = None
        for _ in range(epochs):
            running, seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                seen += xb.size(0)
            last = running / max(seen, 1)
        self.model.eval()
        self.trained = True
        return {"n": len(crops), "loss": last, "epochs": epochs}

    # ── checkpoint ───────────────────────────────────────────────────────────--
    def save(self, path: Path = CKPT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(),
             "classes": self.classes,
             "trained": self.trained},
            path,
        )

    def load(self, path: Path = CKPT_PATH) -> bool:
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=DEVICE)
        if list(ckpt.get("classes", [])) != self.classes:
            return False  # class set changed -> ignore stale checkpoint
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(DEVICE).eval()
        self.trained = bool(ckpt.get("trained", True))
        return True


# ── module-level holder ─────────────────────────────────────────────────────--
_CLF: PlanktonClassifier | None = None


def get_classifier(classes) -> PlanktonClassifier:
    """Return a classifier for ``classes``, rebuilding if the class set changed.

    Loads a checkpoint from disk when one matching the class set exists.
    """
    global _CLF
    classes = list(classes)
    if _CLF is None or _CLF.classes != classes:
        _CLF = PlanktonClassifier(classes)
        _CLF.load()
    return _CLF
