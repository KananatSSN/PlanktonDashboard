"""User-editable settings for the annotation tool (JSON-persisted).

Controls how the MobileNet/torchvision model trains and how the grids behave.
Edited from the Settings tab; loaded once at startup and after each save.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SETTINGS_PATH = Path(os.environ.get("PLANKTON_SETTINGS", "annotate_settings.json"))

DEFAULTS = {
    # model / training
    "backbone": "mobilenet_v3_small",   # any torchvision classification model
    "img_size": 160,
    "freeze_backbone": True,            # train head only (fast on CPU) vs whole net
    "randaug_ops": 3,                  # RandAugment ops per image (aggressive)
    "randaug_magnitude": 9,            # RandAugment strength 0-30
    "lr": 1e-3,
    "batch_size": 32,
    "weight_decay": 1e-4,
    "finetune_epochs": 4,              # epochs per retrain
    "base_epochs": 3,                  # epochs for the initial base train
    "base_max_samples": 1500,          # cap on bootstrap crops
    # loop behaviour
    "retrain_every": 20,               # auto-retrain after this many new labels (0=off)
    "max_pool": 256,                   # crops scored per grid build
    "grid_max": 20,                    # row/column cap in the UI
    # active learning
    "al_sample": 600,                  # pool crops scored to build a queue
    "al_k_uncertain": 200,             # most-uncertain kept before diversity dedup
    "al_queue_size": 40,               # images per queue
}

# Field metadata for rendering the Settings form (label, type, help).
FIELDS = [
    ("backbone", "select", "Model backbone (torchvision)"),
    ("img_size", "int", "Input crop size (px)"),
    ("freeze_backbone", "bool", "Freeze backbone (train head only — faster)"),
    ("randaug_ops", "int", "RandAugment ops per image (0 disables)"),
    ("randaug_magnitude", "int", "RandAugment magnitude (0-30)"),
    ("lr", "float", "Learning rate"),
    ("batch_size", "int", "Batch size"),
    ("weight_decay", "float", "Weight decay"),
    ("finetune_epochs", "int", "Epochs per retrain"),
    ("base_epochs", "int", "Epochs for base train"),
    ("base_max_samples", "int", "Max crops for base train"),
    ("retrain_every", "int", "Auto-retrain after N new labels (0 = manual only)"),
    ("max_pool", "int", "Crops scored per grid build"),
    ("grid_max", "int", "Max grid rows / columns"),
    ("al_sample", "int", "Active learning: pool crops scored"),
    ("al_k_uncertain", "int", "Active learning: uncertain kept before dedup"),
    ("al_queue_size", "int", "Active learning: images per queue"),
]

_CASTS = {"int": int, "float": float,
          "bool": lambda v: str(v).lower() in ("1", "true", "on", "yes"),
          "select": str, "str": str}
_TYPE = {name: typ for name, typ, _ in FIELDS}


def load() -> dict:
    cfg = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            cfg.update(json.loads(SETTINGS_PATH.read_text()))
        except Exception:
            pass
    return cfg


def save(cfg: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(cfg, indent=2))


def coerce(raw: dict) -> dict:
    """Cast a form's string values to typed settings, keeping defaults for the rest."""
    cfg = dict(DEFAULTS)
    for name, _typ, _ in FIELDS:
        if name in raw:
            try:
                cfg[name] = _CASTS[_TYPE[name]](raw[name])
            except Exception:
                cfg[name] = DEFAULTS[name]
        elif _TYPE[name] == "bool":
            cfg[name] = False            # unchecked checkboxes are absent from forms
    return cfg
