"""Dataset-source abstraction for the annotation pipeline.

Everything downstream (DB, embeddings, grids, training) accesses images and
per-item metadata ONLY through :class:`DatasetSource`. Nothing outside this
module may assume the on-disk format (LabelChecker CSV + collage TIFFs today;
plain image folders or other formats later).

Identity rules (see ANNOTATION.md / pipeline spec):
* ``dataset_id`` comes from a ``.dataset_id`` marker file written into the
  dataset folder on first registration — renaming/moving the folder keeps
  identity.
* ``item_key`` is defined by the source and must be stable across re-scans:
  LabelChecker rows use the ``Uuid`` column (falling back to
  ``CollageFile|X|Y|W|H``); plain folders use the relative path.
* No absolute paths are persisted anywhere; the data root is resolved at
  runtime (``PLANKTON_DATA_DIR`` env var, default ``./data``).
"""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT_ENV = "PLANKTON_DATA_DIR"
MARKER_FILE = ".dataset_id"
PRED_COL = "LabelPredicted"
LC_LABEL, LC_EXCLUDED, LC_SEEN, LC_PROBS = ("LC_Label", "LC_Excluded",
                                            "LC_Seen", "LC_Probs")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}


def data_root() -> Path:
    return Path(os.environ.get(DATA_ROOT_ENV, "data"))


def ensure_dataset_id(folder: Path) -> str:
    """Read the folder's identity marker, creating it on first registration."""
    marker = folder / MARKER_FILE
    if marker.exists():
        val = marker.read_text().strip()
        if val:
            return val
    val = uuid.uuid4().hex
    marker.write_text(val + "\n")
    return val


# ── value types ─────────────────────────────────────────────────────────────────

@dataclass
class ItemRef:
    """One annotatable image as enumerated by a source."""
    item_key: str
    meta: dict = field(default_factory=dict)   # predicted, lc_label, lc_excluded, …


@dataclass
class LabelState:
    """Label bookkeeping for one item, as handed to ``export_labels``."""
    label: str = ""
    excluded: str = ""        # pipe-separated ruled-out classes
    seen: int = 0
    probs: str = ""           # JSON {class: prob}


# ── abstract source ──────────────────────────────────────────────────────────────

class DatasetSource(ABC):
    """A folder (or other container) of annotatable images."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.dataset_id = ensure_dataset_id(self.root)
        self.name = self.root.name

    source_type: str = "abstract"
    supports_export: bool = False

    @abstractmethod
    def list_items(self) -> list[ItemRef]:
        """Enumerate all items with stable keys + per-item metadata."""

    @abstractmethod
    def load_images(self, item_keys) -> dict[str, np.ndarray]:
        """Load RGB uint8 arrays for the given keys (missing keys omitted)."""

    def load_image(self, item_key: str) -> np.ndarray | None:
        return self.load_images([item_key]).get(item_key)

    def initial_classes(self) -> list[str]:
        """Seed taxonomy shipped with the data, if any."""
        return []

    def export_labels(self, labels: dict[str, LabelState]) -> int:
        """Write label state back into the native format. Returns rows written."""
        raise NotImplementedError(f"{self.source_type} does not support export")

    def refresh(self) -> None:
        """Drop any caches so the next call re-reads the underlying storage."""

    def __repr__(self):
        return (f"<{type(self).__name__} id={self.dataset_id[:8]} "
                f"name={self.name!r}>")


# ── LabelChecker CSV + collage TIFF crops ────────────────────────────────────────

class LabelCheckerSource(DatasetSource):
    """FlowCam-style dataset: LabelChecker*.csv rows are crops in collage TIFFs."""

    source_type = "labelchecker"
    supports_export = True

    def __init__(self, root: Path, csv_path: Path | None = None):
        super().__init__(root)
        self.csv_path = Path(csv_path) if csv_path else self._find_csv(self.root)
        self._df: pd.DataFrame | None = None
        self._keys: list[str] | None = None

    @staticmethod
    def _find_csv(root: Path) -> Path:
        hits = sorted(root.glob("LabelChecker*.csv"))
        if not hits:
            raise FileNotFoundError(f"no LabelChecker*.csv in {root}")
        return hits[0]

    def refresh(self) -> None:
        self._df, self._keys = None, None

    def _load(self) -> tuple[pd.DataFrame, list[str]]:
        if self._df is None:
            df = pd.read_csv(self.csv_path)
            self._df = df
            self._keys = self._make_keys(df)
        return self._df, self._keys

    @staticmethod
    def _make_keys(df: pd.DataFrame) -> list[str]:
        """Uuid column when usable, else CollageFile|X|Y|W|H (+ dup suffix)."""
        if "Uuid" in df.columns:
            vals = df["Uuid"].fillna("").astype(str).str.strip()
            if (vals != "").all() and vals.nunique() == len(df):
                return vals.tolist()
        keys, seen = [], {}
        for _, row in df.iterrows():
            base = "|".join(str(row.get(c, "")) for c in
                            ("CollageFile", "ImageX", "ImageY", "ImageW", "ImageH"))
            n = seen.get(base, 0)
            seen[base] = n + 1
            keys.append(base if n == 0 else f"{base}#{n}")
        return keys

    @staticmethod
    def _cell(row, col) -> str:
        val = row.get(col, "")
        return "" if pd.isna(val) else str(val)

    def list_items(self) -> list[ItemRef]:
        df, keys = self._load()
        items = []
        for (_, row), key in zip(df.iterrows(), keys):
            seen_raw = pd.to_numeric(row.get(LC_SEEN), errors="coerce")
            items.append(ItemRef(key, {
                "predicted": self._cell(row, PRED_COL),
                "lc_label": self._cell(row, LC_LABEL),
                "lc_excluded": self._cell(row, LC_EXCLUDED),
                "lc_seen": 0 if pd.isna(seen_raw) else int(seen_raw),
                "lc_probs": self._cell(row, LC_PROBS),
            }))
        return items

    def initial_classes(self) -> list[str]:
        df, _ = self._load()
        if PRED_COL not in df.columns:
            return []
        return sorted(c for c in df[PRED_COL].dropna().unique()
                      if isinstance(c, str) and c.strip())

    def load_images(self, item_keys) -> dict[str, np.ndarray]:
        df, keys = self._load()
        index = {k: i for i, k in enumerate(keys)}
        cache: dict[str, Image.Image | None] = {}
        out: dict[str, np.ndarray] = {}
        for key in item_keys:
            i = index.get(key)
            if i is None:
                continue
            row = df.iloc[i]
            cf = row.get("CollageFile")
            if not isinstance(cf, str) or not cf:
                continue
            if cf not in cache:
                path = self._find_collage(cf)
                try:
                    cache[cf] = Image.open(path).convert("RGB") if path else None
                except Exception:
                    cache[cf] = None
            img = cache[cf]
            if img is None:
                continue
            try:
                x, y, w, h = (int(row["ImageX"]), int(row["ImageY"]),
                              int(row["ImageW"]), int(row["ImageH"]))
                out[key] = np.array(img.crop((x, y, x + w, y + h)))
            except Exception:
                continue
        return out

    def _find_collage(self, collage_file: str) -> Path | None:
        candidate = self.root / collage_file
        if candidate.exists():
            return candidate
        for p in self.root.rglob(collage_file):
            return p
        return None

    def export_labels(self, labels: dict[str, LabelState]) -> int:
        """Merge LC_ columns onto a fresh on-disk read (atomic replace).

        Only the LC_ columns are touched so columns written by other tools
        (e.g. the Dashboard's BioVolume) are preserved.
        """
        disk = pd.read_csv(self.csv_path)
        keys = self._make_keys(disk)
        for col in (LC_LABEL, LC_EXCLUDED, LC_PROBS):
            if col not in disk.columns:
                disk[col] = ""
            disk[col] = disk[col].fillna("").astype(str)
        if LC_SEEN not in disk.columns:
            disk[LC_SEEN] = 0
        disk[LC_SEEN] = (pd.to_numeric(disk[LC_SEEN], errors="coerce")
                         .fillna(0).astype(int))
        written = 0
        for i, key in enumerate(keys):
            st = labels.get(key)
            if st is None:
                continue
            disk.at[i, LC_LABEL] = st.label
            disk.at[i, LC_EXCLUDED] = st.excluded
            disk.at[i, LC_SEEN] = int(st.seen)
            disk.at[i, LC_PROBS] = st.probs
            written += 1
        tmp = self.csv_path.with_suffix(".csv.tmp")
        disk.to_csv(tmp, index=False)
        os.replace(tmp, self.csv_path)         # atomic on the same volume
        self.refresh()
        return written


# ── plain folder of image files ──────────────────────────────────────────────────

class ImageFolderSource(DatasetSource):
    """A folder of individual image files; subfolder name = initial label."""

    source_type = "imagefolder"
    supports_export = False

    def _files(self):
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS and p.name != MARKER_FILE:
                yield p

    def list_items(self) -> list[ItemRef]:
        items = []
        for p in self._files():
            rel = p.relative_to(self.root)
            predicted = rel.parent.name if rel.parent != Path(".") else ""
            items.append(ItemRef(rel.as_posix(), {"predicted": predicted}))
        return items

    def initial_classes(self) -> list[str]:
        return sorted({i.meta["predicted"] for i in self.list_items()
                       if i.meta["predicted"]})

    def load_images(self, item_keys) -> dict[str, np.ndarray]:
        out = {}
        for key in item_keys:
            path = self.root / key
            if not path.exists():
                continue
            try:
                out[key] = np.array(Image.open(path).convert("RGB"))
            except Exception:
                continue
        return out


# ── discovery ────────────────────────────────────────────────────────────────────

def open_dataset(folder: Path) -> DatasetSource | None:
    """Pick the right source implementation for one dataset folder."""
    folder = Path(folder)
    if any(folder.glob("LabelChecker*.csv")):
        return LabelCheckerSource(folder)
    if any(p.suffix.lower() in IMAGE_EXTS for p in folder.rglob("*")):
        return ImageFolderSource(folder)
    return None


def discover_datasets(root: Path | None = None) -> list[DatasetSource]:
    """Scan the data root; one DatasetSource per recognisable subfolder."""
    root = Path(root) if root else data_root()
    if not root.exists():
        return []
    sources = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        src = open_dataset(sub)
        if src is not None:
            sources.append(src)
    return sources
