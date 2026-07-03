"""Core logic for the plankton annotation tool (framework-agnostic).

Holds the data access, label bookkeeping (the LC_ columns), grid generation with
x-minimisation, and a stateful ``Session`` used by the FastAPI server. No web
framework is imported here.

Design notes vs. the earlier Dash version:
* The active dataset's DataFrame is kept **in memory** and mutated per click;
  it is flushed to disk only on Save / Regenerate / training (atomic replace),
  instead of rewriting the whole CSV on every click.
* Training runs on a **background thread** so the server stays responsive; the
  grid cannot be (re)built while training to avoid concurrent model access.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import random
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import annotate_model as am

DATA_DIR = Path("data")
PRED_COL = "LabelPredicted"   # bootstrap labels / class source
MAX_POOL = 256                # crops scored per grid generation
LC_LABEL, LC_EXCLUDED, LC_SEEN = "LC_Label", "LC_Excluded", "LC_Seen"
LC_PROBS = "LC_Probs"         # JSON {class: prob} written when a crop is scored/shown
TOOLTIP_TOPK = 8              # classes shown in the hover tooltip


# ── data helpers (mirror dashboard.py) ──────────────────────────────────────────

def find_csv_files():
    return sorted(DATA_DIR.glob("**/LabelChecker*.csv"))


def load_df(csv_path: str) -> pd.DataFrame:
    return ensure_lc_columns(pd.read_csv(csv_path))


def find_image(csv_path: str, collage_file: str) -> Path | None:
    base = Path(csv_path).parent
    candidate = base / collage_file
    if candidate.exists():
        return candidate
    for p in base.parent.rglob(collage_file):
        return p
    return None


def array_to_base64(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── label-state helpers ─────────────────────────────────────────────────────────

def ensure_lc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the LC_ columns if absent and normalise types (NaN -> ""/0)."""
    for col in (LC_LABEL, LC_EXCLUDED, LC_PROBS):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if LC_SEEN not in df.columns:
        df[LC_SEEN] = 0
    df[LC_SEEN] = pd.to_numeric(df[LC_SEEN], errors="coerce").fillna(0).astype(int)
    return df


def probs_to_json(classes, vec) -> str:
    """Full probability array as a compact self-describing JSON object."""
    return json.dumps({c: round(float(p), 4) for c, p in zip(classes, vec)})


def probs_tooltip(classes, vec, row) -> str:
    """Top-K 'class: prob' lines for the image hover tooltip (native title)."""
    pairs = sorted(zip(classes, vec), key=lambda cp: -cp[1])[:TOOLTIP_TOPK]
    return "\n".join([f"row {row}"] + [f"{c}: {float(p):.3f}" for c, p in pairs])


def parse_excluded(val) -> set[str]:
    if not isinstance(val, str) or not val.strip():
        return set()
    return {p for p in val.split("|") if p}


def format_excluded(s: set[str]) -> str:
    return "|".join(sorted(s))


def class_list(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df[PRED_COL].dropna().unique() if isinstance(c, str))


def get_pool(df: pd.DataFrame, target: str) -> list[int]:
    """Rows eligible for the target's grid: not positively labelled, x not ruled out."""
    pool = []
    for r in range(len(df)):
        lab = df.at[r, LC_LABEL]
        if isinstance(lab, str) and lab:
            continue
        if target in parse_excluded(df.at[r, LC_EXCLUDED]):
            continue
        pool.append(r)
    return pool


# ── crop loading (caches each collage once per call) ─────────────────────────────

def load_crops(csv_path: str, df: pd.DataFrame, rows):
    cache: dict[Path, Image.Image | None] = {}
    crops, valid = [], []
    for r in rows:
        row = df.iloc[r]
        cf = row.get("CollageFile")
        if not isinstance(cf, str) or not cf:
            continue
        path = find_image(csv_path, cf)
        if path is None:
            continue
        if path not in cache:
            try:
                cache[path] = Image.open(path).convert("RGB")
            except Exception:
                cache[path] = None
        img = cache[path]
        if img is None:
            continue
        try:
            x, y, w, h = (int(row["ImageX"]), int(row["ImageY"]),
                          int(row["ImageW"]), int(row["ImageH"]))
            crops.append(np.array(img.crop((x, y, x + w, y + h))))
            valid.append(r)
        except Exception:
            continue
    return crops, valid


# ── grid generation with x-minimisation ─────────────────────────────────────────

def score_pool(csv_path, df, clf, target, exclude=None, max_pool=MAX_POOL, conf_skip=1.0):
    """Sample the unseen-for-x pool, score it, return rows ranked by ascending P(x).

    Rows the model predicts with high confidence for *any* class (max softmax
    prob >= ``conf_skip``) are dropped — they're already "known" to the model and
    not worth a human label. Returns (rows, ptar, stats).
    """
    exclude = exclude or set()
    pool = [r for r in get_pool(df, target) if r not in exclude]
    if not pool:
        return [], [], [], {"scored": 0, "high_conf": 0}
    random.shuffle(pool)
    crops, valid = load_crops(csv_path, df, pool[:max_pool])
    if not crops:
        return [], [], [], {"scored": 0, "high_conf": 0}
    probs = clf.predict_proba(crops)
    p = probs[:, clf.class_to_idx[target]]
    pmax = probs.max(axis=1)
    keep = [i for i in range(len(valid)) if pmax[i] < conf_skip]
    keep.sort(key=lambda i: p[i])                     # ascending P(x): least-likely-x first
    stats = {"scored": len(valid), "high_conf": len(valid) - len(keep)}
    return ([valid[i] for i in keep],
            [round(float(p[i]), 3) for i in keep],
            [probs[i].tolist() for i in keep],        # full probability vectors
            stats)


def build_grid(csv_path, df, clf, target, cells, threshold, conf_skip=1.0):
    """Build a grid of ``cells`` tiles plus a reserve queue for replacements.

    Returns (rows, srcs, ptar, reserve, status). ``reserve`` is ``[[row, p], ...]``
    (no images — loaded on demand).
    """
    rows_all, p_all, probs_all, stats = score_pool(csv_path, df, clf, target, conf_skip=conf_skip)
    if not rows_all:
        return [], [], [], [], [], "Pool empty — every image is labelled, ruled out, or high-confidence for this class."

    pmap = dict(zip(rows_all, p_all))
    vmap = dict(zip(rows_all, probs_all))
    crops, valid = load_crops(csv_path, df, rows_all[:cells * 2])   # buffer for dropped crops
    grid_rows = valid[:cells]
    srcs = [array_to_base64(c) for c in crops[:len(grid_rows)]]
    grid_p = [pmap[r] for r in grid_rows]
    grid_probs = [vmap[r] for r in grid_rows]

    used = set(grid_rows)
    rest = [(r, pmap[r]) for r in rows_all if r not in used]
    reserve = [[r, p] for r, p in rest if p < threshold] or [[r, p] for r, p in rest]
    removed = sum(1 for p in p_all if p >= threshold)
    status = (f"Pool {stats['scored']} scored · {stats['high_conf']} high-conf (≥{conf_skip}) "
              f"skipped · {removed} predicted-{target} (≥{threshold}) kept out · "
              f"grid {len(grid_rows)}/{cells} · reserve {len(reserve)}")
    return grid_rows, srcs, grid_p, grid_probs, reserve, status


def next_replacement(csv_path, df, clf, target, threshold, shown_rows, reserve,
                     allow_refill=True, conf_skip=1.0):
    """Pop the next reserve tile (refilling from the pool if allowed).

    Returns (row, src, p, probs_vec, reserve); (None, None, None, None, reserve)
    if exhausted. probs_vec is None when training (model access is blocked).
    """
    shown = set(shown_rows)
    attempts = 2 if allow_refill else 1
    for _ in range(attempts):
        while reserve:
            r, p = reserve.pop(0)
            if r in shown:
                continue
            crops, _ = load_crops(csv_path, df, [r])
            if crops:
                vec = clf.predict_proba(crops)[0].tolist() if allow_refill else None
                return r, array_to_base64(crops[0]), p, vec, reserve
        if not allow_refill:
            break
        rows_all, p_all, _, _ = score_pool(csv_path, df, clf, target, exclude=shown,
                                           conf_skip=conf_skip)
        reserve = ([[r, p] for r, p in zip(rows_all, p_all) if p < threshold]
                   or [[r, p] for r, p in zip(rows_all, p_all)])
        if not reserve:
            break
    return None, None, None, None, reserve


# ── training-set assembly ────────────────────────────────────────────────────────

def collect_training(csv_path, df, classes):
    """Recorded labels -> (crops, label_names). Reverse-one-hot -> random valid class."""
    rows, labels = [], []
    for r in range(len(df)):
        lab = df.at[r, LC_LABEL]
        if isinstance(lab, str) and lab:
            rows.append(r)
            labels.append(lab)
            continue
        ex = parse_excluded(df.at[r, LC_EXCLUDED])
        if ex:
            cand = [c for c in classes if c not in ex]
            if cand:
                rows.append(r)
                labels.append(random.choice(cand))
    crops, valid = load_crops(csv_path, df, rows)
    label_by_row = dict(zip(rows, labels))
    return crops, [label_by_row[v] for v in valid]


def collect_base(csv_path, df, classes):
    """Bootstrap set from the existing LabelPredicted column (subsampled)."""
    known = set(classes)
    rows = [r for r in range(len(df))
            if isinstance(df.at[r, PRED_COL], str) and df.at[r, PRED_COL] in known]
    cap = am.RECIPE["base_max_samples"]
    if len(rows) > cap:
        rows = random.sample(rows, cap)
    crops, valid = load_crops(csv_path, df, rows)
    label_by_row = {r: df.at[r, PRED_COL] for r in rows}
    return crops, [label_by_row[v] for v in valid]


def label_summary(df) -> str:
    pos = int((df[LC_LABEL].str.len() > 0).sum())
    exc = int(((df[LC_EXCLUDED].str.len() > 0) & (df[LC_LABEL].str.len() == 0)).sum())
    seen = int((df[LC_SEEN] > 0).sum())
    return f"labelled +{pos} · reverse-one-hot {exc} · seen {seen}"


# ── health metrics ───────────────────────────────────────────────────────────────

EVAL_MAX = 800   # cap crops scored for Model Health so the request stays responsive


def dataset_health(df, classes) -> dict:
    """Class distribution and labelling progress (no model needed)."""
    from collections import Counter

    n = len(df)
    pred_counts = df[PRED_COL].value_counts().to_dict()
    pos_mask = df[LC_LABEL].str.len() > 0
    unlabeled_mask = ~pos_mask
    pos_counts = df.loc[pos_mask, LC_LABEL].value_counts().to_dict()

    excl_all, excl_unlab = Counter(), Counter()
    excluded = df[LC_EXCLUDED].tolist()
    for i, s in enumerate(excluded):
        ex = parse_excluded(s)
        for c in ex:
            excl_all[c] += 1
            if unlabeled_mask.iat[i]:
                excl_unlab[c] += 1

    n_unlab = int(unlabeled_mask.sum())
    per = [{
        "cls": c,
        "pred": int(pred_counts.get(c, 0)),
        "pos": int(pos_counts.get(c, 0)),
        "ruled_out": int(excl_all.get(c, 0)),
        "pool": n_unlab - int(excl_unlab.get(c, 0)),
    } for c in classes]
    per.sort(key=lambda d: -d["pred"])

    seen = int((df[LC_SEEN] > 0).sum())
    summary = {
        "total": n, "classes": len(classes),
        "labelled": int(pos_mask.sum()),
        "reverse": int((unlabeled_mask & (df[LC_EXCLUDED].str.len() > 0)).sum()),
        "seen": seen, "untouched": n - seen,
    }
    return {"summary": summary, "per_class": per}


def model_health(csv_path, df, clf, classes) -> dict:
    """Per-class precision/recall/F1 of the current model vs human LC_Label."""
    from collections import Counter, defaultdict

    known = set(classes)
    rows = [r for r in range(len(df)) if df.at[r, LC_LABEL] in known]
    total = len(rows)
    if total == 0:
        return {"n": 0, "total": 0}
    sampled = total > EVAL_MAX
    if sampled:
        rows = random.sample(rows, EVAL_MAX)

    crops, valid = load_crops(csv_path, df, rows)
    if not crops:
        return {"n": 0, "total": total}
    true = [df.at[r, LC_LABEL] for r in valid]
    pred = [classes[i] for i in clf.predict_proba(crops).argmax(1)]

    support, tp, fp, fn = Counter(), Counter(), Counter(), Counter()
    conf = defaultdict(lambda: defaultdict(int))
    correct = 0
    for t, p in zip(true, pred):
        support[t] += 1
        conf[t][p] += 1
        if t == p:
            tp[t] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per = []
    for c in sorted(support, key=lambda c: -support[c]):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else None
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        per.append({"cls": c, "support": support[c],
                    "precision": prec, "recall": rec, "f1": f1})
    macro = [x["f1"] for x in per if x["f1"] is not None]
    present = sorted(set(true) | set(pred))
    return {
        "n": len(true), "total": total, "sampled": sampled,
        "accuracy": correct / len(true),
        "macro_f1": (sum(macro) / len(macro)) if macro else None,
        "per_class": per, "present": present,
        "confusion": {t: dict(conf[t]) for t in conf},
    }


# ── stateful session ─────────────────────────────────────────────────────────────

class Session:
    """Single-user annotation session. Holds the in-memory df, model, and grid."""

    def __init__(self):
        self.csv_path: str | None = None
        self.df: pd.DataFrame | None = None
        self.classes: list[str] = []
        self.clf = None
        self.target: str | None = None
        self.threshold: float = 0.5
        self.conf_skip: float = 0.95   # skip images the model predicts (any class) with >= this
        self.m: int = 3            # rows
        self.n: int = 3            # cols
        self.shown: list[int | None] = []
        self.srcs: list[str | None] = []
        self.ptar: list[float | None] = []
        self.probs: list[list | None] = []   # full prob vector per shown cell
        self.reserve: list[list] = []
        self.pos_count: int = 0
        self.dirty: bool = False
        self.status: str = "Pick a target class and press New grid."
        self.train = {"running": False, "msg": ""}
        atexit.register(self._flush_on_exit)

    # ── dataset ----------------------------------------------------------------
    def load_dataset(self, csv_path: str):
        if self.dirty:
            self.save()
        self.csv_path = csv_path
        self.df = load_df(csv_path)
        self.classes = class_list(self.df)
        self.clf = am.get_classifier(self.classes)
        if self.target not in self.classes:
            self.target = self.classes[0] if self.classes else None
        self.shown, self.srcs, self.ptar, self.probs, self.reserve = [], [], [], [], []
        self.pos_count = 0
        self.dirty = False

    def _record_probs(self, row, vec):
        """Persist a scored crop's full probability array into the df (LC_Probs)."""
        if row is not None and vec is not None:
            self.df.at[row, LC_PROBS] = probs_to_json(self.classes, vec)
            self.dirty = True

    # ── label writes (in memory) -----------------------------------------------
    def _commit_positive(self, row: int):
        df, t = self.df, self.target
        df.at[row, LC_SEEN] = 1
        df.at[row, LC_LABEL] = t
        ex = parse_excluded(df.at[row, LC_EXCLUDED])
        ex.discard(t)
        df.at[row, LC_EXCLUDED] = format_excluded(ex)
        self.dirty = True

    def _commit_negative(self, rows) -> int:
        df, t, n = self.df, self.target, 0
        for r in rows:
            if r is None:
                continue
            df.at[r, LC_SEEN] = 1
            if df.at[r, LC_LABEL] == t:
                continue
            ex = parse_excluded(df.at[r, LC_EXCLUDED])
            ex.add(t)
            df.at[r, LC_EXCLUDED] = format_excluded(ex)
            n += 1
        self.dirty = True
        return n

    # ── grid --------------------------------------------------------------------
    def set_params(self, target, m, n, threshold, conf_skip=None):
        self.target = target or self.target
        self.m = max(1, min(8, int(m)))
        self.n = max(1, min(8, int(n)))
        self.threshold = float(threshold)
        if conf_skip is not None:
            self.conf_skip = max(0.0, min(1.0, float(conf_skip)))

    def make_grid(self, commit_neg: bool):
        if self.train["running"]:
            self.status = "Training in progress — please wait before (re)building the grid."
            return
        if commit_neg and any(r is not None for r in self.shown):
            n_neg = self._commit_negative(self.shown)
            self.save()
            prefix = f"Marked {n_neg} not-{self.target}. "
        else:
            prefix = ""
        cells = self.m * self.n
        rows, srcs, ptar, probs, reserve, status = build_grid(
            self.csv_path, self.df, self.clf, self.target, cells, self.threshold,
            conf_skip=self.conf_skip)
        pad = lambda lst, fill: list(lst) + [fill] * (cells - len(lst))
        self.shown = pad(rows, None)
        self.srcs, self.ptar, self.probs = pad(srcs, None), pad(ptar, None), pad(probs, None)
        self.reserve = reserve
        for r, vec in zip(rows, probs):     # persist probabilities for the shown crops
            self._record_probs(r, vec)
        self.status = prefix + status

    def click(self, cell: int) -> bool:
        """Label the clicked tile positive and swap in a replacement. Returns changed?"""
        if cell < 0 or cell >= len(self.shown) or self.shown[cell] is None:
            return False
        row = self.shown[cell]
        self._commit_positive(row)
        self.pos_count += 1
        shown_rows = [r for r in self.shown if r is not None]
        rr, src, p, vec, reserve = next_replacement(
            self.csv_path, self.df, self.clf, self.target, self.threshold,
            shown_rows, self.reserve, allow_refill=not self.train["running"],
            conf_skip=self.conf_skip)
        self.reserve = reserve
        self.shown[cell], self.srcs[cell], self.ptar[cell] = rr, src, p
        self.probs[cell] = vec
        self._record_probs(rr, vec)
        self.status = (f"Labelled row {row} as {self.target} "
                       f"(+{self.pos_count} this session) · reserve {len(reserve)}")
        return True

    # ── training (background) ---------------------------------------------------
    def _train_async(self, collect, epochs, kind):
        def job():
            try:
                crops, labels = collect(self.csv_path, self.df, self.classes)
                if not crops:
                    self.train.update(running=False, msg="Nothing to train on — model unchanged.")
                    return
                with self.clf.lock:
                    res = self.clf.fit(crops, labels, epochs=epochs)
                    self.clf.save()
                self.train.update(
                    running=False,
                    msg=(f"{kind} done: {res['n']} crops, {res['epochs']} epochs, "
                         f"loss {res['loss']:.3f}. Press New grid to use it."))
            except Exception as exc:                       # keep the server alive
                self.train.update(running=False, msg=f"{kind} failed: {exc}")
        self.train.update(running=True, msg=f"{kind} running…")
        threading.Thread(target=job, daemon=True).start()

    def start_update(self):
        if not self.train["running"]:
            self._train_async(collect_training, am.RECIPE["finetune_epochs"], "Fine-tune")

    def start_base(self):
        if not self.train["running"]:
            self._train_async(collect_base, am.RECIPE["base_epochs"], "Base train")

    # ── persistence -------------------------------------------------------------
    def save(self) -> str:
        """Write only the LC_ columns back, preserving other on-disk changes.

        The embedded Dashboard writes columns like BioVolume straight to the CSV;
        merging onto a fresh read (instead of dumping our stale in-memory copy)
        avoids clobbering those. Falls back to a full dump if the row count drifts.
        """
        if self.df is None:
            return "No dataset."
        out = self.df
        try:
            disk = ensure_lc_columns(pd.read_csv(self.csv_path))
            if len(disk) == len(self.df):
                for col in (LC_LABEL, LC_EXCLUDED, LC_SEEN, LC_PROBS):
                    disk[col] = self.df[col].values
                out = disk
        except Exception:
            pass  # disk unreadable -> fall back to writing our own df
        tmp = Path(self.csv_path).with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, self.csv_path)                     # atomic on the same volume
        self.dirty = False
        return "Saved."

    def _flush_on_exit(self):
        try:
            if self.dirty:
                self.save()
        except Exception:
            pass

    # ── views for templates -----------------------------------------------------
    def cells_view(self):
        out = []
        for i, (r, s, p, vec) in enumerate(
                zip(self.shown, self.srcs, self.ptar, self.probs)):
            tip = probs_tooltip(self.classes, vec, r) if vec else (f"row {r}" if r is not None else "")
            out.append({"i": i, "src": s, "row": r, "p": p, "tip": tip})
        return out

    def model_status(self) -> str:
        if not self.classes:
            return "No dataset loaded."
        trained = "trained" if (self.clf and self.clf.trained) else "untrained"
        dirty = f" · unsaved: {'yes' if self.dirty else 'no'}"
        return (f"MobileNetV3-Small · {len(self.classes)} classes · {trained} · "
                f"{label_summary(self.df)}{dirty}")

    # ── health tabs -------------------------------------------------------------
    def dataset_health(self) -> dict:
        return dataset_health(self.df, self.classes)

    def model_health(self) -> dict:
        if self.train["running"]:
            return {"training": True}
        return model_health(self.csv_path, self.df, self.clf, self.classes)
