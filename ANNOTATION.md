# Plankton Annotation Tool — Features & Implementation

A reCAPTCHA-style **active-labelling** web app for the plankton dataset. Pick a
target class, click the grid images that contain it, and a MobileNetV3 classifier
is fine-tuned on what you record. Built with **FastAPI + HTMX** (CPU-only PoC,
runs in the `3dmodel` conda env).

Run:
```powershell
& "C:/Users/acer/anaconda3/envs/3dmodel/python.exe" annotate_server.py
```
→ http://127.0.0.1:8051  (the Dash dashboard stays on 8050)

---

## 1. File layout

| File | Responsibility |
|------|----------------|
| `annotate_server.py` | FastAPI app + HTMX endpoints; owns the single `Session`. |
| `annotate_core.py`   | Framework-agnostic logic: data access, label bookkeeping, grid generation, health metrics, and the stateful `Session`. |
| `annotate_model.py`  | `PlanktonClassifier` — MobileNetV3-Small wrapper (predict / fine-tune / checkpoint) + the `RECIPE`. |
| `dashboard.py`       | The Plotly **Dash** feature-scatter / image / binary tool, mounted under FastAPI as the **Dashboard** tab (also runs standalone on 8050). |
| `templates/`         | Jinja2 templates and HTMX fragments. |
| `static/`            | `app.css` and vendored `htmx.min.js`. |
| `models/`            | Saved checkpoint `annotate_mobilenetv3.pt` (gitignored). |

Data is read exactly like `dashboard.py`: each row of a `LabelChecker*.csv` is a
crop (`ImageX/Y/W/H`) inside a `CollageFile` TIFF.

---

## 2. Labelling data model (new `LC_` columns)

Labels are written into new columns of the `LabelChecker*.csv`:

| Column | Meaning |
|--------|---------|
| `LC_Label`    | Positive class assigned when a crop is **clicked** (a hard one-hot). |
| `LC_Excluded` | Pipe-separated classes **ruled out** for a crop — the zeros of a "reverse one-hot". Pressing **Regenerate** adds the target to every shown crop. |
| `LC_Seen`     | `1` once a crop has appeared in a grid. |
| `LC_Probs`    | JSON `{class: prob}` — the model's full predicted-probability array, written whenever a crop is scored/shown. Also drives the hover tooltip (top-8 `class: prob`). |

Empty cells round-trip through CSV as `NaN`; `ensure_lc_columns()` normalises them
back to `""`/`0` on load so every reader can assume plain strings/ints.

**Reverse one-hot idea:** an unlabelled crop starts with all classes possible.
Each round you work on class `x` and leave a crop unclicked, `x` is removed from
its candidate set (`LC_Excluded += x`). A positive click collapses it to a single
class (`LC_Label = x`).

**Pool for target `x`** (`get_pool`): rows with no positive label **and** `x` not
already in `LC_Excluded` — i.e. "unseen for x".

---

## 3. Grid generation & x-minimisation

The grid deliberately **minimises** images the model already thinks are `x`, so
you surface the model's *missed* positives (hard cases) rather than re-confirming
what it already knows.

`score_pool(target, conf_skip)`:
1. Sample up to `MAX_POOL = 256` rows from the pool for `x`.
2. Load crops (caching each collage TIFF once) and run `predict_proba`.
3. **Drop** rows where `max P(any class) ≥ conf_skip` (default **0.95**) — already
   "known" to the model, not worth a label.
4. Rank the rest by **ascending `P(x)`** (least-likely-x first).

`build_grid(cells, threshold, conf_skip)`:
- Takes the lowest-`P(x)` rows as the `m × n` grid.
- The remaining ranked candidates become a **reserve** queue (`[[row, p], …]`,
  images loaded on demand) used to replace clicked tiles. Reserve prefers rows
  with `P(x) < threshold`.
- Status line, e.g.:
  `Pool 256 scored · 128 high-conf (≥0.95) skipped · 3 predicted-<x> (≥0.5) kept out · grid 6/6 · reserve 122`

`next_replacement(...)`: pops the next reserve tile (skipping ones already shown),
re-scoring the pool only when the reserve empties (and never while training).

### Tunable thresholds (per-grid, in the UI)
- **Rows (m) × Cols (n)** — grid shape, 1–8 each.
- **Remove if P(x) ≥ threshold** (default 0.5) — reserve cutoff / "kept out" count.
- **Skip if max P(any) ≥ conf_skip** (default 0.95) — high-confidence skip rule.

---

## 4. Interaction model (HTMX)

- **Click a tile** → `POST /click`. The crop is labelled `LC_Label = x`
  immediately and the server returns **only that one tile** (`hx-swap="outerHTML"`
  on the tile) plus out-of-band status updates. No full-grid repaint — a single
  DOM node changes, which is what makes swaps instant and flicker-free.
- **Regenerate (none are target)** → `POST /regenerate`: marks every shown crop
  `not-x` (`LC_Excluded += x`), saves, and draws a fresh grid.
- **New grid** → `POST /build`: rebuild without committing negatives (used when
  switching target/size/thresholds).
- **Update Model** / **Train Base Model** → background training (see §6).
- **Save labels** → `POST /save`: flush to disk.

Grid state (shown rows, images, reserve, target, m/n, thresholds, session
positive count) lives **server-side** in the `Session`, so switching tabs and
back preserves the current grid.

---

## 5. Model (MobileNetV3-Small, PyTorch, CPU)

`PlanktonClassifier` in `annotate_model.py`:
- Backbone `torchvision.mobilenet_v3_small` (ImageNet weights), final linear layer
  replaced with `num_classes` (the sorted unique `LabelPredicted` values, 20).
- `predict_proba(crops)` → softmax probabilities; resizes crops to `img_size`,
  ImageNet-normalised, batched.
- `fit(crops, labels, epochs)` → warm-start training; by default only the
  classifier head is trainable (`freeze_backbone`) for speed on CPU.
- `save()/load()` checkpoint to `models/annotate_mobilenetv3.pt` (state dict +
  class list + trained flag). A class-set mismatch ignores a stale checkpoint.

### `RECIPE` (top of `annotate_model.py`)
```
img_size=160, batch_size=32, lr=1e-3, weight_decay=1e-4,
freeze_backbone=True, finetune_epochs=4, base_epochs=3, base_max_samples=1200
```

### Two label sources for training
- **Fine-tune (Update Model)** — `collect_training`: positive `LC_Label` rows use
  their class; reverse-one-hot rows (exclusions, no positive) get a **random class
  from those not excluded**.
- **Base train (Train Base Model)** — `collect_base`: bootstrap on the existing
  `LabelPredicted` column (subsampled to `base_max_samples`) so predictions are
  useful from round one. Click this first.

---

## 6. Background training

Training runs on a **daemon thread** (`Session._train_async`) so the server stays
responsive:
- A `train = {"running": bool, "msg": str}` flag drives an HTMX status fragment
  that polls `GET /train-status` every 1s and stops when done.
- Grid (re)builds are blocked while training; tile replacement won't re-score the
  pool during training (`allow_refill=False`) to avoid concurrent model access.
- `clf.lock` serialises the actual `fit`, and the checkpoint is saved after.

---

## 7. Persistence

- The active dataset's DataFrame is kept **in memory** and mutated per click.
- It is flushed to disk only on **Regenerate**, **Save**, after **training**, and
  on **exit** (`atexit`) — never a full rewrite on every click.
- Writes are **atomic**: `to_csv` to a `.csv.tmp` then `os.replace` onto the
  target (safe on the same volume).
- `save()` merges **only the `LC_` columns** onto a fresh on-disk read, so columns
  written by the embedded Dashboard (e.g. `BioVolume`) are not clobbered.
- `model_status()` shows an `unsaved: yes/no` flag.

> The server loads the first `LabelChecker*.csv` it finds by default and writes
> labels back to it. Work on a copy to avoid touching originals.

---

## 8. Tabs

HTMX swaps each tab body into `#main`; a persistent top nav (`#tabnav`) is updated
out-of-band to show the active tab.

### Annotate
The labelling grid, controls, action buttons, and status lines.

### Dataset Health (`dataset_health`)
No model needed. Summary cards (total / classes / labelled / reverse-one-hot /
seen / untouched) plus a per-class table: original `LabelPredicted` count + share,
human positives, times ruled-out, remaining pool, with a distribution bar.

### Model Health (`model_health`)
Scores the **current model** against your human `LC_Label` positives (chosen
ground truth):
- Overall **accuracy** and **macro-F1**.
- Per-class **precision / recall / F1 / support** (computed without sklearn).
- **Confusion matrix** (rows = true `LC_Label`, cols = predicted).
- Evaluates only labelled rows, capped at `EVAL_MAX = 800` crops for
  responsiveness; shows a friendly empty state before any labels exist and a
  notice if training is in progress.

### Dashboard (embedded Dash)
The existing `dashboard.py` (Plotly Dash: feature scatter, click-to-view image,
binary/threshold tools) is **mounted inside the same FastAPI process** — no second
server or port:

- `annotate_server.py` sets `DASH_URL_PREFIX="/dashboard/"` **before** importing
  `dashboard`, so Dash builds its asset/callback URLs under that prefix; then
  `app.mount("/dashboard", WSGIMiddleware(dashboard.app.server))` (via `a2wsgi`)
  exposes the Flask/WSGI Dash app.
- `dashboard.py` reads that env var at app-creation and passes
  `requests_pathname_prefix`; unset when run standalone, so `python dashboard.py`
  still works on 8050.
- The **Dashboard** tab body is just an `<iframe src="/dashboard/">`.

**Shared-CSV safety:** the Dashboard writes columns like `BioVolume` straight to
the CSV, while the annotation Session holds an in-memory copy. To avoid clobbering
those, `Session.save()` re-reads the on-disk CSV and overwrites **only the `LC_`
columns** (falling back to a full dump if the row count drifts).

---

## 9. HTTP endpoints

| Method & path | Purpose |
|---|---|
| `GET /` | Full page (Annotate tab). |
| `GET /tab/{annotate,dataset,model,dashboard}` | Tab body + OOB nav. Read-only. |
| `/dashboard/*` | The mounted Dash app (its own pages, assets, and callbacks). |
| `POST /dataset` | Switch dataset; returns refreshed target `<option>`s. |
| `POST /build` | Build grid (no negative commit). |
| `POST /regenerate` | Commit shown as not-x, then rebuild. |
| `POST /click` | Label clicked tile positive; return the single replacement tile. |
| `POST /update-model` | Start background fine-tune. |
| `POST /train-base` | Start background bootstrap train. |
| `GET /train-status` | Polled training-status fragment. |
| `POST /save` | Flush labels to CSV. |

Form params for build/regenerate: `target, rows, cols, threshold, conf_skip`.

---

## 10. Key constants

| Constant | Location | Default |
|---|---|---|
| `MAX_POOL` | `annotate_core.py` | 256 (crops scored per grid) |
| `EVAL_MAX` | `annotate_core.py` | 800 (crops scored for Model Health) |
| `conf_skip` | `Session` / UI | 0.95 |
| `threshold` | `Session` / UI | 0.5 |
| grid `m × n` | `Session` / UI | 3 × 3 (1–8 each) |
| `RECIPE` | `annotate_model.py` | see §5 |
| port | `annotate_server.py` | 8051 |

---

## 11. Dependencies (in the `3dmodel` conda env)

`torch`, `torchvision`, `fastapi`, `uvicorn`, `jinja2`, `python-multipart`,
`a2wsgi` (mounts the Dash app), plus `dash`, `plotly`, `scipy`, `scikit-image`
for the Dashboard tab (+ `httpx` for tests). GPU is unavailable — everything runs
on CPU.
