# Plankton Annotation Tool

A human-in-the-loop web app for classifying plankton crops into an **open,
growing set of classes**. Pick a target class, batch-accept the model's
confident guesses or hunt its missed ones in a grid, resolve hard cases in an
active-learning queue, and manage classes as they evolve. A **MobileNetV3 /
torchvision** classifier trains on your labels in the background — with
aggressive augmentation — while you keep annotating.

Built with **FastAPI + HTMX**, backed by **SQLite**. CPU-only; runs in the
`3dmodel` conda env.

```powershell
& "C:/Users/acer/anaconda3/envs/3dmodel/python.exe" annotate_server.py
```
→ http://127.0.0.1:8051  (the embedded Dash dashboard is under `/dashboard/`)

First run: register datasets and import any existing labels, then bootstrap the
model from the **Annotate** tab (Train Base Model):

```powershell
python annotate_db.py sync      # register datasets, migrate existing LC_ labels
```

---

## 1. Architecture at a glance

```
 dataset folders (data/*/)
        │  DatasetSource adapter (annotate_data.py)   ← format-agnostic
        ▼
   SQLite metadata store  (annotate_db.py)            ← source of truth for labels
        │
        ▼   the Session (annotate_session.py) drives every workflow:
   ┌──────────────┬───────────────┬────────────────┬──────────────────┐
   │ Grid review  │ Hard-case grid│ Active learning│ Manage classes   │
   │ (weak labels)│ (human labels)│ (uncertain+div)│ rename/merge/del │
   └──────────────┴───────────────┴────────────────┴──────────────────┘
        │                    ▲
        ▼                    │ background train-on-copy (RandAugment),
   MobileNet / torchvision ──┘ manual or auto every N labels
   (annotate_model.py)
        │
        ▼   export
   LC_ columns written back to LabelChecker*.csv
```

The model scores a **sample** of the unlabelled pool each grid build (crops are
loaded and run through the CNN). Training happens on a **copy** of the model in a
background thread; the fresh model is swapped in atomically when ready, so
annotation never blocks on training.

---

## 2. File layout

| File | Responsibility |
|------|----------------|
| `annotate_server.py` | FastAPI app, HTMX endpoints, mounts the Dash dashboard. Owns the single `Session`. |
| `annotate_session.py`| The stateful, DB-backed `Session`: grids, AL queue, class management, background training, export, health. |
| `annotate_data.py`   | `DatasetSource` abstraction + `LabelCheckerSource`, `ImageFolderSource`, discovery. |
| `annotate_db.py`     | SQLite schema, sync/migrate/export, label ops, class-management ops. CLI. |
| `annotate_model.py`  | `PlanktonClassifier` — any torchvision backbone, RandAugment training, penultimate features. |
| `annotate_settings.py`| JSON-persisted training/grid settings + form field metadata. |
| `annotate_core.py`   | Legacy DataFrame helpers still used for base64 / tooltips. |
| `dashboard.py`       | The Plotly Dash feature-scatter tool, mounted under `/dashboard/`. |
| `templates/`         | Jinja2 + HTMX fragments (one `_tab_*` + body per tab). |
| `static/`            | `app.css`, vendored `htmx.min.js`. |
| `models/`            | `model_<backbone>.pt` checkpoint (gitignored). |
| `annotate.db`        | SQLite store (gitignored). |
| `annotate_settings.json` | Saved settings (gitignored; regenerated from defaults). |
| `data/<ds>/.dataset_id` | Stable identity marker (see §5). |

---

## 3. Data abstraction (`annotate_data.py`)

The real deployment will have **many dataset folders**, and future sources will
**not always be CSV + TIFF**. All image/metadata access goes through:

```python
class DatasetSource(ABC):
    dataset_id: str                       # stable identity (from a marker file)
    def list_items() -> list[ItemRef]     # (item_key, meta) per crop
    def load_images(keys) -> {key: ndarray}
    def initial_classes() -> list[str]    # seed taxonomy, may be []
    def export_labels(labels) -> int      # write-back hook (may no-op)
```

Two implementations ship: **`LabelCheckerSource`** (FlowCam `LabelChecker*.csv` +
collage TIFF crops; `item_key` = the row's `Uuid`; seeds classes from
`LabelPredicted`; exports `LC_` columns) and **`ImageFolderSource`** (a plain
folder of image files; `item_key` = relative path; read-only).
`discover_datasets()` scans the data root (`PLANKTON_DATA_DIR`, default `data/`)
and returns one source per subfolder. The pool and classes span **all**
registered datasets.

### 3.1 Currently supported formats

| Format | Detected by | `item_key` | Seed classes | Export |
|--------|-------------|-----------|--------------|--------|
| **LabelChecker CSV + TIFF** | a `LabelChecker*.csv` | `Uuid` (else `CollageFile\|X\|Y\|W\|H`) | `LabelPredicted` values | `LC_` columns |
| **Image folder** | any image file, no CSV | relative path | subfolder names | none (read-only) |

### 3.2 Adding a new format

Nothing downstream knows the on-disk format — it lives only in the source
subclass. To add one:

1. **Subclass `DatasetSource`** and implement `list_items()` (stable,
   content-derived `item_key`s — never row numbers) and `load_images()` (RGB
   uint8 arrays).
2. **Optional:** `initial_classes()`, `export_labels()` (set
   `supports_export=True`; atomic `.tmp`+`os.replace`, touch only your label
   fields), `refresh()`.
3. **Register** it in `open_dataset()` with a detection rule.
4. **Optional migration:** expose existing labels in item meta as
   `lc_label`/`lc_excluded`/`lc_seen`/`lc_probs` and `annotate_db.migrate_lc`
   imports them on `sync`.

Then `python annotate_db.py sync` — the new dataset joins the same pool and every
workflow with no other code changes. `ImageFolderSource` (~15 lines) is the
reference for a minimal read-only adapter.

---

## 4. Portability: moving / renaming folders

1. **No absolute paths in the DB** — only dataset-relative `item_key`s; the data
   root is `PLANKTON_DATA_DIR` (default `data/`). *Move `data/` → change one env
   var.*
2. **`dataset_id` is a `.dataset_id` marker (UUID)**, not the folder name.
   *Rename/move a folder → labels re-attach automatically.*
3. **The CSV export is a backup** — labels also live in the `LC_` columns, so the
   DB can be rebuilt from the CSVs (`annotate_db.py sync`).

---

## 5. Metadata store (`annotate_db.py`)

SQLite is the working source of truth. Schema:

```
datasets(id, name, source_type, registered_at, last_synced_at)
images(id, dataset_id, item_key, predicted,
       label, label_source ∈ {none,human,weak,pseudo}, confidence,
       status ∈ {unlabeled,seed,reviewed,rejected},
       excluded (pipe-joined ruled-out classes), seen, probs (JSON),
       cluster_id, embedding_row, created_at, updated_at,
       UNIQUE(dataset_id, item_key))
classes(id, name, origin ∈ {seeded,discovered,manual}, created_at)
labeling_events(id, image_id, class_name,
       action ∈ {accept,reject,relabel,exclude,migrate}, round, annotator, created_at)
```

(`cluster_id` / `embedding_row` are legacy columns, currently unused.)

- **`sync_source`** registers a dataset and upserts items (idempotent);
  **`migrate_lc`** imports existing `LC_` labels; **`export_source`** writes them
  back. Reverse-one-hot exclusions round-trip unchanged.
- The connection is opened `check_same_thread=False` and every request handler
  holds `Session.lock` (an `RLock`); the background training thread never touches
  the DB (its data is gathered under the lock before the thread starts).

---

## 6. The model (`annotate_model.py`)

`PlanktonClassifier` wraps **any of the ~80 torchvision classification
backbones** (`available_backbones()`), default **MobileNetV3-Small**.
`_replace_classifier` swaps the final head generically (the `Linear` head of
ResNet / MobileNet / EfficientNet / ConvNeXt / ViT / DenseNet / VGG, and
SqueezeNet's `Conv2d`). Checkpoint: `models/model_<backbone>.pt`, tagged with the
backbone so a mismatch is rejected on load.

- **Aggressive augmentation** — training uses `RandomResizedCrop` + **RandAugment**
  (`num_ops` random operations of `magnitude` strength, from ~14: rotate, shear,
  translate, colour, contrast, brightness, sharpness, posterize, solarize,
  equalize, …) + flips. Controlled from Settings.
- **`predict_proba(crops)`** scores raw crops; **`features(crops)`** returns the
  penultimate embedding (via a forward hook on the head) for active-learning
  diversity; **`fit(crops, labels, …)`** trains with the given recipe.
- Two training entry points: **Base train** (bootstrap on the `predicted` /
  `LabelPredicted` column) and **Retrain** (on accumulated `human`+`weak` labels).

### Background training (train-on-copy)

Training runs in a daemon thread on a **fresh classifier**; when it finishes, the
Session swaps `self.clf` under the lock. Grid scoring keeps using the current
model until the swap, so **annotation continues during training**. Retraining
fires manually (Retrain button) or **automatically after every `retrain_every`
new labels** (Settings; `0` disables). A `train` status fragment polls
`GET /train-status`.

---

## 7. The tabs / workflows

HTMX swaps each tab body into `#main`; a persistent nav is updated out-of-band.

### Annotate — two grid modes
Both score a random **sample** of the unlabelled pool (`max_pool` crops) with the
model and share one tile/reserve mechanic. Grid state lives server-side.

- **Review** (throughput) — only crops the model **predicts as the target**
  (argmax = x), highest-confidence first. It scans the pool in `max_pool`-sized
  chunks (up to a budget) to gather enough for the grid **plus a deep reserve**,
  so every tile — original or reject-replacement — is a genuine x-candidate and
  **Accept all shown** stays valid batch after batch. Assume they're all correct:
  **click only the tiles that are NOT the target** to reject them (records an
  exclusion, swaps in the next x-candidate); **Accept all shown** labels the rest
  as `weak` and rebuilds; **Reject all** excludes and rebuilds. If the model
  currently predicts none of the pool as x, it says so — Retrain to surface more.
- **Hard-case** — the crops the model is **least confident** about (lowest P(x))
  from a random `max_pool` sample, hiding ones it already predicts confidently
  ("Trust the model when P(any) ≥"). Here **click the tiles that ARE the target**
  to label them `human`.

The Annotate tab states what the model does and, per mode, exactly which tiles
to click. Training shows a live **progress bar** (phase + epoch + % ) that polls
`/train-status` while a background train runs.

Controls: target class, mode, rows m × cols n (up to `grid_max`, default 20),
and **Trust the model when P(any) ≥** (hard-case only) — crops the model already
predicts that confidently (any class) are hidden, so you review only its
uncertain/missed ones.

### Active Learning
`Build queue` scores a pool sample, ranks by **margin** (top-1 − top-2 softmax;
smallest = most uncertain), keeps the top `al_k_uncertain`, then does
**farthest-point sampling over the model's penultimate features** for diversity.
Review one-by-one: image + top-5 predictions + class dropdown, with **Assign**
(human), **Create & assign** a new class, or **Skip**.

### Manage Classes
A per-class table (name, origin, labelled count) with **Rename** (onto an existing
name → merges), **Merge into**, and **Delete** (re-pools the images as
unlabelled). Exclusion sets stay consistent; every change is logged to
`labeling_events`.

### Settings
Edits `annotate_settings.json` and applies live. Fields:

| Setting | Meaning |
|---|---|
| `backbone` | torchvision model to train (changing it resets the model) |
| `img_size` | input crop size (changing it resets the model) |
| `freeze_backbone` | train head only (fast) vs whole network |
| `randaug_ops` / `randaug_magnitude` | RandAugment strength (`0` ops disables) |
| `lr` / `batch_size` / `weight_decay` | optimiser |
| `finetune_epochs` / `base_epochs` / `base_max_samples` | training length |
| `retrain_every` | auto-retrain after N new labels (`0` = manual only) |
| `max_pool` | crops scored per grid build |
| `grid_max` | row/column cap in the UI |
| `al_sample` / `al_k_uncertain` / `al_queue_size` | active-learning queue |

### Dataset Health
Class distribution and labelling progress from the DB (no model needed).

### Model Health
Scores the model against `human` labels (loads those crops, capped at
`EVAL_MAX=600`): accuracy, macro-F1, per-class precision/recall/F1, confusion
matrix.

### Dashboard
The existing Plotly Dash tool mounted in-process under `/dashboard/` via
`a2wsgi`. The export writes only `LC_` columns, so Dash-written columns (e.g.
BioVolume) are preserved.

---

## 8. HTTP endpoints

| Method & path | Purpose |
|---|---|
| `GET /` · `GET /tab/{annotate,active,manage,settings,dataset,model,dashboard}` | Page / tab bodies. |
| `POST /dataset` | Switch dataset (by `dataset_id`). |
| `POST /build` · `/click` · `/accept-all` · `/reject-all` | Grid build + tile actions. |
| `POST /train-base` · `/retrain` · `GET /train-status` | Background training + poll. |
| `POST /al/build` · `/al/assign` · `/al/new` · `/al/skip` | Active-learning queue. |
| `POST /manage/rename` · `/merge` · `/delete` | Class management. |
| `POST /settings/save` | Save settings. |
| `POST /save` | Export labels to the CSV. |
| `/dashboard/*` | The mounted Dash app. |

---

## 9. Command-line tools

```powershell
python annotate_db.py sync      # discover datasets, register, upsert items, migrate LC_ labels
python annotate_db.py export    # write DB label state back to LC_ columns
python annotate_db.py status    # DB summary (counts, classes, events)
```

Run `sync` once per new dataset folder; it is idempotent. There is no separate
embedding step — the model works directly on crops.

---

## 10. Dependencies (`3dmodel` conda env, Python 3.12, CPU)

`torch`, `torchvision` (the classifier + backbones), `fastapi`, `uvicorn`,
`jinja2`, `python-multipart`, `a2wsgi` (mounts Dash), plus `dash`, `plotly`,
`scipy`, `scikit-image`, `pandas`, `Pillow` for the Dashboard and CSV I/O. No
DINOv2 / hdbscan / umap.

> **OpenMP note:** torch + numpy-MKL in one process can trip `OMP: Error #15` on
> this env. `annotate_server.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing
> torch; set the same env var for any script that mixes them.

---

## 11. Persistence & safety

- Labels are written to the DB on each action (the source of truth).
- **Save** exports the DB's `LC_` columns back onto a fresh on-disk read of the
  CSV (atomic `.tmp` + `os.replace`), touching **only** `LC_Label`,
  `LC_Excluded`, `LC_Seen`, `LC_Probs` — other columns are preserved.
- The DB is rebuildable from the CSVs; `models/`, `annotate.db*`,
  `annotate_settings.json` are gitignored.
