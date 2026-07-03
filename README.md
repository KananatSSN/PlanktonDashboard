Don't forget to change the data path at line 18 in dashboard.py to your data directory.

## Annotation tool (FastAPI + HTMX)

reCAPTCHA-style active-labelling tool: pick a target class, click the grid
images that contain it, and the model is fine-tuned on what you record.

* `annotate_server.py` — FastAPI app + HTMX endpoints (`templates/`, `static/`).
* `annotate_core.py`   — data access, label bookkeeping, grid generation, health metrics, session.
* `annotate_model.py`  — MobileNetV3-Small classifier (PyTorch, CPU).

Four tabs (HTMX-swapped into `#main`):

* **Annotate** — the labelling grid (below).
* **Dataset Health** — class distribution (LabelPredicted), human positives, times
  ruled-out, and remaining pool per class, with totals.
* **Model Health** — the current model scored against your human `LC_Label`
  positives: per-class precision / recall / F1 + support, overall accuracy /
  macro-F1, and a confusion matrix. Empty until you've labelled some positives
  (capped at 800 crops per evaluation for responsiveness).
* **Dashboard** — the Plotly **Dash** app (`dashboard.py`) mounted under FastAPI
  at `/dashboard` and embedded in an `<iframe>`: feature scatter + click-to-view
  image + binary/threshold tools. No separate server/port needed.

Run it (CPU, uses the `3dmodel` conda env which already has the deps):

```powershell
& "C:/Users/acer/anaconda3/envs/3dmodel/python.exe" annotate_server.py
```

Then open http://127.0.0.1:8051 — the Dashboard tab lives inside it, so you no
longer need to launch `dashboard.py` separately (it still runs standalone on 8050
if you want). Both write the same `LabelChecker*.csv`; the annotation app now
writes back only the `LC_` columns, preserving Dashboard-written columns like
`BioVolume`.

How it works:

1. **Target class** — pick the class `x` you want to find.
2. The **grid** (selectable **rows m × cols n**) is built to *minimise* images
   already predicted as `x`: it samples crops not yet labelled/ruled-out for `x`,
   scores them with MobileNetV3-Small, and keeps the lowest-`P(x)` ones (the
   threshold controls the cutoff). You surface the positives the model missed.
   The remaining scored candidates become a **reserve** queue for replacements.
   Images the model already predicts as *any* class with high confidence
   (`max P(any) ≥ conf_skip`, default **0.95**) are dropped from both the grid
   and the reserve — they're already "known" and not worth a label.
3. **Click** a crop that contains `x`: it is labelled `x` immediately and only
   that one tile is swapped (HTMX `outerHTML` swap) for the next reserve tile.
4. **Regenerate (none are target)** — use when no shown crop is `x`: every shown
   crop is labelled *not-x* and a fresh grid is drawn.
   **Update Model** / **Train Base Model** — run in the background (the page polls
   for status); base train bootstraps on the existing `LabelPredicted` column, so
   click it first to make predictions useful from round one.
   **Save labels** — flush to the CSV (see persistence note below).

Labels live in memory during a session and are written to the `LabelChecker*.csv`
on **Regenerate**, **Save**, after training, and on exit (atomic replace — no
full rewrite per click). New columns:

* `LC_Label`    — positive class assigned when a crop is clicked.
* `LC_Excluded` — pipe-separated classes ruled out for a crop (a "reverse
  one-hot"); pressing **Regenerate** adds `x` to every crop shown at the time.
* `LC_Seen`     — 1 once a crop has appeared in a grid.
* `LC_Probs`    — JSON `{class: prob}` of the model's full prediction for that crop
  (written when scored/shown). Hovering a tile shows the top-8 `class: prob`.

At training time a reverse-one-hot crop (exclusions but no positive label) is
turned into a single label by randomly choosing a class it has *not* been ruled
out of. The training recipe lives in `RECIPE` at the top of `annotate_model.py`;
the checkpoint is saved to `models/annotate_mobilenetv3.pt`.

> Note: the server loads the first `LabelChecker*.csv` it finds by default and
> writes labels back to it. Work on a copy if you don't want to touch originals.
