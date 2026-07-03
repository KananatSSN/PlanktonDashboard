# Build Spec: Open-Set Active Learning Pipeline for Large-Scale Image Classification

## Goal

Build a human-in-the-loop system to classify millions of unlabeled images into an
**unknown, growing number of classes**. The system must let an annotator discover
classes organically (not from a predefined taxonomy), bootstrap a classifier from a
handful of examples per class, and scale labeling via active learning + batch
"grid review" rather than one-image-at-a-time annotation.

Build this as a working system with: an embedding/indexing backend, a lightweight
classifier training loop, a metadata database, and a simple web UI for the four
workflows described below. Prioritize a working end-to-end vertical slice over
completeness — get initialization → seed classifier → grid review → active learning
loop working on a small subset (~10k images) first, then scale.

---

## 1. Architecture Overview

**Components:**

1. **Embedding service** — DINOv2 (frozen) embeds every image once. Embeddings are
   the backbone for clustering, similarity search, uncertainty scoring, and diversity
   sampling.
2. **Vector index** — approximate nearest neighbor (ANN) index over embeddings for
   fast similarity search and clustering at scale.
3. **Metadata store** — tracks per-image state: id, path, embedding_id, label,
   label_source (human/weak/pseudo), confidence, status, cluster_id, timestamps.
4. **Classifier** — a fast, cheaply-retrainable model used *during* the labeling loop
   (see §4 for why this should NOT be MobileNet trained from raw pixels), plus an
   optional final distillation into a MobileNet for deployment.
5. **UI** — three screens: cluster/class naming, grid review, and single-image
   active-learning review queue.
6. **Orchestrator** — a loop/script that ties together: sample pool → score →
   select batch → serve to UI → collect labels → retrain → repeat.

**Data flow:**
```
raw images --> DINOv2 embed --> vector index + embedding store
                                        |
                                        v
                        [Initialization: cluster + name classes]
                                        |
                                        v
                         seed labeled set (i images/class)
                                        |
                                        v
                    [train lightweight classifier head]
                                        |
                                        v
        +---------------[loop]---------------+
        |                                     |
   [grid review: high-confidence          [active learning: uncertain +
    weak labels, batch accept/reject]      diverse samples, one-by-one review]
        |                                     |
        +------------> merge into labeled set +
                                |
                                v
                    [retrain, re-score pool]
                                |
                                v
                 [periodic: novel-class discovery pass]
```

---

## 2. Data & Storage

- **Embedding storage**: store DINOv2 embeddings (ViT-B/14 or ViT-S/14 for speed) as
  float16 vectors, e.g. in a memory-mapped numpy array or a vector DB (FAISS index +
  a sidecar SQLite/Postgres table mapping row index → image_id). At millions of
  images, FAISS with an IVF-PQ or HNSW index is required for sub-second ANN queries;
  a flat index will not scale past ~1-2M vectors comfortably in RAM.
- **Metadata schema** (SQLite for prototype, Postgres for production):
  ```
  images(
    id, path, embedding_row, 
    label (nullable), label_source ENUM(human, weak, pseudo, none),
    confidence FLOAT, status ENUM(unlabeled, seed, reviewed, rejected),
    cluster_id (nullable), created_at, updated_at
  )
  classes(
    id, name, created_at, seed_count, total_labeled_count
  )
  labeling_events(
    id, image_id, class_id, action ENUM(accept, reject, relabel),
    round_number, annotator, timestamp
  )
  ```
- Keep the **pool** (unlabeled images) and **labeled set** as separate logical views
  over the same table via `status`/`label` fields — don't physically move images
  between stores.
- At million-image scale, never score the *entire* pool every round. Maintain a
  **working subset** (e.g. random 50k-200k sample refreshed periodically) that
  active learning and grid review operate on, and periodically rotate in fresh
  random images so no part of the pool is permanently ignored.

---

## 3. Step 1 — Initialization (class discovery + seeding)

Support **both** initialization modes described by the user; implement mode A first,
mode B is a nice-to-have that reuses the same similarity search infra.

### Mode A: Cluster-then-name
1. Randomly sample n images (e.g. n=5,000–20,000) from the pool.
2. Embed with DINOv2 (if not already embedded).
3. Cluster embeddings with **HDBSCAN** (not k-means) — the number of classes is
   unknown, and HDBSCAN naturally handles this plus leaves outliers unclustered
   (`-1` label) rather than forcing them into a bad cluster. Run on the raw
   embeddings or a UMAP-reduced version (UMAP to ~30-50 dims speeds up HDBSCAN
   significantly with negligible quality loss).
4. For each resulting cluster, show the annotator a grid of ~16-25 representative
   images (closest to cluster centroid, plus a few random cluster members for
   diversity of view).
5. Annotator either: (a) names the cluster as a class and selects **i** clean
   representative images as seeds, (b) merges it into an existing class, or
   (c) discards it as noise/mixed/not-a-class.
6. Unclustered (`-1`) outliers are set aside — some become singleton classes later
   via Mode B, some just remain in the pool.

### Mode B: Exemplar-then-expand
1. Annotator manually picks one image and types a class name.
2. System runs a k-NN query against the vector index (k configurable, e.g. 50) to
   retrieve visually similar images.
3. Display as a grid; annotator multi-selects the ones that truly belong, up to **i**
   seed images.
4. This is essentially a manual, single-class version of grid review (§5) — reuse
   the same grid UI component for both.

**Output of this phase**: a `classes` table with N discovered classes (N is
whatever the annotator found — do not hardcode a class count anywhere in the
system), and a seed-labeled set of `i` images per class.

---

## 4. Step 2 — Bootstrap Classifier

**Important design change from the user's original sketch**: don't train MobileNet
directly on raw pixels for the in-loop classifier. Instead:

- Train a **small MLP or linear head on top of frozen DINoV2 embeddings**
  (embeddings are already computed for indexing — reuse them). This trains in
  seconds on CPU even with 10k+ images and needs no GPU during the active learning
  loop, which matters because you'll retrain every round.
- Output layer size = current number of classes; support **adding new classes**
  without full retrain-from-scratch pain by just resizing the final layer and
  continuing training (or simply retraining the head from scratch each round —
  it's cheap enough at this scale that architectural cleverness usually isn't worth it).
- Only distill into an actual **MobileNetV3-Small** trained on raw pixels once you
  want a standalone deployable model that doesn't depend on running DINOv2 at
  inference time (e.g. for edge deployment). Treat this as a separate, later
  pipeline stage, not part of the labeling loop.
- Track per-class validation precision/recall on a small held-out labeled set so
  you can see classifier quality improve round over round.

---

## 5. Step 3 — Grid Review (confidence-based weak labeling)

Purpose: cheaply expand a class far beyond its seed set by leveraging classifier
confidence, with the human only doing fast batch accept/reject instead of per-image
labeling.

1. For a class of interest, run the current classifier over the working pool
   subset, and take the top **m²** highest-confidence *unlabeled* predictions for
   that class.
2. Render as an m×m grid (m configurable, start with m=8-10, i.e. 64-100 images).
3. Annotator behavior: **default is "these are all correct"** — they click to
   deselect/mark the wrong ones (faster than opt-in selection when precision is
   typically high at this stage). Provide a "reject all" fallback for bad batches.
4. Accepted images → labeled set with `label_source=weak`, confidence recorded.
   Rejected images → stay in pool but flagged so they aren't immediately
   re-suggested for the same class (avoid a reject-loop).
5. Repeat across classes, prioritizing classes with the fewest labeled examples so
   far (or ones the orchestrator flags as underrepresented).
6. Retrain the classifier after each grid session (or after N grids), since new
   labels shift decision boundaries.

This step is your main **throughput** mechanism — it should account for the large
majority of labels collected, with the active learning loop (§6) handling the hard
cases grid review can't resolve confidently.

---

## 6. Step 4 — Active Learning Loop (hard cases + novel class discovery)

Two sub-processes, run periodically (e.g. after every K grid-review sessions or
every retrain):

### 6a. Uncertainty + diversity sampling (refining existing classes)
1. Score the working pool subset with the current classifier; compute an
   uncertainty measure — start with **margin sampling** (difference between top-1
   and top-2 softmax scores) since it's simple and works well for multi-class;
   entropy is a reasonable alternative.
2. Take the top-K most uncertain samples (e.g. K=500).
3. Within that K, cluster by embedding (k-means with k ≈ K/10, or just greedy
   farthest-point sampling) and pick 1 representative per cluster to avoid handing
   the annotator 50 near-duplicate ambiguous images.
4. Present these one-by-one (or in a lightweight multi-select grid with class
   dropdown) for the annotator to assign the correct label or mark "none of the
   above" / "new class".
5. Add to labeled set with `label_source=human`, retrain.

### 6b. Novel class discovery (open-set handling)
Since the total number of classes is unknown and grows over time, periodically:
1. Pull all pool images where the classifier's **max softmax probability is
   below a low-confidence threshold** across *all* known classes (these are
   "doesn't fit anything well" images) — or where the DINOv2 embedding is far
   (above some distance threshold) from every known class centroid.
2. Run the same HDBSCAN clustering from §3 on just this low-confidence subset.
3. If a meaningful cluster emerges (size above some minimum), surface it to the
   annotator via the same cluster-naming UI as initialization — it's either a new
   class, or a subgroup that should be split off an existing class.
4. This step is what prevents the system from silently ignoring classes that
   weren't in the original seed set.

**Stopping/tuning criteria to implement:**
- Track labels-collected-per-annotator-minute per round; when a round's yield
  drops below some threshold relative to earlier rounds, that's a signal to either
  broaden the working pool sample or focus effort elsewhere.
- Track per-class validation precision on a held-out set; flag classes whose
  precision is dropping (may indicate label noise from grid review being too
  permissive).

---

## 7. Class Management Operations (build these into the UI/backend early)

- **Merge two classes** (annotator realizes they're duplicates)
- **Split a class** (re-cluster its labeled images, annotator names the sub-clusters)
- **Rename a class**
- **Delete/deprecate a class** and re-pool its images as unlabeled

These will be needed constantly once real data is involved — don't treat them as
optional polish.

---

## 8. Suggested Tech Stack

- **Embeddings**: DINOv2 via `torch.hub` or HuggingFace `transformers`
  (`facebook/dinov2-base` or `-small` for speed at scale). Batch inference on GPU;
  budget for this being the main compute cost at millions of images.
- **Clustering**: `hdbscan` + `umap-learn` for dimensionality reduction pre-clustering.
- **ANN index**: `faiss` (IVF-PQ or HNSW index depending on RAM budget).
- **Classifier**: PyTorch, simple `nn.Linear` or 2-layer MLP head on embeddings;
  `scikit-learn` `LogisticRegression`/`MLPClassifier` is also fine for prototyping.
- **Metadata DB**: SQLite for prototype → Postgres when concurrent annotators are needed.
- **UI**: Streamlit or Gradio for a fast prototype (grid display with click-to-toggle
  is straightforward in both); move to a custom React app only once the workflow is
  validated and multiple annotators need to work concurrently.
- **Orchestration**: plain Python scripts/CLI commands per phase is fine initially
  (`init.py`, `grid_review.py`, `active_learning_round.py`, `discover_novel.py`);
  don't over-engineer a task queue until the manual loop is proven useful.

---

## 9. Build Order (recommended milestones)

1. Embedding pipeline + FAISS index + metadata DB, tested on ~10k image subset.
2. Mode A initialization (HDBSCAN clustering + naming UI) working end-to-end.
3. Bootstrap classifier (linear head on embeddings) + validation tracking.
4. Grid review UI + accept/reject flow + retrain trigger.
5. Active learning loop (margin sampling + diversity dedup) + single-image review UI.
6. Novel class discovery pass.
7. Class management operations (merge/split/rename/delete).
8. Scale-out: move from 10k → full pool, add working-subset rotation, add FAISS
   IVF-PQ index, add Postgres if multi-annotator.
9. (Optional, later) MobileNet distillation for deployment.

Build and validate each milestone on a small subset before scaling — the clustering
and classifier quality assumptions should be checked on real data early rather than
assumed.
