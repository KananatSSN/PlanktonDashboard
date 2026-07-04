"""DB-backed annotation session (MobileNet / torchvision, no DINOv2).

Labels live in SQLite (the source of truth). The single in-loop model is a
torchvision backbone (default MobileNetV3-Small) trained on raw crops with
aggressive RandAugment. Two grid modes share one tile/reserve mechanic:

* **review** — top ``m²`` HIGHEST P(x) unlabelled predictions. Default-accept:
  click a tile to reject it; *Accept all shown* commits the rest as ``weak``.
* **hardcase** — LOWEST P(x) unlabelled (x-minimising): click to accept as human.

Training runs in the **background on a copy** of the model while annotation
continues; the fresh model is swapped in atomically when done. Retraining is
triggered manually or automatically after every ``retrain_every`` new labels.
"""

from __future__ import annotations

import atexit
import json
import random
import threading

import numpy as np

import annotate_db as db
import annotate_model as am
import annotate_settings as st
from annotate_core import array_to_base64, probs_tooltip
from annotate_data import DatasetSource, discover_datasets


def _parse_excluded(val: str) -> set[str]:
    return {p for p in (val or "").split("|") if p}


def _farthest_point_sample(vecs: np.ndarray, k: int) -> list[int]:
    """Greedy farthest-point sampling: pick k spread-out rows of ``vecs``."""
    n = len(vecs)
    if n == 0:
        return []
    k = min(k, n)
    picked = [int(np.argmax(np.linalg.norm(vecs - vecs.mean(0), axis=1)))]
    dist = np.linalg.norm(vecs - vecs[picked[0]], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(dist))
        picked.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(vecs - vecs[nxt], axis=1))
    return picked


class Session:
    """Single-user labelling session over one dataset, backed by the DB."""

    def __init__(self):
        self.conn = db.connect(check_same_thread=False)
        self.lock = threading.RLock()
        self.settings = st.load()
        self.sources: dict[str, DatasetSource] = {
            s.dataset_id: s for s in discover_datasets()}
        self.source: DatasetSource | None = None
        self.classes: list[str] = []
        self.clf: am.PlanktonClassifier | None = None

        self.target: str | None = None
        self.mode = "review"
        self.m, self.n = 3, 3
        self.conf_skip = 0.95              # trust the model when max P(any) >= this

        self.shown: list[int | None] = []
        self.srcs: list[str | None] = []
        self.ptar: list[float | None] = []
        self.probs: list[list | None] = []
        self.reserve: list[list] = []      # [[image_id, p, probs], …]
        self._crops: dict[int, np.ndarray] = {}   # crop cache for the current sample
        self.session_count = 0
        self.labels_since_train = 0
        self.status = "Pick a target class and press New grid."
        self.train = {"running": False, "msg": "", "frac": 0.0, "phase": ""}

        # active-learning queue
        self.al_queue: list[int] = []
        self.al_top: dict[int, list] = {}
        self.al_pos = 0
        self.al_labelled = 0
        self.al_status = "Press Build queue to start."

        self.manage_status = ""

        if self.sources:
            self.load_dataset(next(iter(self.sources)))
        atexit.register(self.close)

    # ── dataset / model --------------------------------------------------------
    def dataset_options(self) -> list[dict]:
        return [{"value": sid, "label": s.name} for sid, s in self.sources.items()]

    def load_dataset(self, dataset_id: str):
        src = self.sources.get(dataset_id)
        if src is None:
            self.status = f"Unknown dataset {dataset_id}"
            return
        self.source = src
        self.classes = db.class_names(self.conn)
        self._build_model()
        if self.target not in self.classes:
            self.target = self.classes[0] if self.classes else None
        self._clear_grid()
        self.status = f"Loaded {src.name} · {len(self.classes)} classes."

    def _build_model(self):
        """Construct the classifier for the current class set + settings."""
        self.clf = am.PlanktonClassifier(
            self.classes, backbone=self.settings["backbone"],
            img_size=self.settings["img_size"])
        self.clf.load()

    def _clear_grid(self):
        self.shown, self.srcs, self.ptar, self.probs, self.reserve = [], [], [], [], []
        self._crops = {}

    # ── pool scoring (MobileNet over sampled crops) ----------------------------
    def _pool_rows(self) -> list[tuple[int, str]]:
        """Eligible unlabelled images for the target (x not ruled out)."""
        rows = self.conn.execute(
            "SELECT id, item_key, excluded FROM images "
            "WHERE dataset_id = ? AND label = ''", (self.source.dataset_id,)).fetchall()
        return [(r["id"], r["item_key"]) for r in rows
                if self.target not in _parse_excluded(r["excluded"])]

    def _score_rows(self, batch) -> list[tuple[int, float, np.ndarray]]:
        """Load + score a batch of (id, item_key); cache crops for rendering."""
        ti = self.clf.class_to_idx.get(self.target)
        if ti is None or not batch:
            return []
        keys = [k for _, k in batch]
        images = self.source.load_images(keys)
        valid = [(iid, k) for iid, k in batch if k in images]
        if not valid:
            return []
        crops = [images[k] for _, k in valid]
        probs = self.clf.predict_proba(crops)
        out = []
        for i, (iid, _) in enumerate(valid):
            self._crops[iid] = crops[i]
            out.append((iid, float(probs[i, ti]), probs[i]))
        return out

    def _score_pool(self) -> list[tuple[int, float, np.ndarray]]:
        """Score one random sample of up to ``max_pool`` crops (hard-case)."""
        pool = self._pool_rows()
        if not pool:
            return []
        sample = random.sample(pool, min(self.settings["max_pool"], len(pool)))
        return self._score_rows(sample)

    def _collect_review(self, want, exclude=()) -> list[tuple[int, float, np.ndarray]]:
        """Scan the pool in chunks, keeping only crops the model **predicts as x**
        (argmax = target), until ``want`` are found or the scan budget is spent.
        Sorted highest P(x) first, so grid + reserve are all accept-able candidates.
        """
        ti = self.clf.class_to_idx.get(self.target)
        if ti is None:
            return []
        exclude = set(exclude)
        pool = [r for r in self._pool_rows() if r[0] not in exclude]
        random.shuffle(pool)
        chunk = max(self.settings["max_pool"], 1)
        budget = max(chunk * 4, 800)          # cap crops scored per build
        found, scanned = [], 0
        for i in range(0, len(pool), chunk):
            found += [s for s in self._score_rows(pool[i:i + chunk])
                      if int(np.argmax(s[2])) == ti]
            scanned += chunk
            if len(found) >= want or scanned >= budget:
                break
        found.sort(key=lambda s: -s[1])
        return found

    def _render(self, image_id: int) -> str | None:
        crop = self._crops.get(image_id)
        if crop is None:
            row = self.conn.execute("SELECT item_key FROM images WHERE id = ?",
                                    (image_id,)).fetchone()
            crop = self.source.load_image(row["item_key"]) if row else None
        return array_to_base64(crop) if crop is not None else None

    def _persist_probs(self, image_id: int, vec):
        self.conn.execute("UPDATE images SET probs = ?, seen = 1 WHERE id = ?",
                          (json.dumps({c: round(float(p), 4)
                                       for c, p in zip(self.classes, vec)}), image_id))

    # ── grid build -------------------------------------------------------------
    def set_params(self, target=None, mode=None, m=None, n=None, conf_skip=None):
        cap = self.settings["grid_max"]
        if target:
            self.target = target
        if mode in ("review", "hardcase"):
            self.mode = mode
        if m is not None:
            self.m = max(1, min(cap, int(m)))
        if n is not None:
            self.n = max(1, min(cap, int(n)))
        if conf_skip is not None:
            self.conf_skip = max(0.0, min(1.0, float(conf_skip)))

    def _rank_hardcase(self, scored):
        # hide crops the model already predicts confidently (trusted), then show
        # the least-confident-for-x first
        ranked = [s for s in scored if max(s[2]) < self.conf_skip]
        ranked.sort(key=lambda s: s[1])
        note = f"{len(scored) - len(ranked)} trusted (P(any)≥{self.conf_skip:g}) hidden"
        return ranked, note

    def make_grid(self):
        if self.train["running"] and not (self.clf and self.clf.trained):
            self.status = "Base training… please wait before building the grid."
            return
        if not self.clf or not self.clf.trained:
            self.status = "Model not trained yet — press Train Base Model first."
            self._clear_grid()
            return
        self._crops = {}
        cells = self.m * self.n
        if self.mode == "review":
            ranked = self._collect_review(want=cells + 30)
            note = f"{len(ranked)} predicted {self.target}"
            if not ranked:
                self._clear_grid()
                self.status = (f"Model predicts no unlabelled crops as "
                               f"{self.target} yet — try Retrain or another class.")
                return
        else:
            scored = self._score_pool()
            if not scored:
                self._clear_grid()
                self.status = f"No unlabelled pool left for {self.target}."
                return
            ranked, note = self._rank_hardcase(scored)
        head, tail = ranked[:cells], ranked[cells:]
        self.shown, self.srcs, self.ptar, self.probs = [], [], [], []
        for iid, p, vec in head:
            self.shown.append(iid)
            self.srcs.append(self._render(iid))
            self.ptar.append(p)
            self.probs.append(vec.tolist())
            self._persist_probs(iid, vec)
        pad = cells - len(self.shown)
        self.shown += [None] * pad
        self.srcs += [None] * pad
        self.ptar += [None] * pad
        self.probs += [None] * pad
        self.reserve = [[iid, p, vec.tolist()] for iid, p, vec in tail]
        self.conn.commit()
        verb = "accept the rest" if self.mode == "review" else "click the positives"
        self.status = (f"[{self.mode}] {self.target}: {note} · grid "
                       f"{len([s for s in self.shown if s is not None])}/{cells} · "
                       f"reserve {len(self.reserve)} · {verb}")

    def _next_reserve(self):
        shown = {s for s in self.shown if s is not None}
        refills = 0
        while True:
            while self.reserve:
                iid, p, vec = self.reserve.pop(0)
                if iid in shown:
                    continue
                src = self._render(iid)
                if src is None:
                    continue
                return iid, src, p, vec
            # reserve empty: scan more of the pool (unless a base train is running)
            if self.train["running"] or refills >= 2:
                return None, None, None, None
            refills += 1
            if self.mode == "review":
                fresh = self._collect_review(self.m * self.n, exclude=shown)
            else:
                fresh = [(i, p, v) for i, p, v in
                         self._rank_hardcase(self._score_pool())[0] if i not in shown]
            if not fresh:
                return None, None, None, None
            self.reserve = [[i, p, v.tolist()] for i, p, v in fresh]

    # ── tile interactions ------------------------------------------------------
    def click(self, cell: int) -> bool:
        if cell < 0 or cell >= len(self.shown) or self.shown[cell] is None:
            return False
        image_id = self.shown[cell]
        if self.mode == "hardcase":
            db.set_label(self.conn, image_id, self.target, "human",
                         confidence=self.ptar[cell], action="accept")
            self.session_count += 1
            self._note_labels(1)
            msg = f"Labelled {self.target} (+{self.session_count})"
        else:
            db.add_exclusion(self.conn, image_id, self.target)
            msg = f"Rejected — not {self.target}"
        self.conn.commit()
        iid, src, p, vec = self._next_reserve()
        self.shown[cell], self.srcs[cell] = iid, src
        self.ptar[cell], self.probs[cell] = p, vec
        if iid is not None:
            self._persist_probs(iid, np.array(vec))
            self.conn.commit()
        self.status = f"{msg} · reserve {len(self.reserve)}"
        return True

    def accept_all(self):
        if self.mode != "review":
            return
        ids = [(iid, self.ptar[i]) for i, iid in enumerate(self.shown)
               if iid is not None]
        for iid, p in ids:
            db.set_label(self.conn, iid, self.target, "weak", confidence=p,
                         action="accept")
        self.conn.commit()
        self.session_count += len(ids)
        self.status = f"Accepted {len(ids)} as weak {self.target} (+{self.session_count})."
        self._note_labels(len(ids))
        self.make_grid()

    def reject_all(self):
        ids = [iid for iid in self.shown if iid is not None]
        for iid in ids:
            db.add_exclusion(self.conn, iid, self.target)
        self.conn.commit()
        self.make_grid()
        self.status = f"Rejected {len(ids)} for {self.target}. " + self.status

    # ── training (background, train-on-copy) -----------------------------------
    def _note_labels(self, n: int):
        """Count new labels; auto-retrain once the threshold is reached."""
        self.labels_since_train += n
        every = self.settings.get("retrain_every", 0)
        if every and self.labels_since_train >= every and not self.train["running"]:
            self.start_retrain(auto=True)

    def _collect(self, sql, params) -> tuple[list[str], list[str]]:
        rows = self.conn.execute(sql, params).fetchall()
        return [r["item_key"] for r in rows], [r["label"] for r in rows]

    def _start_training(self, keys, labels, kind, epochs):
        if self.train["running"] or not keys:
            if not keys:
                self.train["msg"] = f"{kind}: nothing to train on."
            return
        classes = list(self.classes)
        cfg = dict(self.settings)
        source = self.source

        def on_progress(frac, ep, loss):
            self.train.update(
                frac=frac, phase=f"{kind}: epoch {ep}/{epochs} · loss {loss:.3f}")

        def job():
            try:
                self.train.update(phase=f"{kind}: loading crops…")
                images = source.load_images(keys)
                crops = [images[k] for k in keys if k in images]
                labs = [lab for k, lab in zip(keys, labels) if k in images]
                known = set(classes)
                keep = [i for i, l in enumerate(labs) if l in known]
                crops, labs = [crops[i] for i in keep], [labs[i] for i in keep]
                if not crops:
                    self.train.update(running=False, frac=0.0, phase="",
                                      msg=f"{kind}: nothing to train on.")
                    return
                clf = am.PlanktonClassifier(classes, backbone=cfg["backbone"],
                                            img_size=cfg["img_size"])
                res = clf.fit(crops, labs, epochs=epochs, lr=cfg["lr"],
                              batch_size=cfg["batch_size"],
                              weight_decay=cfg["weight_decay"],
                              freeze=cfg["freeze_backbone"],
                              randaug_ops=cfg["randaug_ops"],
                              randaug_magnitude=cfg["randaug_magnitude"],
                              progress=on_progress)
                clf.save()
                with self.lock:
                    self.clf = clf                 # atomic swap; scoring uses new model
                    self.labels_since_train = 0
                self.train.update(
                    running=False, frac=1.0, phase="",
                    msg=f"{kind} done: {res['n']} crops, {res['epochs']} epochs, "
                        f"loss {res['loss']:.3f}. New model live.")
            except Exception as exc:
                self.train.update(running=False, frac=0.0, phase="",
                                  msg=f"{kind} failed: {exc}")

        self.train.update(running=True, frac=0.0, phase=f"{kind}: starting…",
                          msg=f"{kind} running in background…")
        threading.Thread(target=job, daemon=True).start()

    def start_retrain(self, auto=False):
        if self.train["running"] or self.source is None:
            return
        keys, labels = self._collect(
            "SELECT item_key, label FROM images WHERE dataset_id = ? "
            "AND label != '' AND label_source IN ('human','weak')",
            (self.source.dataset_id,))
        self._start_training(keys, labels, "Auto-retrain" if auto else "Retrain",
                             self.settings["finetune_epochs"])

    def start_base(self):
        if self.train["running"] or self.source is None:
            return
        keys, labels = self._collect(
            "SELECT item_key, predicted AS label FROM images "
            "WHERE dataset_id = ? AND predicted != ''", (self.source.dataset_id,))
        cap = self.settings["base_max_samples"]
        if len(keys) > cap:
            idx = random.sample(range(len(keys)), cap)
            keys, labels = [keys[i] for i in idx], [labels[i] for i in idx]
        self._start_training(keys, labels, "Base train", self.settings["base_epochs"])

    # ── active-learning queue --------------------------------------------------
    def build_al_queue(self):
        if self.train["running"]:
            self.al_status = "Training… wait, then Build queue."
            return
        if not self.clf or not self.clf.trained:
            self.al_status = "Model not trained — press Train Base Model first."
            return
        rows = self.conn.execute(
            "SELECT id, item_key FROM images WHERE dataset_id = ? AND label = ''",
            (self.source.dataset_id,)).fetchall()
        if not rows:
            self.al_queue, self.al_pos = [], 0
            self.al_status = "No unlabelled images left."
            return
        sample = random.sample(rows, min(self.settings["al_sample"], len(rows)))
        keys = [r["item_key"] for r in sample]
        images = self.source.load_images(keys)
        valid = [r for r in sample if r["item_key"] in images]
        crops = [images[r["item_key"]] for r in valid]
        ids = np.array([r["id"] for r in valid])
        probs = self.clf.predict_proba(crops)
        part = np.sort(probs, axis=1)
        margin = part[:, -1] - (part[:, -2] if probs.shape[1] > 1 else 0)
        k = min(self.settings["al_k_uncertain"], len(ids))
        unc = np.argsort(margin)[:k]
        feats = self.clf.features([crops[i] for i in unc])
        pick = _farthest_point_sample(feats, self.settings["al_queue_size"])
        chosen = unc[pick]
        self.al_queue = ids[chosen].tolist()
        self.al_top = {}
        for j, iid in zip(chosen, self.al_queue):
            p = probs[j]
            self.al_top[iid] = [{"cls": self.clf.classes[t], "p": round(float(p[t]), 3)}
                                for t in np.argsort(-p)[:5]]
        self.al_pos, self.al_labelled = 0, 0
        self.al_status = (f"Queue of {len(self.al_queue)} uncertain+diverse images "
                          f"(margin {margin[unc].min():.3f}–{margin[chosen].max():.3f}).")

    def _al_view(self) -> dict:
        if self.al_pos >= len(self.al_queue):
            return {"done": True, "labelled": self.al_labelled,
                    "total": len(self.al_queue), "status": self.al_status}
        image_id = self.al_queue[self.al_pos]
        row = self.conn.execute(
            "SELECT item_key, predicted FROM images WHERE id = ?", (image_id,)).fetchone()
        img = self.source.load_image(row["item_key"]) if row else None
        return {"done": False, "image_id": image_id,
                "src": array_to_base64(img) if img is not None else None,
                "predicted": row["predicted"] if row else "",
                "top": self.al_top.get(image_id, []),
                "pos": self.al_pos + 1, "total": len(self.al_queue),
                "labelled": self.al_labelled, "status": self.al_status}

    def _al_advance(self):
        self.al_pos += 1
        if self.al_pos >= len(self.al_queue):
            self.al_status = (f"Queue complete — labelled {self.al_labelled} of "
                              f"{len(self.al_queue)}.")

    def al_assign(self, class_name: str):
        if self.al_pos >= len(self.al_queue) or not class_name:
            return
        db.set_label(self.conn, self.al_queue[self.al_pos], class_name, "human",
                     action="relabel")
        self.conn.commit()
        self.al_labelled += 1
        self._note_labels(1)
        self._al_advance()

    def al_new_class(self, name: str):
        name = (name or "").strip()
        if not name or self.al_pos >= len(self.al_queue):
            return
        db.ensure_classes(self.conn, [name], origin="manual")
        self.classes = db.class_names(self.conn)
        self.al_assign(name)
        self.al_status = f"New class '{name}' created. " + self.al_status

    def al_skip(self):
        if self.al_pos < len(self.al_queue):
            self.conn.execute("UPDATE images SET seen = 1 WHERE id = ?",
                              (self.al_queue[self.al_pos],))
            self.conn.commit()
            self._al_advance()

    def al_view(self) -> dict:
        return self._al_view()

    # ── class management -------------------------------------------------------
    def class_table(self) -> list[dict]:
        return db.class_table(self.conn)

    def _refresh_classes(self, msg: str):
        self.classes = db.class_names(self.conn)
        if self.target not in self.classes:
            self.target = self.classes[0] if self.classes else None
        self._build_model()          # class set changed -> rebuild + reload ckpt
        self.manage_status = msg

    def rename_class(self, old: str, new: str):
        r = db.rename_class(self.conn, old, new)
        verb = "Merged" if r["action"] == "merge" else "Renamed"
        self._refresh_classes(f"{verb} '{old}' → '{r.get('to', new)}' ({r['n']} images).")

    def merge_classes(self, src: str, dst: str):
        r = db.merge_classes(self.conn, src, dst)
        self._refresh_classes(f"Merged '{src}' → '{dst}' ({r['n']} images).")

    def delete_class(self, name: str):
        r = db.delete_class(self.conn, name)
        self._refresh_classes(f"Deleted '{name}', re-pooled {r['n']} images.")

    def manage_status_str(self) -> str:
        return self.manage_status

    # ── settings ---------------------------------------------------------------
    def get_settings(self) -> dict:
        return self.settings

    def update_settings(self, raw: dict) -> str:
        new = st.coerce(raw)
        rebuild = (new["backbone"] != self.settings["backbone"]
                   or new["img_size"] != self.settings["img_size"])
        self.settings = new
        st.save(new)
        if rebuild:
            self._build_model()
            return (f"Settings saved. Backbone/size changed → model reset to "
                    f"{new['backbone']}; press Train Base Model.")
        return "Settings saved."

    # ── persistence / views ----------------------------------------------------
    def save(self) -> str:
        if self.source is None:
            return "No dataset."
        if not self.source.supports_export:
            return f"{self.source.name}: no CSV export for this source type."
        n = db.export_source(self.conn, self.source)
        return f"Exported {n} rows to {self.source.csv_path.name}."

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def cells_view(self):
        out = []
        for i, (iid, s, p, vec) in enumerate(
                zip(self.shown, self.srcs, self.ptar, self.probs)):
            tip = (probs_tooltip(self.classes, vec, iid) if vec
                   else (f"id {iid}" if iid is not None else ""))
            out.append({"i": i, "src": s, "row": iid, "p": p, "tip": tip})
        return out

    def model_status(self) -> str:
        if not self.classes:
            return "No dataset loaded."
        trained = "trained" if (self.clf and self.clf.trained) else "untrained"
        counts = self.conn.execute(
            "SELECT SUM(label != '') AS pos, SUM(label_source='human') AS human, "
            "SUM(label_source='weak') AS weak FROM images WHERE dataset_id = ?",
            (self.source.dataset_id,)).fetchone() if self.source else None
        lab = (f"labelled {counts['pos'] or 0} (human {counts['human'] or 0}, "
               f"weak {counts['weak'] or 0})" if counts else "")
        pend = (f" · {self.labels_since_train}/{self.settings['retrain_every']} to retrain"
                if self.settings.get("retrain_every") else "")
        return (f"{self.settings['backbone']} · {len(self.classes)} classes · "
                f"{trained} · {lab}{pend}")

    def dataset_health(self) -> dict:
        return db_dataset_health(self.conn, self.source, self.classes)

    def model_health(self) -> dict:
        if self.train["running"]:
            return {"training": True}
        return db_model_health(self.conn, self.source, self.clf)


# ── health metrics ────────────────────────────────────────────────────────────────

def db_dataset_health(conn, source, classes) -> dict:
    if source is None:
        return {"summary": {}, "per_class": []}
    did = source.dataset_id
    total = conn.execute("SELECT COUNT(*) FROM images WHERE dataset_id = ?",
                         (did,)).fetchone()[0]
    rows = conn.execute("SELECT predicted, label, excluded, seen FROM images "
                        "WHERE dataset_id = ?", (did,)).fetchall()
    from collections import Counter
    pred_c, pos_c, excl_c, excl_unlab = Counter(), Counter(), Counter(), Counter()
    seen = reverse = labelled = n_unlab = 0
    for r in rows:
        if r["predicted"]:
            pred_c[r["predicted"]] += 1
        if r["label"]:
            pos_c[r["label"]] += 1
            labelled += 1
        else:
            n_unlab += 1
            if r["excluded"]:
                reverse += 1
        for c in _parse_excluded(r["excluded"]):
            excl_c[c] += 1
            if not r["label"]:
                excl_unlab[c] += 1
        if r["seen"]:
            seen += 1
    per = [{"cls": c, "pred": pred_c.get(c, 0), "pos": pos_c.get(c, 0),
            "ruled_out": excl_c.get(c, 0),
            "pool": n_unlab - excl_unlab.get(c, 0)} for c in classes]
    per.sort(key=lambda d: -d["pred"])
    return {"summary": {"total": total, "classes": len(classes), "labelled": labelled,
                        "reverse": reverse, "seen": seen, "untouched": total - seen},
            "per_class": per}


EVAL_MAX = 600   # cap crops scored for Model Health so the request stays responsive


def db_model_health(conn, source, clf) -> dict:
    """Per-class precision/recall/F1 of the model vs human labels (loads crops)."""
    if source is None or clf is None or not clf.trained:
        return {"n": 0, "total": 0}
    rows = conn.execute(
        "SELECT item_key, label FROM images WHERE dataset_id = ? AND label != '' "
        "AND label_source = 'human'", (source.dataset_id,)).fetchall()
    total = len(rows)
    if total == 0:
        return {"n": 0, "total": 0}
    sampled = total > EVAL_MAX
    if sampled:
        rows = random.sample(rows, EVAL_MAX)
    images = source.load_images([r["item_key"] for r in rows])
    pairs = [(r["label"], images[r["item_key"]]) for r in rows if r["item_key"] in images]
    if not pairs:
        return {"n": 0, "total": total}
    true = [t for t, _ in pairs]
    pred = [clf.classes[i] for i in clf.predict_proba([c for _, c in pairs]).argmax(1)]

    from collections import Counter, defaultdict
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
        per.append({"cls": c, "support": support[c], "precision": prec,
                    "recall": rec, "f1": f1})
    macro = [x["f1"] for x in per if x["f1"] is not None]
    return {"n": len(true), "total": total, "sampled": sampled,
            "accuracy": correct / len(true),
            "macro_f1": (sum(macro) / len(macro)) if macro else None,
            "per_class": per, "present": sorted(set(true) | set(pred)),
            "confusion": {t: dict(conf[t]) for t in conf}}
