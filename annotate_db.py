"""SQLite metadata store for the annotation pipeline (spec §2).

The DB is the *working* source of truth for labels; the LabelChecker CSVs
remain both the import source (existing LC_ columns are migrated in) and an
export target (``export_source`` writes LC_ columns back), so the DB can
always be rebuilt from the CSVs and vice versa.

Identity: images are keyed on ``(dataset_id, item_key)`` as defined by
``annotate_data.DatasetSource`` — no absolute paths, no DataFrame row numbers.

Run ``python annotate_db.py sync`` to discover datasets under the data root,
register them, pull in new images, and migrate any existing LC_ labels.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from annotate_data import (DatasetSource, LabelState, data_root,
                           discover_datasets)

DB_PATH_ENV = "PLANKTON_DB"
DEFAULT_DB = "annotate.db"

LABEL_SOURCES = ("none", "human", "weak", "pseudo")
STATUSES = ("unlabeled", "seed", "reviewed", "rejected")
ACTIONS = ("accept", "reject", "relabel", "exclude", "migrate")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,          -- .dataset_id marker value
    name          TEXT NOT NULL,             -- folder name (informational)
    source_type   TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    last_synced_at TEXT
);
CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY,
    dataset_id    TEXT NOT NULL REFERENCES datasets(id),
    item_key      TEXT NOT NULL,
    predicted     TEXT NOT NULL DEFAULT '',  -- source's initial label
    label         TEXT NOT NULL DEFAULT '',
    label_source  TEXT NOT NULL DEFAULT 'none'
                  CHECK (label_source IN ('none','human','weak','pseudo')),
    confidence    REAL,
    status        TEXT NOT NULL DEFAULT 'unlabeled'
                  CHECK (status IN ('unlabeled','seed','reviewed','rejected')),
    excluded      TEXT NOT NULL DEFAULT '',  -- pipe-separated ruled-out classes
    seen          INTEGER NOT NULL DEFAULT 0,
    probs         TEXT NOT NULL DEFAULT '',  -- JSON {class: prob}
    cluster_id    INTEGER,
    embedding_row INTEGER,                   -- legacy, unused
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    UNIQUE (dataset_id, item_key)
);
CREATE INDEX IF NOT EXISTS idx_images_label   ON images(label);
CREATE INDEX IF NOT EXISTS idx_images_dataset ON images(dataset_id);
CREATE TABLE IF NOT EXISTS classes (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    origin     TEXT NOT NULL DEFAULT 'seeded',   -- seeded|discovered|manual
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS labeling_events (
    id         INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL REFERENCES images(id),
    class_name TEXT NOT NULL DEFAULT '',
    action     TEXT NOT NULL
               CHECK (action IN ('accept','reject','relabel','exclude','migrate')),
    round      INTEGER NOT NULL DEFAULT 0,
    annotator  TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    return Path(os.environ.get(DB_PATH_ENV, DEFAULT_DB))


def connect(path: Path | str | None = None,
            check_same_thread: bool = True) -> sqlite3.Connection:
    """Open the DB. ``check_same_thread=False`` lets the FastAPI threadpool
    share one connection; callers must serialise access with a lock."""
    conn = sqlite3.connect(str(path or db_path()),
                           check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA)
    return conn


# ── sync: source -> DB ───────────────────────────────────────────────────────────

def sync_source(conn: sqlite3.Connection, source: DatasetSource) -> dict:
    """Register the dataset and upsert its items. New items are inserted;
    existing rows keep their label state (only ``predicted`` is refreshed)."""
    now = _now()
    conn.execute(
        """INSERT INTO datasets (id, name, source_type, registered_at, last_synced_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               name = excluded.name, source_type = excluded.source_type,
               last_synced_at = excluded.last_synced_at""",
        (source.dataset_id, source.name, source.source_type, now, now))

    items = source.list_items()
    known = {r["item_key"] for r in conn.execute(
        "SELECT item_key FROM images WHERE dataset_id = ?", (source.dataset_id,))}
    new = 0
    for it in items:
        pred = str(it.meta.get("predicted", "") or "")
        if it.item_key in known:
            conn.execute(
                """UPDATE images SET predicted = ?, updated_at = ?
                   WHERE dataset_id = ? AND item_key = ? AND predicted != ?""",
                (pred, now, source.dataset_id, it.item_key, pred))
        else:
            conn.execute(
                """INSERT INTO images (dataset_id, item_key, predicted,
                                       created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (source.dataset_id, it.item_key, pred, now, now))
            new += 1
    ensure_classes(conn, source.initial_classes(), origin="seeded")
    conn.commit()
    missing = len(known) - len({i.item_key for i in items} & known)
    return {"dataset": source.name, "items": len(items), "new": new,
            "missing_from_source": missing}


def ensure_classes(conn, names, origin="manual") -> int:
    added = 0
    for name in names:
        name = (name or "").strip()
        if not name:
            continue
        cur = conn.execute(
            "INSERT OR IGNORE INTO classes (name, origin, created_at) VALUES (?, ?, ?)",
            (name, origin, _now()))
        added += cur.rowcount
    return added


# ── migration: existing LC_ columns -> DB ────────────────────────────────────────

def migrate_lc(conn: sqlite3.Connection, source: DatasetSource) -> dict:
    """Import LC_ label state from the source's metadata (idempotent: only
    rows still untouched in the DB are filled)."""
    now, labels, exclusions = _now(), 0, 0
    for it in source.list_items():
        lc_label = str(it.meta.get("lc_label", "") or "")
        lc_excluded = str(it.meta.get("lc_excluded", "") or "")
        lc_seen = int(it.meta.get("lc_seen", 0) or 0)
        lc_probs = str(it.meta.get("lc_probs", "") or "")
        if not (lc_label or lc_excluded or lc_seen):
            continue
        row = conn.execute(
            """SELECT id, label, excluded, seen FROM images
               WHERE dataset_id = ? AND item_key = ?""",
            (source.dataset_id, it.item_key)).fetchone()
        if row is None or row["label"] or row["excluded"] or row["seen"]:
            continue                       # already synced or touched in DB
        status = "reviewed" if lc_label else "unlabeled"
        src = "human" if lc_label else "none"
        conn.execute(
            """UPDATE images SET label = ?, label_source = ?, status = ?,
                   excluded = ?, seen = ?, probs = ?, updated_at = ?
               WHERE id = ?""",
            (lc_label, src, status, lc_excluded, lc_seen, lc_probs, now, row["id"]))
        conn.execute(
            """INSERT INTO labeling_events (image_id, class_name, action,
                                            annotator, created_at)
               VALUES (?, ?, 'migrate', 'lc-import', ?)""",
            (row["id"], lc_label, now))
        labels += bool(lc_label)
        exclusions += bool(lc_excluded)
    conn.commit()
    return {"dataset": source.name, "labels": labels, "exclusions": exclusions}


# ── export: DB -> source native format ───────────────────────────────────────────

def export_source(conn: sqlite3.Connection, source: DatasetSource) -> int:
    if not source.supports_export:
        return 0
    labels = {
        r["item_key"]: LabelState(label=r["label"], excluded=r["excluded"],
                                  seen=r["seen"], probs=r["probs"])
        for r in conn.execute(
            """SELECT item_key, label, excluded, seen, probs FROM images
               WHERE dataset_id = ? AND (label != '' OR excluded != '' OR seen != 0)""",
            (source.dataset_id,))
    }
    return source.export_labels(labels) if labels else 0


# ── label operations (used by the Session in later phases) ───────────────────────

def set_label(conn, image_id: int, class_name: str, label_source: str = "human",
              confidence: float | None = None, action: str = "accept",
              annotator: str = "", round_no: int = 0):
    conn.execute(
        """UPDATE images SET label = ?, label_source = ?, status = 'reviewed',
               confidence = ?, seen = 1, updated_at = ? WHERE id = ?""",
        (class_name, label_source, confidence, _now(), image_id))
    conn.execute(
        """INSERT INTO labeling_events (image_id, class_name, action, round,
                                        annotator, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (image_id, class_name, action, round_no, annotator, _now()))


def add_exclusion(conn, image_id: int, class_name: str, round_no: int = 0):
    row = conn.execute("SELECT excluded FROM images WHERE id = ?",
                       (image_id,)).fetchone()
    ex = {p for p in (row["excluded"] or "").split("|") if p}
    if class_name in ex:
        return
    ex.add(class_name)
    conn.execute("UPDATE images SET excluded = ?, seen = 1, updated_at = ? WHERE id = ?",
                 ("|".join(sorted(ex)), _now(), image_id))
    conn.execute(
        """INSERT INTO labeling_events (image_id, class_name, action, round, created_at)
           VALUES (?, ?, 'exclude', ?, ?)""",
        (image_id, class_name, round_no, _now()))


def class_names(conn) -> list[str]:
    return [r["name"] for r in
            conn.execute("SELECT name FROM classes ORDER BY name")]


def class_table(conn) -> list[dict]:
    """Every class with its current labelled count (across all datasets)."""
    counts = {r["label"]: r["n"] for r in conn.execute(
        "SELECT label, COUNT(*) n FROM images WHERE label != '' GROUP BY label")}
    return [{"name": r["name"], "origin": r["origin"],
             "count": counts.get(r["name"], 0)}
            for r in conn.execute(
                "SELECT name, origin FROM classes ORDER BY name")]


# ── class management (merge / rename / delete-and-repool) ─────────────────────────

def _log(conn, image_id, class_name, action, annotator="manage"):
    conn.execute(
        """INSERT INTO labeling_events (image_id, class_name, action, annotator,
                                        created_at) VALUES (?, ?, ?, ?, ?)""",
        (image_id, class_name, action, annotator, _now()))


def _remap_excluded(conn, old: str, new: str | None):
    """Rename (new given) or drop (new=None) ``old`` inside reverse-one-hot sets."""
    for r in conn.execute(
            "SELECT id, excluded FROM images WHERE excluded != ''").fetchall():
        parts = {p for p in r["excluded"].split("|") if p}
        if old not in parts:
            continue
        parts.discard(old)
        if new:
            parts.add(new)
        conn.execute("UPDATE images SET excluded = ? WHERE id = ?",
                     ("|".join(sorted(parts)), r["id"]))


def rename_class(conn, old: str, new: str) -> dict:
    new = (new or "").strip()
    if not new or new == old:
        return {"action": "rename", "n": 0}
    if conn.execute("SELECT 1 FROM classes WHERE name = ?", (new,)).fetchone():
        return merge_classes(conn, old, new)      # target exists -> merge
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM images WHERE label = ?", (old,))]
    conn.execute("UPDATE classes SET name = ? WHERE name = ?", (new, old))
    conn.execute("UPDATE images SET label = ?, updated_at = ? WHERE label = ?",
                 (new, _now(), old))
    for iid in ids:
        _log(conn, iid, new, "relabel")
    _remap_excluded(conn, old, new)
    conn.commit()
    return {"action": "rename", "n": len(ids), "from": old, "to": new}


def merge_classes(conn, src: str, dst: str) -> dict:
    if src == dst:
        return {"action": "merge", "n": 0}
    ensure_classes(conn, [dst])
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM images WHERE label = ?", (src,))]
    conn.execute("UPDATE images SET label = ?, updated_at = ? WHERE label = ?",
                 (dst, _now(), src))
    for iid in ids:
        _log(conn, iid, dst, "relabel")
    conn.execute("DELETE FROM classes WHERE name = ?", (src,))
    _remap_excluded(conn, src, dst)
    conn.commit()
    return {"action": "merge", "n": len(ids), "from": src, "to": dst}


def delete_class(conn, name: str) -> dict:
    """Delete a class and re-pool its images as unlabelled."""
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM images WHERE label = ?", (name,))]
    conn.execute(
        """UPDATE images SET label = '', label_source = 'none',
               status = 'unlabeled', confidence = NULL, updated_at = ?
           WHERE label = ?""", (_now(), name))
    for iid in ids:
        _log(conn, iid, "", "relabel")
    conn.execute("DELETE FROM classes WHERE name = ?", (name,))
    _remap_excluded(conn, name, None)
    conn.commit()
    return {"action": "delete", "n": len(ids), "name": name}


# ── status / CLI ─────────────────────────────────────────────────────────────────

def summary(conn) -> str:
    lines = []
    for d in conn.execute("SELECT * FROM datasets ORDER BY name"):
        c = conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(label != '') AS labelled,
                      SUM(excluded != '') AS excl,
                      SUM(seen) AS seen
               FROM images WHERE dataset_id = ?""", (d["id"],)).fetchone()
        lines.append(f"  {d['name']} [{d['source_type']}] id={d['id'][:8]}… : "
                     f"{c['total']} images · {c['labelled'] or 0} labelled · "
                     f"{c['excl'] or 0} with exclusions · {c['seen'] or 0} seen")
    n_cls = conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
    n_ev = conn.execute("SELECT COUNT(*) FROM labeling_events").fetchone()[0]
    lines.append(f"  classes: {n_cls} · labeling events: {n_ev}")
    return "\n".join(lines) if lines else "  (empty)"


def cmd_sync():
    sources = discover_datasets()
    if not sources:
        print(f"No datasets found under {data_root().resolve()}")
        return
    with connect() as conn:
        for src in sources:
            s = sync_source(conn, src)
            m = migrate_lc(conn, src)
            print(f"synced {s['dataset']}: {s['items']} items ({s['new']} new, "
                  f"{s['missing_from_source']} missing from source) · migrated "
                  f"{m['labels']} labels, {m['exclusions']} exclusion rows")
        print("\nDB status:\n" + summary(conn))


def cmd_export():
    with connect() as conn:
        for src in discover_datasets():
            n = export_source(conn, src)
            print(f"exported {src.name}: {n} rows written"
                  if src.supports_export else f"skipped {src.name}: no export")


def cmd_status():
    with connect() as conn:
        print(f"DB: {db_path().resolve()}\n" + summary(conn))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sync"
    {"sync": cmd_sync, "export": cmd_export, "status": cmd_status}.get(
        cmd, lambda: print(f"usage: python annotate_db.py [sync|export|status]"))()
