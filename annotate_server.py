"""FastAPI + HTMX plankton annotation tool.

reCAPTCHA-style active labelling: pick a target class, click the tiles that
contain it (each click labels it and HTMX swaps in a fresh tile — only that one
DOM node changes), and fine-tune the model on what you record.

Run (CPU, 3dmodel conda env which has torch + fastapi):
    & "C:/Users/acer/anaconda3/envs/3dmodel/python.exe" annotate_server.py
Then open http://127.0.0.1:8051
"""

from __future__ import annotations

import os

import uvicorn
from a2wsgi import WSGIMiddleware
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import annotate_core as core

DASH_PREFIX = "/dashboard/"

app = FastAPI(title="Plankton Annotator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Mount the existing Plotly Dash dashboard (dashboard.py) as a sub-app. The env
# var must be set before importing dashboard so Dash builds asset/callback URLs
# under /dashboard/. The Dashboard tab embeds it in an <iframe>.
os.environ["DASH_URL_PREFIX"] = DASH_PREFIX
import dashboard  # noqa: E402  (import after setting DASH_URL_PREFIX)
app.mount("/dashboard", WSGIMiddleware(dashboard.app.server))

S = core.Session()
_csvs = core.find_csv_files()
if _csvs:
    S.load_dataset(str(_csvs[0]))


def _datasets():
    return [{"label": str(p.relative_to(core.DATA_DIR)), "value": str(p)}
            for p in core.find_csv_files()]


def _ctx(request: Request) -> dict:
    return {
        "request": request,
        "datasets": _datasets(),
        "csv_path": S.csv_path,
        "targets": S.classes,
        "target": S.target,
        "m": S.m, "n": S.n, "threshold": S.threshold, "conf_skip": S.conf_skip,
        "cells": S.cells_view(),
        "status": S.status,
        "model_status": S.model_status(),
        "train": S.train,
        "active": "annotate",
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", _ctx(request))


# ── tabs ─────────────────────────────────────────────────────────────────────--

@app.get("/tab/annotate", response_class=HTMLResponse)
def tab_annotate(request: Request):
    ctx = _ctx(request)
    ctx["active"] = "annotate"
    return templates.TemplateResponse(request, "_tab_annotate.html", ctx)


@app.get("/tab/dataset", response_class=HTMLResponse)
def tab_dataset(request: Request):
    ctx = _ctx(request)
    ctx["active"] = "dataset"
    ctx["health"] = S.dataset_health()
    return templates.TemplateResponse(request, "_tab_dataset.html", ctx)


@app.get("/tab/model", response_class=HTMLResponse)
def tab_model(request: Request):
    ctx = _ctx(request)
    ctx["active"] = "model"
    ctx["health"] = S.model_health()
    return templates.TemplateResponse(request, "_tab_model.html", ctx)


@app.get("/tab/dashboard", response_class=HTMLResponse)
def tab_dashboard(request: Request):
    ctx = _ctx(request)
    ctx["active"] = "dashboard"
    ctx["dash_url"] = DASH_PREFIX
    return templates.TemplateResponse(request, "_tab_dashboard.html", ctx)


@app.post("/dataset", response_class=HTMLResponse)
def set_dataset(request: Request, dataset: str = Form(...)):
    S.load_dataset(dataset)
    # Refresh the target <select> options; the grid is cleared until rebuilt.
    return templates.TemplateResponse(request, "_targets.html", _ctx(request))


@app.post("/build", response_class=HTMLResponse)
def build(request: Request, target: str = Form(...), rows: int = Form(...),
          cols: int = Form(...), threshold: float = Form(...),
          conf_skip: float = Form(0.95)):
    S.set_params(target, rows, cols, threshold, conf_skip)
    S.make_grid(commit_neg=False)
    return templates.TemplateResponse(request, "_grid_oob.html", _ctx(request))


@app.post("/regenerate", response_class=HTMLResponse)
def regenerate(request: Request, target: str = Form(...), rows: int = Form(...),
               cols: int = Form(...), threshold: float = Form(...),
               conf_skip: float = Form(0.95)):
    S.set_params(target, rows, cols, threshold, conf_skip)
    S.make_grid(commit_neg=True)
    return templates.TemplateResponse(request, "_grid_oob.html", _ctx(request))


@app.post("/click", response_class=HTMLResponse)
def click(request: Request, cell: int = Form(...)):
    S.click(cell)
    # Return just the clicked tile; an out-of-band fragment refreshes the status.
    ctx = _ctx(request)
    ctx["cell"] = ctx["cells"][cell]
    return templates.TemplateResponse(request, "_tile_oob.html", ctx)


@app.post("/update-model", response_class=HTMLResponse)
def update_model(request: Request):
    S.start_update()
    return templates.TemplateResponse(request, "_train.html", _ctx(request))


@app.post("/train-base", response_class=HTMLResponse)
def train_base(request: Request):
    S.start_base()
    return templates.TemplateResponse(request, "_train.html", _ctx(request))


@app.get("/train-status", response_class=HTMLResponse)
def train_status(request: Request):
    return templates.TemplateResponse(request, "_train.html", _ctx(request))


@app.post("/save", response_class=HTMLResponse)
def save(request: Request):
    msg = S.save()
    ctx = _ctx(request)
    ctx["save_msg"] = msg
    return templates.TemplateResponse(request, "_save.html", ctx)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8051)
