"""FastAPI + HTMX plankton annotation tool (MobileNet / torchvision backend).

Two grid modes over one tile mechanic (review = default-accept, hard-case =
find), an active-learning queue, class management, and a Settings page that
controls how the model trains. Labels live in SQLite; the model is a torchvision
backbone trained on raw crops with RandAugment, retrained in the background
(manually or automatically) while annotation continues.

Run (CPU, 3dmodel conda env):
    & "C:/Users/acer/anaconda3/envs/3dmodel/python.exe" annotate_server.py
Then open http://127.0.0.1:8051
"""

from __future__ import annotations

import os

# torch + sklearn/numpy-mkl in one process trip an OpenMP double-init on this
# conda env; set before any of them import. See memory: openmp-conflict.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uvicorn
from a2wsgi import WSGIMiddleware
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import annotate_model as am
import annotate_settings as st
import annotate_session as sess

DASH_PREFIX = "/dashboard/"
BACKBONES = am.available_backbones()

app = FastAPI(title="Plankton Annotator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.environ["DASH_URL_PREFIX"] = DASH_PREFIX
import dashboard  # noqa: E402  (import after setting DASH_URL_PREFIX)
app.mount("/dashboard", WSGIMiddleware(dashboard.app.server))

S = sess.Session()


def _ctx(request: Request) -> dict:
    return {
        "request": request,
        "datasets": S.dataset_options(),
        "dataset_id": S.source.dataset_id if S.source else None,
        "targets": S.classes,
        "target": S.target,
        "mode": S.mode,
        "m": S.m, "n": S.n, "conf_skip": S.conf_skip,
        "grid_max": S.settings["grid_max"],
        "cells": S.cells_view(),
        "status": S.status,
        "model_status": S.model_status(),
        "train": S.train,
        "active": "annotate",
    }


def _al_ctx(request: Request) -> dict:
    ctx = _ctx(request)
    ctx["al"] = S.al_view()
    return ctx


def _manage_ctx(request: Request) -> dict:
    ctx = _ctx(request)
    ctx["active"] = "manage"
    ctx["classes_table"] = S.class_table()
    ctx["manage_status"] = S.manage_status_str()
    return ctx


def _settings_ctx(request: Request) -> dict:
    ctx = _ctx(request)
    ctx["active"] = "settings"
    ctx["settings"] = S.get_settings()
    ctx["fields"] = st.FIELDS
    ctx["backbones"] = BACKBONES
    ctx["settings_msg"] = ""
    return ctx


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    with S.lock:
        return templates.TemplateResponse(request, "index.html", _ctx(request))


# ── tabs ─────────────────────────────────────────────────────────────────────--

@app.get("/tab/annotate", response_class=HTMLResponse)
def tab_annotate(request: Request):
    with S.lock:
        ctx = _ctx(request)
        return templates.TemplateResponse(request, "_tab_annotate.html", ctx)


@app.get("/tab/active", response_class=HTMLResponse)
def tab_active(request: Request):
    with S.lock:
        ctx = _al_ctx(request)
        ctx["active"] = "active"
        return templates.TemplateResponse(request, "_tab_active.html", ctx)


@app.get("/tab/manage", response_class=HTMLResponse)
def tab_manage(request: Request):
    with S.lock:
        return templates.TemplateResponse(request, "_tab_manage.html", _manage_ctx(request))


@app.get("/tab/settings", response_class=HTMLResponse)
def tab_settings(request: Request):
    with S.lock:
        return templates.TemplateResponse(request, "_tab_settings.html", _settings_ctx(request))


@app.get("/tab/dataset", response_class=HTMLResponse)
def tab_dataset(request: Request):
    with S.lock:
        ctx = _ctx(request)
        ctx["active"] = "dataset"
        ctx["health"] = S.dataset_health()
        return templates.TemplateResponse(request, "_tab_dataset.html", ctx)


@app.get("/tab/model", response_class=HTMLResponse)
def tab_model(request: Request):
    with S.lock:
        ctx = _ctx(request)
        ctx["active"] = "model"
        ctx["health"] = S.model_health()
        return templates.TemplateResponse(request, "_tab_model.html", ctx)


@app.get("/tab/dashboard", response_class=HTMLResponse)
def tab_dashboard(request: Request):
    with S.lock:
        ctx = _ctx(request)
        ctx["active"] = "dashboard"
        ctx["dash_url"] = DASH_PREFIX
        return templates.TemplateResponse(request, "_tab_dashboard.html", ctx)


# ── annotate grids ───────────────────────────────────────────────────────────--

@app.post("/dataset", response_class=HTMLResponse)
def set_dataset(request: Request, dataset: str = Form(...)):
    with S.lock:
        S.load_dataset(dataset)
        return templates.TemplateResponse(request, "_targets.html", _ctx(request))


@app.post("/build", response_class=HTMLResponse)
def build(request: Request, target: str = Form(...), mode: str = Form("review"),
          rows: int = Form(...), cols: int = Form(...),
          conf_skip: float = Form(0.95)):
    with S.lock:
        S.set_params(target=target, mode=mode, m=rows, n=cols, conf_skip=conf_skip)
        S.make_grid()
        return templates.TemplateResponse(request, "_grid_oob.html", _ctx(request))


@app.post("/click", response_class=HTMLResponse)
def click(request: Request, cell: int = Form(...)):
    with S.lock:
        S.click(cell)
        ctx = _ctx(request)
        ctx["cell"] = ctx["cells"][cell]
        return templates.TemplateResponse(request, "_tile_oob.html", ctx)


@app.post("/accept-all", response_class=HTMLResponse)
def accept_all(request: Request):
    with S.lock:
        S.accept_all()
        return templates.TemplateResponse(request, "_grid_oob.html", _ctx(request))


@app.post("/reject-all", response_class=HTMLResponse)
def reject_all(request: Request):
    with S.lock:
        S.reject_all()
        return templates.TemplateResponse(request, "_grid_oob.html", _ctx(request))


# ── training ─────────────────────────────────────────────────────────────────--

@app.post("/train-base", response_class=HTMLResponse)
def train_base(request: Request):
    with S.lock:
        S.start_base()
        return templates.TemplateResponse(request, "_train.html", _ctx(request))


@app.post("/retrain", response_class=HTMLResponse)
def retrain(request: Request):
    with S.lock:
        S.start_retrain()
        return templates.TemplateResponse(request, "_train.html", _ctx(request))


@app.get("/train-status", response_class=HTMLResponse)
def train_status(request: Request):
    with S.lock:
        return templates.TemplateResponse(request, "_train.html", _ctx(request))


# ── active learning ──────────────────────────────────────────────────────────--

@app.post("/al/build", response_class=HTMLResponse)
def al_build(request: Request):
    with S.lock:
        S.build_al_queue()
        return templates.TemplateResponse(request, "_al_card.html", _al_ctx(request))


@app.post("/al/assign", response_class=HTMLResponse)
def al_assign(request: Request, class_name: str = Form(...)):
    with S.lock:
        S.al_assign(class_name)
        return templates.TemplateResponse(request, "_al_card.html", _al_ctx(request))


@app.post("/al/new", response_class=HTMLResponse)
def al_new(request: Request, new_class: str = Form(...)):
    with S.lock:
        S.al_new_class(new_class)
        return templates.TemplateResponse(request, "_al_card.html", _al_ctx(request))


@app.post("/al/skip", response_class=HTMLResponse)
def al_skip(request: Request):
    with S.lock:
        S.al_skip()
        return templates.TemplateResponse(request, "_al_card.html", _al_ctx(request))


# ── class management ─────────────────────────────────────────────────────────--

@app.post("/manage/rename", response_class=HTMLResponse)
def manage_rename(request: Request, old: str = Form(...), new: str = Form(...)):
    with S.lock:
        S.rename_class(old, new)
        return templates.TemplateResponse(request, "_manage_body.html", _manage_ctx(request))


@app.post("/manage/merge", response_class=HTMLResponse)
def manage_merge(request: Request, src: str = Form(...), dst: str = Form(...)):
    with S.lock:
        S.merge_classes(src, dst)
        return templates.TemplateResponse(request, "_manage_body.html", _manage_ctx(request))


@app.post("/manage/delete", response_class=HTMLResponse)
def manage_delete(request: Request, name: str = Form(...)):
    with S.lock:
        S.delete_class(name)
        return templates.TemplateResponse(request, "_manage_body.html", _manage_ctx(request))


# ── settings ─────────────────────────────────────────────────────────────────--

@app.post("/settings/save", response_class=HTMLResponse)
async def settings_save(request: Request):
    form = await request.form()
    with S.lock:
        msg = S.update_settings(dict(form))
        ctx = _settings_ctx(request)
        ctx["settings_msg"] = msg
        return templates.TemplateResponse(request, "_settings_body.html", ctx)


@app.post("/save", response_class=HTMLResponse)
def save(request: Request):
    with S.lock:
        msg = S.save()
        ctx = _ctx(request)
        ctx["save_msg"] = msg
        return templates.TemplateResponse(request, "_save.html", ctx)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8051)
