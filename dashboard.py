import base64
import io
import os
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from PIL import Image

from scipy.ndimage import convolve, binary_fill_holes
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize, closing, disk, remove_small_objects
from skimage.filters import threshold_otsu

DATA_DIR = Path("data")
METRICS_COL = "BioVolume"  # Column in LabelChecker CSV updated by Recalculate Metrics

# Histogram figure margins — slider wrapper uses the same values to align tracks
_HIST_MARGIN_L = 40
_HIST_MARGIN_R = 10


# ── image processing ──────────────────────────────────────────────────────────

def binary_cleanup(input_image=None):
    output_image = closing(input_image, disk(1))
    output_image = remove_small_objects(output_image.astype(bool), min_size=10)
    return output_image


def to_grayscale(input_image):
    if isinstance(input_image, np.ndarray):
        return np.array(Image.fromarray(input_image).convert('L'))
    return np.array(input_image.convert('L'))


def apply_threshold(gray_img, thresh):
    thresholded = 1 - (gray_img > thresh)
    return binary_cleanup(thresholded)


def threshold_method(input_image):
    gray = to_grayscale(input_image)
    thresh = threshold_otsu(gray)
    return apply_threshold(gray, int(thresh))


def overlay_mask_on_image(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend binary mask as semi-transparent green over the original RGB image."""
    result = original.copy().astype(np.float32)
    green = np.array([0, 200, 0], dtype=np.float32)
    alpha = 0.45
    mask_bool = mask.astype(bool)
    result[mask_bool] = alpha * green + (1 - alpha) * result[mask_bool]
    return np.clip(result, 0, 255).astype(np.uint8)


def calculate_metrics(binary_image: np.ndarray) -> float:
    """Placeholder: compute morphological metrics from binary mask."""
    return 0


def save_bin_crop(binary_crop: np.ndarray, image_path: Path,
                  x: int, y: int, w: int, h: int) -> Path:
    """Write binary crop into a full-size canvas at (x, y, w, h) and save as [stem]_binImage.tif.

    If the output file already exists its contents are preserved so that saving
    multiple samples accumulates regions rather than overwriting the whole file.
    """
    out_path = image_path.parent / (image_path.stem + "_binImage.tif")

    if out_path.exists():
        # Read bytes into memory first — avoids Windows file-lock when saving back to same path
        canvas = np.array(Image.open(io.BytesIO(out_path.read_bytes())).convert("L"))
    else:
        with Image.open(image_path) as orig:
            canvas = np.zeros((orig.height, orig.width), dtype=np.uint8)

    canvas[y:y + h, x:x + w] = binary_crop.astype(np.uint8) * 255
    Image.fromarray(canvas).save(out_path, format="TIFF")
    return out_path


def make_hist_figure(hist_counts: list, threshold: int, otsu: int | None) -> go.Figure:
    """Static histogram of pixel intensities with a threshold line and optional Otsu reference."""
    x = list(range(256))
    colors = [
        "rgba(0,160,0,0.75)" if i <= threshold else "rgba(160,160,160,0.5)"
        for i in x
    ]

    # x-axis range: trim to the actual extent of non-zero bins
    nonzero = [i for i, c in enumerate(hist_counts) if c > 0]
    x_min = (nonzero[0] - 2) if nonzero else 0
    x_max = (nonzero[-1] + 2) if nonzero else 255

    shapes = [
        {   # threshold line
            "type": "line",
            "x0": threshold, "x1": threshold, "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": "red", "width": 2},
        },
    ]
    annotations = [{
        "x": threshold, "y": 1, "yref": "paper",
        "text": f"T={threshold}",
        "showarrow": False,
        "xanchor": "left" if threshold < x_max * 0.85 else "right",
        "yanchor": "top",
        "font": {"size": 9, "color": "red"},
        "bgcolor": "rgba(255,255,255,0.85)", "borderpad": 2,
    }]

    if otsu is not None and otsu != threshold:
        shapes.append({   # Otsu reference line (paper coords — works on any y scale)
            "type": "line",
            "x0": otsu, "x1": otsu, "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": "orange", "width": 1.5, "dash": "dot"},
        })
        annotations.append({
            "x": otsu, "y": 1, "yref": "paper",
            "text": f"Otsu={otsu}",
            "showarrow": False,
            "xanchor": "right" if otsu >= threshold else "left",
            "yanchor": "top",
            "font": {"size": 9, "color": "orange"},
            "bgcolor": "rgba(255,255,255,0.85)", "borderpad": 2,
        })

    fig = go.Figure(go.Bar(
        x=x, y=hist_counts,
        marker_color=colors, marker_line_width=0,
        hovertemplate="Intensity %{x}: %{y} px<extra></extra>",
    ))
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        margin={"t": 10, "b": 22, "l": _HIST_MARGIN_L, "r": _HIST_MARGIN_R},
        height=130,
        plot_bgcolor="#fafafa", paper_bgcolor="#fff",
        bargap=0, showlegend=False,
        dragmode=False,
        xaxis={
            "range": [x_min, x_max], "title": None,
            "tickfont": {"size": 9},
        },
        yaxis={
            "type": "log", "title": "px",
            "tickfont": {"size": 9},
            "title_font": {"size": 9}, "title_standoff": 2,
        },
    )
    return fig


# ── helpers ──────────────────────────────────────────────────────────────────

def find_csv_files():
    return sorted(DATA_DIR.glob("**/LabelChecker*.csv"))


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["_row"] = range(len(df))
    return df


def numeric_cols(df: pd.DataFrame):
    skip = {"_row", "ImageX", "ImageY", "ImageW", "ImageH", "CalImage", "CalConst"}
    return [c for c in df.select_dtypes(include="number").columns if c not in skip]


def categorical_cols(df: pd.DataFrame):
    skip = {"CollageFile", "_row"}
    obj = [c for c in df.select_dtypes(include="object").columns if c not in skip]
    extra = [
        c for c in df.select_dtypes(include="number").columns
        if c not in skip and 0 < df[c].nunique() <= 20
        and c not in obj and c not in {"_row"}
    ]
    return obj + extra


def find_image(csv_path: str, collage_file: str) -> Path | None:
    base = Path(csv_path).parent
    candidate = base / collage_file
    if candidate.exists():
        return candidate
    for p in base.parent.rglob(collage_file):
        return p
    return None


def crop_image(image_path: Path, x: int, y: int, w: int, h: int) -> np.ndarray:
    with Image.open(image_path) as img:
        box = (x, y, x + w, y + h)
        cropped = img.crop(box).convert("RGB")
        return np.array(cropped)


def array_to_base64(arr: np.ndarray) -> str:
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gen_bin_image(image: np.ndarray) -> np.ndarray:
    """Generate a binary image from the original plankton image."""
    return threshold_method(image)


# ── app layout ───────────────────────────────────────────────────────────────

# When mounted under the FastAPI annotation app (annotate_server.py), the env var
# tells Dash to request its assets/callbacks under that path prefix (e.g.
# "/dashboard/"). Unset when run standalone, so the dashboard still works on 8050.
_dash_prefix = os.environ.get("DASH_URL_PREFIX")
app = dash.Dash(
    __name__,
    **({"requests_pathname_prefix": _dash_prefix} if _dash_prefix else {}),
)
app.title = "Plankton Dashboard"

csv_files = find_csv_files()
csv_options = [
    {"label": str(p.relative_to(DATA_DIR)), "value": str(p)}
    for p in csv_files
]
default_csv = str(csv_files[0]) if csv_files else None

_btn_style = {
    "fontSize": "11px", "padding": "2px 10px", "cursor": "pointer",
    "borderRadius": "4px", "border": "1px solid #aaa", "background": "#f0f0f0",
}

_input_style = {
    "width": "54px", "fontSize": "12px", "padding": "2px 4px",
    "border": "1px solid #aaa", "borderRadius": "4px",
    "textAlign": "center", "flexShrink": "0",
}

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "16px",
           "background": "#f5f5f5", "minHeight": "100vh"},
    children=[
        html.H2("Plankton Dashboard", style={"margin": "0 0 12px 0", "color": "#222"}),

        # ── top controls ─────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                   "marginBottom": "12px", "alignItems": "flex-end"},
            children=[
                html.Div([
                    html.Label("Dataset", style={"fontWeight": "bold", "fontSize": "12px"}),
                    dcc.Dropdown(id="csv-select", options=csv_options, value=default_csv,
                                 clearable=False, style={"width": "320px"}),
                ]),
                html.Div([
                    html.Label("X axis", style={"fontWeight": "bold", "fontSize": "12px"}),
                    dcc.Dropdown(id="x-col", clearable=False, style={"width": "220px"}),
                ]),
                html.Div([
                    html.Label("Y axis", style={"fontWeight": "bold", "fontSize": "12px"}),
                    dcc.Dropdown(id="y-col", clearable=False, style={"width": "220px"}),
                ]),
                html.Div([
                    html.Label("Color by", style={"fontWeight": "bold", "fontSize": "12px"}),
                    dcc.Dropdown(id="color-col", style={"width": "220px"}, placeholder="(none)"),
                ]),
            ],
        ),

        # ── group filter panel ───────────────────────────────────────────────
        html.Div(
            id="group-filter-panel",
            style={
                "marginBottom": "12px", "background": "#fff", "borderRadius": "8px",
                "boxShadow": "0 1px 4px rgba(0,0,0,.15)", "padding": "10px 14px",
                "display": "none",
            },
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center",
                           "gap": "8px", "marginBottom": "6px"},
                    children=[
                        html.Span("Show groups:", style={"fontWeight": "bold", "fontSize": "12px"}),
                        html.Button("All",  id="btn-all",  n_clicks=0, style=_btn_style),
                        html.Button("None", id="btn-none", n_clicks=0, style=_btn_style),
                    ],
                ),
                html.Div(
                    style={"maxHeight": "90px", "overflowY": "auto"},
                    children=[
                        dcc.Checklist(id="group-filter", options=[], value=[], inline=True,
                                      labelStyle={"marginRight": "14px", "fontSize": "12px"}),
                    ],
                ),
            ],
        ),

        # ── main area ────────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "flex-start"},
            children=[
                html.Div(
                    style={"flex": "1", "background": "#fff", "borderRadius": "8px",
                           "boxShadow": "0 1px 4px rgba(0,0,0,.15)", "padding": "8px"},
                    children=[dcc.Graph(id="scatter", style={"height": "70vh"})],
                ),
                html.Div(
                    style={
                        "width": "340px", "background": "#fff", "borderRadius": "8px",
                        "boxShadow": "0 1px 4px rgba(0,0,0,.15)", "padding": "16px",
                        "display": "flex", "flexDirection": "column", "alignItems": "center",
                    },
                    children=[
                        html.H4("Plankton Image", style={"margin": "0 0 8px 0", "color": "#333"}),
                        html.Div(
                            id="plankton-info",
                            style={"fontSize": "12px", "color": "#555", "marginBottom": "10px",
                                   "textAlign": "center", "wordBreak": "break-all"},
                            children="Click a point to view the image",
                        ),
                        html.Img(id="plankton-img", style={
                            "maxWidth": "100%", "imageRendering": "pixelated",
                            "border": "1px solid #ddd", "borderRadius": "4px",
                        }),
                        html.Button(
                            "Generate Bin Image", id="btn-gen-bin", n_clicks=0, disabled=True,
                            style={
                                "marginTop": "12px", "padding": "6px 14px",
                                "fontSize": "12px", "cursor": "pointer",
                                "borderRadius": "4px", "border": "1px solid #888",
                                "background": "#e8f0fe",
                            },
                        ),
                        html.Div(
                            id="bin-img-container",
                            style={
                                "marginTop": "10px", "display": "none",
                                "flexDirection": "column", "alignItems": "center",
                                "width": "100%",
                            },
                            children=[
                                html.Div("Binary Overlay",
                                         style={"fontSize": "12px", "fontWeight": "bold",
                                                "color": "#333", "marginBottom": "4px"}),
                                html.Img(id="bin-img", style={
                                    "maxWidth": "100%", "imageRendering": "pixelated",
                                    "border": "1px solid #ddd", "borderRadius": "4px",
                                }),

                                # ── histogram + slider ────────────────────────
                                html.Div(
                                    style={"width": "100%", "marginTop": "12px"},
                                    children=[
                                        html.Div(
                                            "Pixel Intensity Distribution",
                                            style={"fontSize": "10px", "color": "#888",
                                                   "marginBottom": "2px", "textAlign": "center"},
                                        ),
                                        dcc.Graph(
                                            id="intensity-hist",
                                            config={"displayModeBar": False},
                                            style={"width": "100%"},
                                        ),
                                        # Slider padded to align its track with the
                                        # histogram's plot-area left/right margins
                                        html.Div(
                                            style={
                                                "paddingLeft": f"{_HIST_MARGIN_L}px",
                                                "paddingRight": f"{_HIST_MARGIN_R}px",
                                                "marginTop": "-12px",
                                            },
                                            children=[
                                                dcc.Slider(
                                                    id="threshold-slider",
                                                    min=0, max=255, step=1, value=128,
                                                    marks=None,
                                                    updatemode="drag",
                                                ),
                                            ],
                                        ),
                                        # Precise numeric entry
                                        html.Div(
                                            style={
                                                "display": "flex", "alignItems": "center",
                                                "gap": "8px", "marginTop": "6px",
                                                "justifyContent": "center",
                                            },
                                            children=[
                                                html.Span("Threshold:",
                                                          style={"fontSize": "11px", "color": "#555"}),
                                                dcc.Input(
                                                    id="threshold-input",
                                                    type="number", min=0, max=255, step=1,
                                                    value=128, debounce=True,
                                                    style=_input_style,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),

                                # ── action buttons ────────────────────────────
                                html.Div(
                                    style={
                                        "display": "flex", "gap": "8px", "marginTop": "10px",
                                        "flexWrap": "wrap", "justifyContent": "center",
                                    },
                                    children=[
                                        html.Button("Recalculate Metrics",
                                                    id="recalc-metrics-btn", n_clicks=0,
                                                    style=_btn_style),
                                        html.Button("Save Binary Image",
                                                    id="save-bin-btn", n_clicks=0,
                                                    style=_btn_style),
                                    ],
                                ),
                                html.Div(id="bin-img-status",
                                         style={"fontSize": "11px", "color": "#555",
                                                "marginTop": "6px", "textAlign": "center"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        dcc.Store(id="current-csv"),
        dcc.Store(id="otsu-threshold-store"),
        dcc.Store(id="hist-data-store"),
    ],
)


# ── callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("x-col", "options"), Output("x-col", "value"),
    Output("y-col", "options"), Output("y-col", "value"),
    Output("color-col", "options"), Output("color-col", "value"),
    Output("current-csv", "data"),
    Input("csv-select", "value"),
)
def update_column_menus(csv_path):
    if not csv_path:
        return [], None, [], None, [], None, None
    df = load_df(csv_path)
    num = numeric_cols(df)
    cat = categorical_cols(df)
    num_opts = [{"label": c, "value": c} for c in num]
    cat_opts = [{"label": c, "value": c} for c in cat]
    return (
        num_opts, num[0] if num else None,
        num_opts, num[1] if len(num) > 1 else (num[0] if num else None),
        cat_opts, cat[0] if cat else None,
        csv_path,
    )


@app.callback(
    Output("group-filter", "options"),
    Output("group-filter", "value"),
    Output("group-filter-panel", "style"),
    Input("color-col", "value"),
    Input("btn-all", "n_clicks"),
    Input("btn-none", "n_clicks"),
    State("current-csv", "data"),
    State("group-filter", "options"),
)
def update_group_filter(color_col, n_all, n_none, csv_path, current_opts):
    _ = n_all, n_none
    triggered = dash.ctx.triggered_id
    panel_hidden = {
        "marginBottom": "12px", "background": "#fff", "borderRadius": "8px",
        "boxShadow": "0 1px 4px rgba(0,0,0,.15)", "padding": "10px 14px", "display": "none",
    }
    panel_visible = {**panel_hidden, "display": "block"}
    if not color_col or not csv_path:
        return [], [], panel_hidden
    if triggered in ("btn-all", "btn-none"):
        vals = [o["value"] for o in current_opts] if triggered == "btn-all" else []
        return current_opts, vals, panel_visible
    df = load_df(csv_path)
    groups = sorted(df[color_col].dropna().unique().tolist(), key=str)
    opts = [{"label": str(g), "value": g} for g in groups]
    return opts, groups, panel_visible


@app.callback(
    Output("scatter", "figure"),
    Input("x-col", "value"), Input("y-col", "value"),
    Input("color-col", "value"), Input("group-filter", "value"),
    State("current-csv", "data"),
)
def update_scatter(x_col, y_col, color_col, selected_groups, csv_path):
    if not csv_path or not x_col or not y_col:
        return go.Figure()
    df = load_df(csv_path)
    if color_col and selected_groups is not None:
        df = df[df[color_col].isin(selected_groups)]
    hover_cols = [c for c in ["Id", "CollageFile", "LabelPredicted", "LabelTrue"] if c in df.columns]
    fig = px.scatter(df, x=x_col, y=y_col,
                     color=color_col if color_col else None,
                     hover_data=hover_cols if hover_cols else None,
                     custom_data=["_row"])
    fig.update_traces(marker={"size": 8, "opacity": 0.8}, selector={"mode": "markers"})
    fig.update_layout(
        clickmode="event", margin={"t": 30, "b": 40, "l": 50, "r": 10},
        legend={"title": color_col or ""},
        plot_bgcolor="#fafafa", paper_bgcolor="#fff",
    )
    return fig


def _row_from_click(click_data, csv_path):
    if not click_data or not csv_path:
        return None, "Click a point to view the image"
    point = click_data["points"][0]
    if "customdata" not in point:
        return None, "Point has no row reference"
    row_idx = int(point["customdata"][0])
    df = load_df(csv_path)
    if row_idx >= len(df):
        return None, f"Row {row_idx} out of range"
    return df.iloc[row_idx], None


@app.callback(
    Output("plankton-img", "src"),
    Output("plankton-info", "children"),
    Output("btn-gen-bin", "disabled"),
    Input("scatter", "clickData"),
    State("current-csv", "data"),
)
def show_image(click_data, csv_path):
    row, err = _row_from_click(click_data, csv_path)
    if row is None:
        return None, err, True
    collage_file = row.get("CollageFile")
    if not collage_file or pd.isna(collage_file):
        return None, "No CollageFile for this point", True
    x, y, w, h = int(row["ImageX"]), int(row["ImageY"]), int(row["ImageW"]), int(row["ImageH"])
    image_path = find_image(csv_path, collage_file)
    if image_path is None:
        return None, f"Image not found: {collage_file}", True
    try:
        src = array_to_base64(crop_image(image_path, x, y, w, h))
        id_val = row.get("Id", "?")
        label = row.get("LabelPredicted", "")
        info = [
            html.B(f"Id: {id_val}"), html.Br(),
            f"File: {collage_file}", html.Br(),
            f"Crop: ({x},{y}) {w}×{h}px", html.Br(),
            html.B(f"Label: {label}") if label else "",
        ]
        return src, info, False
    except Exception as exc:
        return None, f"Error loading image: {exc}", True


@app.callback(
    Output("bin-img-container", "style"),
    Output("threshold-slider", "value"),
    Output("threshold-slider", "min"),
    Output("threshold-slider", "max"),
    Output("threshold-input", "value"),
    Output("otsu-threshold-store", "data"),
    Output("hist-data-store", "data"),
    Input("btn-gen-bin", "n_clicks"),
    Input("scatter", "clickData"),
    State("current-csv", "data"),
    prevent_initial_call=True,
)
def update_bin_setup(n_clicks, click_data, csv_path):
    hidden = {
        "marginTop": "10px", "display": "none",
        "flexDirection": "column", "alignItems": "center", "width": "100%",
    }
    visible = {**hidden, "display": "flex"}
    triggered = dash.ctx.triggered_id
    nu = dash.no_update

    if triggered == "scatter":
        return hidden, nu, nu, nu, nu, nu, nu

    if triggered != "btn-gen-bin" or not n_clicks:
        return hidden, nu, nu, nu, nu, nu, nu

    row, _ = _row_from_click(click_data, csv_path)
    if row is None:
        return hidden, nu, nu, nu, nu, nu, nu

    collage_file = row.get("CollageFile")
    if not collage_file or pd.isna(collage_file):
        return hidden, nu, nu, nu, nu, nu, nu

    x, y, w, h = int(row["ImageX"]), int(row["ImageY"]), int(row["ImageW"]), int(row["ImageH"])
    image_path = find_image(csv_path, collage_file)
    if image_path is None:
        return hidden, nu, nu, nu, nu, nu, nu

    try:
        original = crop_image(image_path, x, y, w, h)
        gray = to_grayscale(original)
        otsu = int(threshold_otsu(gray))
        hist_counts = np.bincount(gray.ravel(), minlength=256).tolist()
        nonzero = [i for i, c in enumerate(hist_counts) if c > 0]
        x_min = nonzero[0] if nonzero else 0
        x_max = nonzero[-1] if nonzero else 255
        return visible, otsu, x_min, x_max, otsu, otsu, hist_counts
    except Exception:
        return hidden, nu, nu, nu, nu, nu, nu


@app.callback(
    Output("bin-img", "src"),
    Input("threshold-slider", "value"),
    State("scatter", "clickData"),
    State("current-csv", "data"),
    prevent_initial_call=True,
)
def update_bin_display(threshold, click_data, csv_path):
    if threshold is None:
        return None
    row, _ = _row_from_click(click_data, csv_path)
    if row is None:
        return None
    collage_file = row.get("CollageFile")
    if not collage_file or pd.isna(collage_file):
        return None
    x, y, w, h = int(row["ImageX"]), int(row["ImageY"]), int(row["ImageW"]), int(row["ImageH"])
    image_path = find_image(csv_path, collage_file)
    if image_path is None:
        return None
    try:
        original = crop_image(image_path, x, y, w, h)
        gray = to_grayscale(original)
        binary = apply_threshold(gray, threshold)
        return array_to_base64(overlay_mask_on_image(original, binary))
    except Exception:
        return None


@app.callback(
    Output("intensity-hist", "figure"),
    Input("threshold-slider", "value"),
    State("hist-data-store", "data"),
    State("otsu-threshold-store", "data"),
    prevent_initial_call=True,
)
def update_intensity_hist(threshold, hist_data, otsu):
    if threshold is None or hist_data is None:
        return go.Figure()
    return make_hist_figure(hist_data, threshold, otsu)


# ── bidirectional sync between slider and input box ───────────────────────────

@app.callback(
    Output("threshold-input", "value", allow_duplicate=True),
    Input("threshold-slider", "value"),
    prevent_initial_call=True,
)
def sync_slider_to_input(val):
    return val


@app.callback(
    Output("threshold-slider", "value", allow_duplicate=True),
    Input("threshold-input", "value"),
    prevent_initial_call=True,
)
def sync_input_to_slider(val):
    if val is None:
        return dash.no_update
    return int(max(0, min(255, val)))


# ── save / recalculate actions ────────────────────────────────────────────────

@app.callback(
    Output("bin-img-status", "children"),
    Input("save-bin-btn", "n_clicks"),
    Input("recalc-metrics-btn", "n_clicks"),
    State("scatter", "clickData"),
    State("current-csv", "data"),
    State("threshold-slider", "value"),
    prevent_initial_call=True,
)
def handle_bin_actions(save_clicks, recalc_clicks, click_data, csv_path, threshold):
    triggered = dash.ctx.triggered_id
    if not triggered:
        return ""
    row, _ = _row_from_click(click_data, csv_path)
    if row is None:
        return "No image selected."
    collage_file = row.get("CollageFile")
    if not collage_file or pd.isna(collage_file):
        return "No CollageFile for this point."
    image_path = find_image(csv_path, collage_file)
    if image_path is None:
        return f"Image not found: {collage_file}"
    x, y, w, h = int(row["ImageX"]), int(row["ImageY"]), int(row["ImageW"]), int(row["ImageH"])
    try:
        original = crop_image(image_path, x, y, w, h)
        gray = to_grayscale(original)
        thresh = threshold if threshold is not None else int(threshold_otsu(gray))
        binary = apply_threshold(gray, thresh)
        if triggered == "save-bin-btn":
            out_path = save_bin_crop(binary, image_path, x, y, w, h)
            return f"Saved: {out_path.name}"
        if triggered == "recalc-metrics-btn":
            metric = calculate_metrics(binary)
            df = pd.read_csv(csv_path)
            row_idx = int(row["_row"])
            if METRICS_COL not in df.columns:
                df[METRICS_COL] = None
            df.at[row_idx, METRICS_COL] = metric
            df.to_csv(csv_path, index=False)
            return f"{METRICS_COL} updated to {metric} (row {row_idx})"
    except Exception as exc:
        return f"Error: {exc}"
    return ""


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
