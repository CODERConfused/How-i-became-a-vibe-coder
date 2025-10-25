import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import io

# App config
st.set_page_config(page_title="Minecraft Factions Map Viewer", layout="wide")

# Defaults (no controls bar)
DELIMITER = ","
HAS_HEADER = False
TREAT_EMPTY_AS_UNLOADED = True
UNLOADED_TOKENS = {
    "UNEXPLORED",
    "UNLOADED",
    "UNEXPLORED_CHUNK",
    "UNLOADED_CHUNK",
    "UNKNOWN",
}
ORIGIN_X = 0
ORIGIN_Z = 0


def get_palette(n: int) -> list:
    if n <= 0:
        return []
    base = px.colors.qualitative.Plotly
    if n <= len(base):
        return base[:n]
    reps = (n + len(base) - 1) // len(base)
    return (base * reps)[:n]


@st.cache_data(show_spinner=False)
def load_map(file_source, delimiter: str, has_header: bool) -> pd.DataFrame:
    header = 0 if has_header else None
    # Use memory_map only for real files; disable for in-memory buffers
    if isinstance(file_source, (str, Path)):
        return pd.read_csv(
            file_source,
            header=header,
            sep=delimiter,
            dtype=str,
            engine="c",
            na_filter=True,
            memory_map=True,
        )
    elif isinstance(file_source, (bytes, bytearray)):
        return pd.read_csv(
            io.BytesIO(file_source),
            header=header,
            sep=delimiter,
            dtype=str,
            engine="c",
            na_filter=True,
            memory_map=False,
        )
    else:
        raise TypeError("file_source must be a path or bytes")


@st.cache_data(show_spinner=False)
def build_image(
    df: pd.DataFrame,
    unloaded_tokens: set,
    treat_empty: bool,
    origin_x: int,
    origin_z: int,
):
    # Normalize to numpy array
    arr = df.to_numpy(dtype=str)
    arr = np.char.strip(arr)
    na_mask = df.isna().to_numpy()
    is_empty = (arr == "") if treat_empty else np.zeros_like(arr, dtype=bool)
    upper = np.char.upper(arr)

    tokens_arr = np.array(sorted(unloaded_tokens), dtype=str)
    is_unloaded_token = np.isin(upper, tokens_arr)
    unloaded_mask = is_empty | is_unloaded_token | na_mask

    # Factions list (exclude unloaded)
    factions = pd.unique(arr[~unloaded_mask].ravel())
    factions = factions[~pd.isna(factions)]
    factions = list(sorted(factions.tolist()))

    # Map factions to indices
    flat = arr.ravel()
    cat = pd.Categorical(flat, categories=factions)
    codes = cat.codes.reshape(arr.shape)  # -1 for not in categories (unloaded)

    # Colors
    palette = get_palette(len(factions))

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    colors_rgb = np.array([hex_to_rgb(h) for h in palette], dtype=np.uint8)

    h, w = arr.shape
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Unloaded: light gray with low alpha (but still visible)
    img[..., 0:3] = 220
    img[..., 3] = 120

    loaded_mask = codes >= 0
    if loaded_mask.any():
        img[loaded_mask, 0:3] = colors_rgb[codes[loaded_mask]]
        img[loaded_mask, 3] = 255

    # Legend with counts
    counts = []
    for i, name in enumerate(factions):
        counts.append(int((codes == i).sum()))
    unloaded_count = int(unloaded_mask.sum())

    legend = [("Unloaded", "#DCDCDC", unloaded_count)] + list(
        zip(factions, palette, counts)
    )
    return img, legend, factions, unloaded_mask


def render_map(img: np.ndarray):
    # Interactive Plotly image with pan/zoom
    fig = go.Figure(go.Image(z=img))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
    )
    fig.update_xaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        constrain="domain",
        scaleanchor="y",
    )
    fig.update_yaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        autorange="reversed",  # keep CSV top-left as north-west
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "doubleClick": "reset",
            "displaylogo": False,
            "modeBarButtonsToAdd": [
                "zoom2d",
                "pan2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ],
        },
    )


def render_legend(legend: list):
    # legend entries are (name, color_hex, count)
    st.subheader("Map Key")
    # Build HTML without leading indentation (Markdown treats indented lines as code)
    items_html = []
    for name, color, count in legend:
        items_html.append(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f"background:#f7f7f9;border:1px solid #ececf0;"
            f"border-radius:8px;padding:6px 10px;margin:6px;"
            f'white-space:nowrap;">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'border:1px solid #888;background:{color};border-radius:3px;"></span>'
            f'<span style="font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'
            f'font-size:13px;color:#222;">{name}</span>'
            f'<span style="margin-left:4px;color:#666;font-size:12px;">({count})</span>'
            f"</div>"
        )
    html = (
        '<div style="display:flex;flex-wrap:wrap;align-items:flex-start;margin:-6px;">'
        + "".join(items_html)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# UI
st.title("Minecraft Factions Map Viewer")

MAPS_DIR = Path(__file__).parent / "maps"
csv_files = sorted(MAPS_DIR.glob("*.csv"))

if not csv_files:
    st.info("No CSV files found in the 'maps' folder next to this app.")
else:
    for csv_path in csv_files:
        # Use a fresh container per map to avoid element collisions when looping
        with st.container():
            try:
                # Pass the path directly so pandas can memory-map the file
                df = load_map(csv_path, DELIMITER, HAS_HEADER)
                if not HAS_HEADER:
                    df.columns = [f"C{c}" for c in range(df.shape[1])]

                img, legend, factions, unloaded_mask = build_image(
                    df,
                    UNLOADED_TOKENS,
                    TREAT_EMPTY_AS_UNLOADED,
                    int(ORIGIN_X),
                    int(ORIGIN_Z),
                )

                st.subheader(csv_path.stem)
                lcol, rcol = st.columns([4, 1], vertical_alignment="top")
                with lcol:
                    render_map(img)
                with rcol:
                    render_legend(legend)
                    st.caption(
                        "Unloaded chunks are light gray. "
                        "Top-left of the CSV is the north-west corner."
                    )
                    st.write(f"Map size: {df.shape[0]} rows × {df.shape[1]} cols")
                    st.write(f"Factions: {len(factions)}")
                st.divider()
            except Exception as e:
                st.error(f"Failed to parse or render {csv_path.name}: {e}")
