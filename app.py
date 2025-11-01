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
# Mark these as "Unclaimed"
UNCLAIMED_TOKENS = {"1", "UNCLAIMED", "UNCLAIM"}
UNCLAIMED_COLOR = "#A9A9A9"
# Treat these as the center (0,0) of the map
ORIGIN_X = 0
ORIGIN_Z = 0
CHUNK_BLOCK_SIZE = 8  # each cell is an 8×8 block chunk


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
    unclaimed_tokens: set,
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

    unclaimed_arr = np.array(sorted(unclaimed_tokens), dtype=str)
    is_unclaimed_token = np.isin(upper, unclaimed_arr)

    unloaded_mask = is_empty | is_unloaded_token | na_mask
    unclaimed_mask = is_unclaimed_token & ~unloaded_mask

    # Factions list (exclude unloaded and unclaimed)
    factions = pd.unique(arr[~(unloaded_mask | unclaimed_mask)].ravel())
    factions = factions[~pd.isna(factions)]
    factions = list(sorted(factions.tolist()))

    # Map factions to indices
    flat = arr.ravel()
    cat = pd.Categorical(flat, categories=factions)
    codes = cat.codes.reshape(
        arr.shape
    )  # -1 for not in categories (unloaded/unclaimed)

    # Colors
    palette = get_palette(len(factions))

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    colors_rgb = np.array([hex_to_rgb(h) for h in palette], dtype=np.uint8)
    unclaimed_rgb = np.array(hex_to_rgb(UNCLAIMED_COLOR), dtype=np.uint8)

    h, w = arr.shape
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Unloaded: light gray with low alpha
    img[..., 0:3] = 220
    img[..., 3] = 120

    # Unclaimed: solid mid-gray
    if unclaimed_mask.any():
        img[unclaimed_mask, 0:3] = unclaimed_rgb
        img[unclaimed_mask, 3] = 255

    # Claimed factions
    loaded_mask = codes >= 0
    if loaded_mask.any():
        img[loaded_mask, 0:3] = colors_rgb[codes[loaded_mask]]
        img[loaded_mask, 3] = 255

    # Legend with counts
    counts = []
    for i, name in enumerate(factions):
        counts.append(int((codes == i).sum()))
    unloaded_count = int(unloaded_mask.sum())
    unclaimed_count = int(unclaimed_mask.sum())

    legend = [("Unloaded", "#DCDCDC", unloaded_count)]
    if unclaimed_count > 0:
        legend.append(("Unclaimed", UNCLAIMED_COLOR, unclaimed_count))
    legend += list(zip(factions, palette, counts))

    # Hover text per cell
    hover_text = np.where(
        unloaded_mask, "Unloaded", np.where(unclaimed_mask, "Unclaimed", arr)
    )

    return img, legend, factions, unloaded_mask, hover_text


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

# NEW: sidebar controls
st.sidebar.title("Controls")
if not csv_files:
    st.info("No CSV files found in the 'maps' folder next to this app.")
else:
    map_names = [p.stem for p in csv_files]
    sel_name = st.sidebar.selectbox("Map", map_names, index=0)
    csv_path = next(p for p in csv_files if p.stem == sel_name)

    term = st.sidebar.text_input("Search faction", "")
    show_legend = st.sidebar.checkbox("Show legend", True)
    show_grid = st.sidebar.checkbox("Gridlines", True)
    highlight = st.sidebar.checkbox(
        "Highlight matches", True, help="Yellow overlay on matching chunks"
    )
    dim_others = st.sidebar.checkbox(
        "Dim non-matches", True, help="Darken non-matching chunks when searching"
    )

    # Use a fresh container per map to avoid element collisions
    with st.container():
        try:
            # Pass the path directly so pandas can memory-map the file
            df = load_map(csv_path, DELIMITER, HAS_HEADER)
            if not HAS_HEADER:
                df.columns = [f"C{c}" for c in range(df.shape[1])]

            img, legend, factions, unloaded_mask, hover_text = build_image(
                df,
                UNLOADED_TOKENS,
                UNCLAIMED_TOKENS,
                TREAT_EMPTY_AS_UNLOADED,
                int(ORIGIN_X),
                int(ORIGIN_Z),
            )

            # Compute top-left anchor so that (ORIGIN_X, ORIGIN_Z) is at the map center
            h, w = df.shape
            s = CHUNK_BLOCK_SIZE
            x0 = int(ORIGIN_X - (w * s) / 2)
            y0 = int(ORIGIN_Z - (h * s) / 2)

            # Build figure
            lcol, rcol = st.columns([4, 1], vertical_alignment="top")
            with lcol:
                # Base map
                h_img, w_img = img.shape[:2]
                fig = go.Figure()
                fig.add_trace(
                    go.Image(
                        z=img, x0=x0, y0=y0, dx=CHUNK_BLOCK_SIZE, dy=CHUNK_BLOCK_SIZE
                    )
                )

                # Prepare search mask (case-insensitive)
                if term:
                    arr_upper = df.astype(str).applymap(lambda x: x.strip().upper())
                    mask = arr_upper.applymap(
                        lambda x: (
                            term.strip().upper() in x if isinstance(x, str) else False
                        )
                    ).to_numpy()
                else:
                    mask = np.zeros((h_img, w_img), dtype=bool)

                # Optional highlight overlay (yellow)
                if highlight and mask.any():
                    fig.add_trace(
                        go.Heatmap(
                            z=mask.astype(np.uint8),
                            x0=x0,
                            dx=CHUNK_BLOCK_SIZE,
                            y0=y0,
                            dy=CHUNK_BLOCK_SIZE,
                            showscale=False,
                            colorscale=[
                                [0, "rgba(0,0,0,0)"],
                                [1, "rgba(255,255,0,0.5)"],
                            ],
                            hoverinfo="skip",
                        )
                    )

                # Optional dim overlay (darken non-matches)
                if term and dim_others:
                    inv = (~mask).astype(np.uint8)
                    fig.add_trace(
                        go.Heatmap(
                            z=inv,
                            x0=x0,
                            dx=CHUNK_BLOCK_SIZE,
                            y0=y0,
                            dy=CHUNK_BLOCK_SIZE,
                            showscale=False,
                            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0.35)"]],
                            hoverinfo="skip",
                        )
                    )

                # Transparent heatmap overlay to provide hover text + coords
                fig.add_trace(
                    go.Heatmap(
                        z=np.zeros((h_img, w_img), dtype=np.uint8),
                        x0=x0,
                        dx=CHUNK_BLOCK_SIZE,
                        y0=y0,
                        dy=CHUNK_BLOCK_SIZE,
                        text=hover_text,
                        hovertemplate="Faction: %{text}<br>Block X: %{x}, Z: %{y}<extra></extra>",
                        showscale=False,
                        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                        hoverongaps=False,
                    )
                )

                # NEW: auto-zoom axes to matched chunks (if any)
                if mask.any():
                    ii, jj = np.where(mask)
                    s = CHUNK_BLOCK_SIZE
                    x_min = x0 + int(jj.min()) * s
                    x_max = x0 + (int(jj.max()) + 1) * s
                    y_min = y0 + int(ii.min()) * s
                    y_max = y0 + (int(ii.max()) + 1) * s
                    pad_x = (x_max - x_min) * 0.10
                    pad_y = (y_max - y_min) * 0.10
                    fig.update_xaxes(range=[x_min - pad_x, x_max + pad_x])
                    # keep y reversed: larger first, then smaller
                    fig.update_yaxes(
                        autorange=False, range=[y_max + pad_y, y_min - pad_y]
                    )

                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode="pan")
                fig.update_xaxes(
                    visible=False,
                    showgrid=show_grid,
                    zeroline=False,
                    constrain="domain",
                    scaleanchor="y",
                )
                fig.update_yaxes(
                    visible=False,
                    showgrid=show_grid,
                    zeroline=False,
                    autorange="reversed" if not mask.any() else None,
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

            with rcol:
                if show_legend:
                    render_legend(legend)
                st.caption(
                    "Map center is (0,0). Unloaded chunks are light gray. "
                    "Each cell is an 8×8-block area; hover shows the NW corner block. "
                    "Top-left of the CSV is the north-west corner."
                )
                st.write(f"Map size: {df.shape[0]} rows × {df.shape[1]} cols")
                st.write(f"Factions: {len(factions)}")

        except Exception as e:
            st.error(f"Failed to parse or render {csv_path.name}: {e}")




