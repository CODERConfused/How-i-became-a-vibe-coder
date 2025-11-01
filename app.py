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

# Session state for map select and search auto-zoom
if "map_select" not in st.session_state:
    st.session_state["map_select"] = 0
if "search_term" not in st.session_state:
    st.session_state["search_term"] = ""
if "last_token" not in st.session_state:
    st.session_state["last_token"] = None
if "do_autozoom" not in st.session_state:
    st.session_state["do_autozoom"] = False

st.sidebar.title("Controls")
if not csv_files:
    st.info("No CSV files found in the 'maps' folder next to this app.")
else:
    map_names = [p.stem for p in csv_files]

    st.sidebar.write(
        "Select a map and/or search for a faction. If multiple maps contain matches, jump between them below."
    )
    # Use session_state to drive the selectbox index instead of a keyed widget
    current_idx = int(st.session_state.get("map_select", 0))
    sel_idx = st.sidebar.selectbox(
        "Map",
        list(range(len(map_names))),
        index=current_idx,
        format_func=lambda i: map_names[i],
    )
    if sel_idx != current_idx:
        st.session_state["map_select"] = sel_idx
    sel_name = map_names[sel_idx]
    csv_path = csv_files[sel_idx]

    # Clear search BEFORE creating the text input
    if st.sidebar.button("Clear search"):
        st.session_state["search_term"] = ""
        st.session_state["last_token"] = None
        st.session_state["do_autozoom"] = False
        st.rerun()

    # Text input without key; synced with session_state
    default_term = st.session_state.get("search_term", "")
    term = st.sidebar.text_input(
        "Search faction", value=default_term, placeholder="e.g., aj"
    )
    if term != default_term:
        st.session_state["search_term"] = term
    term_upper = term.strip().upper()

    show_legend = st.sidebar.checkbox("Show legend in sidebar", True)
    show_grid = st.sidebar.checkbox(
        "Gridlines", True, help="Draws chunk boundaries over the image"
    )
    # Highlight + dim toggles
    highlight = st.sidebar.checkbox("Highlight matches", True)
    dim_others = st.sidebar.checkbox(
        "Dim non-matches", True, help="Dim other claimed factions when searching"
    )

    # Scan all maps for matches
    results = []
    if term_upper:
        for i, p in enumerate(csv_files):
            try:
                df_i = load_map(p, DELIMITER, HAS_HEADER)
                mask_i_df = df_i.astype(str).apply(
                    lambda col: col.str.contains(
                        term, case=False, na=False, regex=False
                    )
                )
                count = int(mask_i_df.to_numpy().sum())
                if count > 0:
                    results.append((i, map_names[i], count))
            except Exception:
                pass

        if results:
            st.sidebar.markdown("Matches found in:")
            for _, n, c in results:
                st.sidebar.write(f"- {n} ({c})")

            idxs = [i for i, _, _ in results]
            if st.sidebar.button("Jump to first match"):
                st.session_state["map_select"] = idxs[0]
                st.session_state["do_autozoom"] = True
                st.rerun()
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                if st.button("Prev map"):
                    if sel_idx in idxs:
                        pos = idxs.index(sel_idx)
                        new_idx = idxs[pos - 1] if pos > 0 else idxs[-1]
                    else:
                        new_idx = idxs[-1]
                    st.session_state["map_select"] = new_idx
                    st.session_state["do_autozoom"] = True
                    st.rerun()
            with col_b:
                if st.button("Next map"):
                    if sel_idx in idxs:
                        pos = idxs.index(sel_idx)
                        new_idx = idxs[(pos + 1) % len(idxs)]
                    else:
                        new_idx = idxs[0]
                    st.session_state["map_select"] = new_idx
                    st.session_state["do_autozoom"] = True
                    st.rerun()
        else:
            st.sidebar.info("No matches found in any map.")

    # Auto-zoom token gating (once per map+term)
    token = f"{sel_name}|{term_upper}"
    if term_upper and st.session_state["last_token"] != token:
        st.session_state["do_autozoom"] = True
        st.session_state["last_token"] = token

    # Render selected map
    try:
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

        # Anchor top-left so center aligns with (0,0)
        h, w = df.shape
        s = CHUNK_BLOCK_SIZE
        x0 = int(ORIGIN_X - (w * s) / 2)
        y0 = int(ORIGIN_Z - (h * s) / 2)

        # Build normalized strings and masks with pandas, then convert to numpy
        upper_df = df.astype(str).apply(lambda col: col.str.strip().str.upper())
        is_empty = (
            (upper_df == "")
            if TREAT_EMPTY_AS_UNLOADED
            else pd.DataFrame(False, index=upper_df.index, columns=upper_df.columns)
        )
        is_unloaded_token = upper_df.isin(list(UNLOADED_TOKENS))
        is_unclaimed_token = upper_df.isin(list(UNCLAIMED_TOKENS))
        unloaded_mask_map = (is_unloaded_token | is_empty).to_numpy()
        unclaimed_mask_map = (is_unclaimed_token & ~is_unloaded_token).to_numpy()

        # Claims-only search mask (exclude unloaded/unclaimed) using substring matching (no regex)
        if term:
            search_hits_df = df.astype(str).apply(
                lambda col: col.str.contains(term, case=False, na=False, regex=False)
            )
            claims_mask = (
                (search_hits_df.to_numpy())
                & (~unloaded_mask_map)
                & (~unclaimed_mask_map)
            )
        else:
            claims_mask = np.zeros((h, w), dtype=bool)

        # Build figure
        fig = go.Figure()
        fig.add_trace(
            go.Image(z=img, x0=x0, y0=y0, dx=CHUNK_BLOCK_SIZE, dy=CHUNK_BLOCK_SIZE)
        )

        # Dim overlay for other claimed factions (non-matches), unaffected unloaded/unclaimed
        if dim_others and term:
            dim_mask = (~claims_mask) & (~unloaded_mask_map) & (~unclaimed_mask_map)
            if dim_mask.any():
                fig.add_trace(
                    go.Heatmap(
                        z=dim_mask.astype(np.uint8),
                        x0=x0,
                        dx=CHUNK_BLOCK_SIZE,
                        y0=y0,
                        dy=CHUNK_BLOCK_SIZE,
                        zmin=0,
                        zmax=1,
                        showscale=False,
                        colorscale=[
                            [0, "rgba(0,0,0,0)"],
                            [1, "rgba(0,0,0,0.60)"],
                        ],
                        hoverinfo="skip",
                    )
                )

        # Highlight overlay for matches
        if highlight and claims_mask.any():
            fig.add_trace(
                go.Heatmap(
                    z=claims_mask.astype(np.uint8),
                    x0=x0,
                    dx=CHUNK_BLOCK_SIZE,
                    y0=y0,
                    dy=CHUNK_BLOCK_SIZE,
                    zmin=0,
                    zmax=1,
                    showscale=False,
                    colorscale=[
                        [0, "rgba(0,0,0,0)"],
                        [1, "rgba(255,255,0,0.85)"],
                    ],
                    hoverinfo="skip",
                )
            )

        # Hover overlay
        fig.add_trace(
            go.Heatmap(
                z=np.zeros((h, w), dtype=np.uint8),
                x0=x0,
                dx=CHUNK_BLOCK_SIZE,
                y0=y0,
                dy=CHUNK_BLOCK_SIZE,
                text=hover_text,
                hovertemplate="Faction: %{text}<br>Block X: %{x}, Z: %{y}<extra></extra>",
                showscale=False,
                colorscale=[
                    [0, "rgba(0,0,0,0)"],
                    [1, "rgba(0,0,0,0)"],
                ],
                hoverongaps=False,
            )
        )

        # Gridlines as shapes above image
        shapes = []
        if show_grid:
            grid_color = "rgba(255,255,255,0.15)"
            for j in range(w + 1):
                x = x0 + j * s
                shapes.append(
                    dict(
                        type="line",
                        x0=x,
                        x1=x,
                        y0=y0,
                        y1=y0 + h * s,
                        line=dict(color=grid_color, width=1),
                        layer="above",
                    )
                )
            for i in range(h + 1):
                y = y0 + i * s
                shapes.append(
                    dict(
                        type="line",
                        x0=x0,
                        x1=x0 + w * s,
                        y0=y,
                        y1=y,
                        line=dict(color=grid_color, width=1),
                        layer="above",
                    )
                )
        fig.update_layout(shapes=shapes)

        # One-time auto-zoom to matched claims (only claimed chunks)
        if st.session_state.get("do_autozoom") and claims_mask.any():
            ii, jj = np.where(claims_mask)
            x_min = x0 + int(jj.min()) * s
            x_max = x0 + (int(jj.max()) + 1) * s
            y_min = y0 + int(ii.min()) * s
            y_max = y0 + (int(ii.max()) + 1) * s
            pad_x = (x_max - x_min) * 0.10
            pad_y = (y_max - y_min) * 0.10
            fig.update_xaxes(range=[x_min - pad_x, x_max + pad_x])
            fig.update_yaxes(autorange=False, range=[y_max + pad_y, y_min - pad_y])
            st.session_state["do_autozoom"] = False
        else:
            fig.update_yaxes(autorange="reversed")

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode="pan",
            height=900,  # large/full-screen feel
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
            showticklabels=False,
            ticks="",
        )
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, ticks="")

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

        # Sidebar legend
        if show_legend:
            with st.sidebar.expander("Legend", expanded=True):
                render_legend(legend)

        # Info
        st.caption(
            "Map center is (0,0). Unloaded chunks are light gray. "
            f"Each cell is an {CHUNK_BLOCK_SIZE}×{CHUNK_BLOCK_SIZE}-block area; hover shows the NW corner block. "
            "Top-left of the CSV is the north-west corner."
        )
        st.write(
            f"Map: {sel_name} — size: {df.shape[0]} rows × {df.shape[1]} cols — Factions: {len(factions)}"
        )

    except Exception as e:
        st.error(f"Failed to parse or render {csv_path.name}: {e}")





