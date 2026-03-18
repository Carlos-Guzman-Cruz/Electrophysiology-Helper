"""
Filename: patch_clamp_hub.py

Author: Carlos A. Guzman-Cruz
Date: Jan 2026
Version: 2.0.1
Description:
This file serves to aid with patch clamp recordings. From visualization,
interspike interval, instant frequency, and splicing. A good way to do some preliminary
plotting and analysis while also help in developing the final figures. This file is the main
python that displays the UI and calls patch_clamp_analysis_helper to run python DataFrame
analysis on the abf files.
"""

__author__ = "Carlos A. Guzman-Cruz"
__email__ = "carguz2002@gmail.com"
__version__ = "2.0.1"

import pyabf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output
from plotly.subplots import make_subplots
import fnmatch
import math
import plotly.io as pio
import json
from ipyfilechooser import FileChooser
from pathlib import Path
import joblib

# ─────────────────────────────────────────────
# Global State Variables
# ─────────────────────────────────────────────

data_paths_found = {}       # {'data_file_name': 'path_to_data'}

selected_file_names = []    # ['selected_data_file_name_1', 'selected_data_file_name_2', ...]

isi_key_data = {}           # { 'data_file_name': { 'path' : 'string_actual_absolute_path',
                            #                       'tmStrT' : float start time for analysis,
                            #                       'tmEND'  : float end time for analysis,
                            #                       'upTHR'  : float,
                            #                       'lwTHR'  : float,
                            #                       'cleaned_df'    : dataFrame,
                            #                       'peaks_df'      : dataFrame,
                            #                       'isi_df'        : dataFrame,
                            #                       'isi_mean'      : float mean interval interspike,
                            #                       'inst_freq_mean': float mean instant frequency,
                            #                      },
                            #        ...
                            #   }


# ─────────────────────────────────────────────
# Helper: find all .abf files under a directory
# ─────────────────────────────────────────────

def _find_abf_files(root_dir: str) -> dict:
    """Recursively search root_dir for all .abf files.
    Returns {filename: absolute_path}."""
    found = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fnmatch.fnmatch(fname.lower(), "*.abf"):
                abs_path = str(Path(dirpath) / fname)
                # Handle duplicate file names by appending parent folder
                key = fname
                if key in found:
                    parent = Path(dirpath).name
                    key = f"{parent}/{fname}"
                found[key] = abs_path
    return found


# ─────────────────────────────────────────────
# 1.  data_scan()
# ─────────────────────────────────────────────

def data_scan():
    """
    Scan for .abf files.
    - Choose a folder OR individual files via FileChooser.
    - Results populate data_paths_found.
    - Keys can be renamed (without renaming the file on disk).
    - Save / Save-to-memory (joblib) / Load-previous-selection.
    """
    global data_paths_found

    # ── Layout containers ──────────────────────────────────────────────
    out_status   = widgets.Output()
    out_table    = widgets.Output()
    out_rename   = widgets.Output()

    # ── File / Folder chooser ──────────────────────────────────────────
    fc = FileChooser(os.getcwd(), title="<b>Select a folder OR a single .abf file:</b>")
    fc.show_only_dirs = False

    mode_toggle = widgets.ToggleButtons(
        options=["Scan Folder (recursive)", "Pick Individual File"],
        description="Mode:",
        button_style=""
    )

    btn_scan   = widgets.Button(description="Find Now",    button_style="primary", icon="search")
    btn_rescan = widgets.Button(description="Re-Scan",     button_style="warning", icon="refresh")
    btn_save   = widgets.Button(description="Save",        button_style="success", icon="check")
    btn_mem    = widgets.Button(description="Save to Memory (joblib)", button_style="info", icon="download")
    btn_load   = widgets.Button(description="Load Previous Selection", button_style="", icon="upload")

    rename_area = widgets.VBox([])   # populated after first scan

    # ── Internal state ────────────────────────────────────────────────
    _temp_found   = {}   # staging area before Save
    _rename_fields = {}  # {original_key: Text widget}

    def _show_table(d: dict):
        with out_table:
            clear_output(wait=True)
            if not d:
                print("No .abf files found.")
                return
            rows = "".join(
                f"<tr><td style='padding:2px 8px'>{k}</td>"
                f"<td style='padding:2px 8px; color:gray; font-size:0.85em'>{v}</td></tr>"
                for k, v in d.items()
            )
            from IPython.display import HTML
            display(HTML(
                f"<b>{len(d)} file(s) found:</b>"
                f"<table style='font-family:monospace'><thead>"
                f"<tr><th>Key</th><th>Path</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
            ))

    def _build_rename_ui(d: dict):
        nonlocal _rename_fields
        _rename_fields = {}
        rows = []
        for k in d:
            txt = widgets.Text(value=k, layout=widgets.Layout(width="320px"))
            _rename_fields[k] = txt
            rows.append(widgets.HBox([
                widgets.Label(k, layout=widgets.Layout(width="240px")),
                widgets.Label(" → ", layout=widgets.Layout(width="30px")),
                txt
            ]))
        with out_rename:
            clear_output(wait=True)
            if rows:
                display(widgets.Label("✏️  Rename keys (optional, does NOT rename files):"))
                display(widgets.VBox(rows))

    def _do_scan(_):
        nonlocal _temp_found
        with out_status:
            clear_output(wait=True)
            try:
                selected_path = fc.selected
                if not selected_path:
                    print("⚠️  Please select a folder or file first.")
                    return

                selected_path = str(Path(selected_path))

                if mode_toggle.value == "Scan Folder (recursive)":
                    target = selected_path if os.path.isdir(selected_path) else str(Path(selected_path).parent)
                    _temp_found = _find_abf_files(target)
                else:
                    # Individual file
                    if not selected_path.lower().endswith(".abf"):
                        print("⚠️  Selected file is not an .abf file.")
                        return
                    _temp_found = {Path(selected_path).name: selected_path}

                print(f"✅  Scan complete — {len(_temp_found)} file(s) found. Press 'Save' to confirm.")
            except Exception as e:
                print(f"❌  Error during scan: {e}")

        _show_table(_temp_found)
        _build_rename_ui(_temp_found)

    def _do_save(_):
        global data_paths_found
        nonlocal _temp_found

        with out_status:
            clear_output(wait=True)
            if not _temp_found:
                print("⚠️  Nothing to save. Run 'Find Now' first.")
                return

            # Apply renames
            renamed = {}
            used_keys = set()
            for orig_key, txt_widget in _rename_fields.items():
                new_key = txt_widget.value.strip() or orig_key
                if new_key in used_keys:
                    new_key = f"{new_key}_{orig_key}"
                used_keys.add(new_key)
                renamed[new_key] = _temp_found[orig_key]

            data_paths_found = renamed
            print(f"✅  data_paths_found updated with {len(data_paths_found)} entry(ies).")

        _show_table(data_paths_found)

    def _do_save_mem(_):
        with out_status:
            clear_output(wait=True)
            if not data_paths_found:
                print("⚠️  data_paths_found is empty. Save first.")
                return
            try:
                joblib.dump(data_paths_found, "paths_found.joblib")
                print("✅  Saved as 'paths_found.joblib'.")
            except Exception as e:
                print(f"❌  Error saving: {e}")

    def _do_load(_):
        global data_paths_found
        with out_status:
            clear_output(wait=True)
            try:
                loaded = joblib.load("paths_found.joblib")
                data_paths_found = loaded
                print(f"✅  Loaded {len(data_paths_found)} entry(ies) from 'paths_found.joblib'.")
            except FileNotFoundError:
                print("❌  'paths_found.joblib' not found in current directory.")
            except Exception as e:
                print(f"❌  Error loading: {e}")
        _show_table(data_paths_found)

    btn_scan.on_click(_do_scan)
    btn_rescan.on_click(_do_scan)
    btn_save.on_click(_do_save)
    btn_mem.on_click(_do_save_mem)
    btn_load.on_click(_do_load)

    ui = widgets.VBox([
        widgets.HTML("<h3>🔬 Step 1 — Data Scan</h3>"),
        mode_toggle,
        fc,
        widgets.HBox([btn_scan, btn_rescan, btn_save, btn_mem, btn_load]),
        out_status,
        out_table,
        out_rename,
    ])
    display(ui)


# ─────────────────────────────────────────────
# 2.  select_files()
# ─────────────────────────────────────────────

def select_files():
    """
    Choose one or more files from data_paths_found for subsequent analysis.
    Populates selected_file_names.
    Save / Save-to-memory (joblib) / Load-previous-selection.
    """
    global selected_file_names

    out_status = widgets.Output()

    def _build_ui():
        nonlocal chk_box_group
        keys = list(data_paths_found.keys())
        if not keys:
            with out_status:
                clear_output(wait=True)
                print("⚠️  data_paths_found is empty. Run data_scan() first.")
            return

        options = [(k, k) for k in keys]
        chk_box_group = widgets.SelectMultiple(
            options=options,
            rows=min(len(keys), 12),
            description="Files:",
            layout=widgets.Layout(width="600px")
        )
        box_area.children = [
            widgets.Label("Hold Ctrl / ⌘ to select multiple files:"),
            chk_box_group
        ]

    chk_box_group = None
    box_area = widgets.VBox([])

    btn_make   = widgets.Button(description="Make Selection",           button_style="primary", icon="check")
    btn_resel  = widgets.Button(description="Re-Select (Reload Keys)",  button_style="warning", icon="refresh")
    btn_save   = widgets.Button(description="Save",                     button_style="success", icon="save")
    btn_mem    = widgets.Button(description="Save to Memory (joblib)",  button_style="info",    icon="download")
    btn_load   = widgets.Button(description="Load Previous Selection",  button_style="",        icon="upload")

    def _on_make(_):
        _build_ui()
        with out_status:
            clear_output(wait=True)
            print("✅  File list loaded. Select files above then press 'Save'.")

    def _on_resel(_):
        _build_ui()
        with out_status:
            clear_output(wait=True)
            print("🔄  Keys reloaded from data_paths_found.")

    def _on_save(_):
        global selected_file_names
        with out_status:
            clear_output(wait=True)
            if chk_box_group is None or not chk_box_group.value:
                print("⚠️  No files selected.")
                return
            selected_file_names = list(chk_box_group.value)
            print(f"✅  selected_file_names updated ({len(selected_file_names)} file(s)):")
            for f in selected_file_names:
                print(f"   • {f}")

    def _on_save_mem(_):
        with out_status:
            clear_output(wait=True)
            if not selected_file_names:
                print("⚠️  Nothing selected. Press 'Save' first.")
                return
            try:
                joblib.dump(selected_file_names, "selected_files.joblib")
                print("✅  Saved as 'selected_files.joblib'.")
            except Exception as e:
                print(f"❌  Error: {e}")

    def _on_load(_):
        global selected_file_names
        with out_status:
            clear_output(wait=True)
            try:
                selected_file_names = joblib.load("selected_files.joblib")
                print(f"✅  Loaded {len(selected_file_names)} file(s) from 'selected_files.joblib':")
                for f in selected_file_names:
                    print(f"   • {f}")
            except FileNotFoundError:
                print("❌  'selected_files.joblib' not found.")
            except Exception as e:
                print(f"❌  Error: {e}")

    btn_make.on_click(_on_make)
    btn_resel.on_click(_on_resel)
    btn_save.on_click(_on_save)
    btn_mem.on_click(_on_save_mem)
    btn_load.on_click(_on_load)

    ui = widgets.VBox([
        widgets.HTML("<h3>📂 Step 2 — Select Files</h3>"),
        widgets.HBox([btn_make, btn_resel, btn_save, btn_mem, btn_load]),
        box_area,
        out_status,
    ])
    display(ui)


# ─────────────────────────────────────────────
# 3.  simple_plots()
# ─────────────────────────────────────────────

# Plot type configs ────────────────────────────────────────────────────────────

_PLOT_TYPES = [
    "Cell Attached (Voltage Clamp)",
    "Whole Cell (Current Clamp)",
    "Whole Cell (Voltage Clamp)",
    "IH (Voltage Clamp)",
    "IH (Current Clamp)",
]

_PLOT_COLORS = {
    "Cell Attached (Voltage Clamp)": "black",
    "Whole Cell (Current Clamp)":    "darkred",
    "Whole Cell (Voltage Clamp)":    "navy",
    "IH (Voltage Clamp)":            "darkgreen",
    "IH (Current Clamp)":            "purple",
}


def _get_abf_meta(abf) -> str:
    """Return a formatted metadata string for the given abf object."""
    lines = [
        f"File ID        : {abf.abfID}",
        f"Protocol       : {abf.protocol}",
        f"Sample Rate    : {abf.sampleRate} Hz",
        f"Channels       : {abf.channelCount}",
        f"Sweep Count    : {abf.sweepCount}",
        f"Sweep Duration : {abf.sweepLengthSec:.4f} s",
        f"Recording Start: {abf.abfDateTime}",
        f"ADC Units      : {abf.adcUnits}",
    ]
    # Compute overall time span from sweep data
    abf.setSweep(0)
    t_min = float(np.min(abf.sweepX))
    abf.setSweep(abf.sweepCount - 1)
    t_max = float(np.max(abf.sweepX))
    lines.append(f"Time Span      : {t_min:.3f} – {t_max:.3f} s  ({(t_max - t_min)/60:.2f} min)")
    return "\n".join(lines)


def _plot_matplotlib(abf, plot_type: str,
                     x_start: float, x_end: float,
                     y_lower: float, y_upper: float,
                     pad: float = 10.0,
                     save: bool = False, save_fmt: str = "svg",
                     save_dir: str = ""):
    """Render the chosen plot type with Matplotlib."""
    color = _PLOT_COLORS.get(plot_type, "black")
    multi_sweep = plot_type in ("IH (Voltage Clamp)", "IH (Current Clamp)")

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype']  = 42
    try:
        plt.style.use('seaborn-v0_8-paper')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

    if multi_sweep:
        colors_sweep = plt.cm.magma(np.linspace(0, 0.8, abf.sweepCount))
        for i, sw in enumerate(abf.sweepList):
            abf.setSweep(sw, channel=0)
            mask = (abf.sweepX >= x_start) & (abf.sweepX <= x_end)
            ax.plot(abf.sweepX[mask], abf.sweepY[mask],
                    color=colors_sweep[i], linewidth=0.5, alpha=0.9,
                    label=f"Sweep {sw}")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    else:
        abf.setSweep(0)
        mask = (abf.sweepX >= x_start) & (abf.sweepX <= x_end)
        xw = abf.sweepX[mask]
        yw = abf.sweepY[mask]
        # apply y threshold filter
        yw_mask = (yw >= y_lower) & (yw <= y_upper)
        ax.plot(xw[yw_mask], yw[yw_mask], color=color, linewidth=0.5)

    abf.setSweep(0)
    ax.set_title(f"{plot_type}: {abf.abfID}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{abf.sweepUnitsY}")
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_lower - pad, y_upper + pad)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save:
        fname = f"{plot_type.replace(' ', '_').replace('(', '').replace(')', '')}_{abf.abfID}.{save_fmt}"
        fpath = os.path.join(save_dir, fname) if save_dir else fname
        fig.savefig(fpath, dpi=300)
        print(f"✅  Saved: {fpath}")

    plt.show()


def _plot_plotly(abf, plot_type: str,
                 x_start: float, x_end: float,
                 y_lower: float, y_upper: float,
                 pad: float = 10.0,
                 save: bool = False, save_fmt: str = "svg",
                 save_dir: str = ""):
    """Render the chosen plot type with Plotly."""
    multi_sweep = plot_type in ("IH (Voltage Clamp)", "IH (Current Clamp)")

    # Decimation factor – high sample-rate files get decimated more
    n = max(1, abf.sampleRate // 2000)

    fig = go.Figure()

    if multi_sweep:
        import plotly.colors as pc
        palette = pc.sample_colorscale("magma", abf.sweepCount)
        for i, sw in enumerate(abf.sweepList):
            abf.setSweep(sw, channel=0)
            mask = (abf.sweepX >= x_start) & (abf.sweepX <= x_end)
            fig.add_trace(go.Scattergl(
                x=abf.sweepX[mask][::n],
                y=abf.sweepY[mask][::n],
                mode="lines",
                name=f"Sweep {sw}",
                line=dict(width=1),
                opacity=0.9
            ))
    else:
        abf.setSweep(0)
        mask = (abf.sweepX >= x_start) & (abf.sweepX <= x_end)
        xw = abf.sweepX[mask]
        yw = abf.sweepY[mask]
        yw_mask = (yw >= y_lower) & (yw <= y_upper)
        color_hex = _PLOT_COLORS.get(plot_type, "#636EFA")
        fig.add_trace(go.Scattergl(
            x=xw[yw_mask][::n],
            y=yw[yw_mask][::n],
            mode="lines",
            name=plot_type,
            line=dict(color=color_hex, width=1)
        ))

    abf.setSweep(0)
    fig.update_layout(
        title=f"{plot_type}: {abf.abfID}",
        xaxis_title="Time (s)",
        yaxis_title=f"{abf.sweepUnitsY}",
        height=500,
        template="plotly_white",
        xaxis=dict(
            range=[x_start, x_end],
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(range=[y_lower - pad, y_upper + pad]),
        modebar_add=['v1hovermode', 'togglespikecounts']
    )
    fig.show()

    if save:
        fname = f"{plot_type.replace(' ', '_').replace('(', '').replace(')', '')}_{abf.abfID}.{save_fmt}"
        fpath = os.path.join(save_dir, fname) if save_dir else fname
        try:
            pio.write_image(fig, fpath)
            print(f"✅  Saved: {fpath}")
        except Exception as e:
            print(f"❌  Could not save figure: {e}  (kaleido may be required for static export)")


def simple_plots():
    """
    Interactive plotting UI.
    1. Select a file from selected_file_names.
    2. Choose plot type (Cell Attached VC, Whole Cell CC, Whole Cell VC, IH VC, IH CC).
    3. View peak qualities.
    4. Clean-up axes (time window + y-threshold).
    5. Re-plot with updated params.
    6. Optionally download as SVG or EPS.
    """
    out_status  = widgets.Output()
    out_meta    = widgets.Output()
    out_plot    = widgets.Output()

    # ── Widgets ────────────────────────────────────────────────────────
    if not selected_file_names:
        display(widgets.HTML("<b>⚠️  selected_file_names is empty — run select_files() first.</b>"))
        return

    dd_file = widgets.Dropdown(
        options=selected_file_names,
        description="File:",
        layout=widgets.Layout(width="400px")
    )
    dd_type = widgets.Dropdown(
        options=_PLOT_TYPES,
        description="Plot Type:",
        layout=widgets.Layout(width="360px")
    )
    dd_lib = widgets.Dropdown(
        options=["Matplotlib", "Plotly"],
        description="Library:",
        layout=widgets.Layout(width="220px")
    )

    btn_qualities = widgets.Button(description="Peak Qualities", button_style="info",    icon="info")
    btn_plot      = widgets.Button(description="Plot",           button_style="primary", icon="bar-chart")
    btn_cleanup   = widgets.Button(description="Clean Up",       button_style="warning", icon="filter")
    btn_replot    = widgets.Button(description="Re-Plot",        button_style="success", icon="refresh")

    # Cleanup / axis controls (hidden until btn_cleanup pressed)
    float_xstart = widgets.FloatText(description="X Start (s):", layout=widgets.Layout(width="220px"))
    float_xend   = widgets.FloatText(description="X End (s):",   layout=widgets.Layout(width="220px"))
    float_ylower = widgets.FloatText(description="Y Lower:",     layout=widgets.Layout(width="220px"))
    float_yupper = widgets.FloatText(description="Y Upper:",     layout=widgets.Layout(width="220px"))
    float_pad    = widgets.FloatText(description="Y Pad:",       value=10.0, layout=widgets.Layout(width="220px"))

    cleanup_box = widgets.VBox([
        widgets.HTML("<b>🔧 Axis Clean-Up Controls</b>"),
        widgets.HBox([float_xstart, float_xend]),
        widgets.HBox([float_ylower, float_yupper, float_pad]),
    ])
    cleanup_box.layout.display = "none"

    # Download option
    chk_download = widgets.Checkbox(value=False, description="Download figure", indent=False)
    dd_fmt = widgets.Dropdown(
        options=["svg", "eps"],
        description="Format:",
        layout=widgets.Layout(width="180px", display="none")
    )

    def _toggle_fmt(change):
        dd_fmt.layout.display = "" if change["new"] else "none"

    chk_download.observe(_toggle_fmt, names="value")

    # ── Internal helpers ────────────────────────────────────────────────
    _abf_cache = {}  # {key: abf object}

    def _load_abf(key: str):
        if key not in _abf_cache:
            path = data_paths_found.get(key, "")
            if not path:
                raise FileNotFoundError(f"Path not found for key '{key}'")
            _abf_cache[key] = pyabf.ABF(path)
        return _abf_cache[key]

    def _default_axes(abf):
        """Compute default x/y limits from sweep 0."""
        abf.setSweep(0)
        x = abf.sweepX
        y = abf.sweepY
        return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))

    def _on_qualities(_):
        with out_meta:
            clear_output(wait=True)
            try:
                abf = _load_abf(dd_file.value)
                print(_get_abf_meta(abf))
            except Exception as e:
                print(f"❌  {e}")

    def _do_plot(use_cleanup_vals: bool):
        with out_status:
            clear_output(wait=True)
        with out_plot:
            clear_output(wait=True)
            try:
                abf = _load_abf(dd_file.value)
                x0, x1, y0, y1 = _default_axes(abf)

                if use_cleanup_vals:
                    xs = float_xstart.value if float_xstart.value != 0 else x0
                    xe = float_xend.value   if float_xend.value   != 0 else x1
                    yl = float_ylower.value if float_ylower.value != 0 else y0
                    yu = float_yupper.value if float_yupper.value != 0 else y1
                else:
                    xs, xe, yl, yu = x0, x1, y0, y1
                    # Populate cleanup widgets with defaults
                    float_xstart.value = round(xs, 4)
                    float_xend.value   = round(xe, 4)
                    float_ylower.value = round(yl, 4)
                    float_yupper.value = round(yu, 4)

                pad      = float_pad.value
                save     = chk_download.value
                save_fmt = dd_fmt.value
                save_dir = str(Path(data_paths_found.get(dd_file.value, "")).parent)

                if dd_lib.value == "Matplotlib":
                    _plot_matplotlib(abf, dd_type.value, xs, xe, yl, yu, pad, save, save_fmt, save_dir)
                else:
                    _plot_plotly(abf, dd_type.value, xs, xe, yl, yu, pad, save, save_fmt, save_dir)

            except Exception as e:
                with out_status:
                    print(f"❌  Plotting error: {e}")

    def _on_plot(_):
        _do_plot(use_cleanup_vals=False)

    def _on_cleanup(_):
        cleanup_box.layout.display = ""
        # Make sure defaults are populated
        try:
            abf = _load_abf(dd_file.value)
            x0, x1, y0, y1 = _default_axes(abf)
            float_xstart.value = round(x0, 4)
            float_xend.value   = round(x1, 4)
            float_ylower.value = round(y0, 4)
            float_yupper.value = round(y1, 4)
        except Exception:
            pass

    def _on_replot(_):
        _do_plot(use_cleanup_vals=True)

    btn_qualities.on_click(_on_qualities)
    btn_plot.on_click(_on_plot)
    btn_cleanup.on_click(_on_cleanup)
    btn_replot.on_click(_on_replot)

    ui = widgets.VBox([
        widgets.HTML("<h3>📊 Step 3 — Simple Plots</h3>"),
        widgets.HBox([dd_file, dd_type, dd_lib]),
        widgets.HBox([btn_qualities, btn_plot, btn_cleanup, btn_replot]),
        widgets.HBox([chk_download, dd_fmt]),
        out_status,
        out_meta,
        cleanup_box,
        out_plot,
    ])
    display(ui)
