#!/usr/bin/env python3
"""
generate_figures.py  –  Publication-quality visualization pipeline for
network cognition simulation results.

Usage:
    python generate_figures.py <survey_dir> [<survey_dir2> ...] [--output-dir DIR]

Each <survey_dir> is the path to a run's survey/ directory containing the
analysis CSVs produced by analyze_survey.py.  Multiple directories can be
supplied to overlay runs (Figs 1–2) or compare conditions side-by-side
(Figs 3–5).

Output files per figure (PDF + PNG):
    fig1_network_structure
    fig2_micro_tie_accuracy
    fig3_centrality_bridge
    fig4_shock_delta_summary
    fig5_retrieval_diagnostics
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


# ── style ──────────────────────────────────────────────────────────────────


def _apply_style(script_dir: Path) -> None:
    style_path = script_dir / "style.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))
    else:
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.labelsize": 10,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.titlesize": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "lines.linewidth": 1.5,
            }
        )


# ── palette ────────────────────────────────────────────────────────────────

# Okabe-Ito colorblind-safe palette (8 colors)
COLORS = sns.color_palette("colorblind", 8)

TRUTH_LABELS: dict[str, str] = {
    "observed_interaction": "Observed interaction",
    "background_social_tie": "Background social tie",
    "background_or_interaction": "Background or interaction",
}


# ── data loading ───────────────────────────────────────────────────────────


def _csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV file; return None on any error or if missing."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"  [warn] cannot read {path.name}: {exc}", file=sys.stderr)
        return None


def _shock_step(survey_dir: Path) -> Optional[int]:
    """Return the shock step from shock_log.jsonl, or None if absent."""
    p = survey_dir / "shock_log.jsonl"
    if not p.exists():
        return None
    try:
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        if not lines:
            return None
        for line in lines:
            obj = json.loads(line)
            if obj.get("command", "").startswith("shock_isolate"):
                return int(obj["step"])
        return None
    except Exception as exc:
        print(f"  [warn] shock_log.jsonl: {exc}", file=sys.stderr)
        return None


def load_run(survey_dir: Path) -> dict:
    """
    Load all analysis CSVs for one survey directory into a dict.

    Keys: label, survey_dir, network, micro_tie, centrality, bridge,
          prepost, retrieval, shock_step.
    Values are DataFrames (or None when the file is absent/unreadable).
    """
    d = Path(survey_dir).resolve()

    ppost = _csv(d / "shock_aligned_prepost_summary.csv")

    # Derive a human-readable run label
    label: Optional[str] = None
    if ppost is not None and "treatment_type" in ppost.columns:
        label = str(ppost["treatment_type"].iloc[0])
    if label is None:
        p = d / "shock_log.jsonl"
        if p.exists():
            try:
                for _line in p.read_text().splitlines():
                    _line = _line.strip()
                    if not _line:
                        continue
                    obj = json.loads(_line)
                    if obj.get("command", "").startswith("shock_isolate"):
                        label = obj.get("treatment_type")
                        break
            except Exception:
                pass
    if label is None:
        label = d.parent.name if d.name == "survey" else d.name

    return dict(
        label=label,
        survey_dir=d,
        network=_csv(d / "network_summary_over_time.csv"),
        micro_tie=_csv(d / "analysis_micro_tie_metrics_by_truth.csv"),
        centrality=_csv(d / "shock_aligned_centrality_self_position.csv"),
        bridge=_csv(d / "analysis_bridge_rank_metrics.csv"),
        prepost=ppost,
        retrieval=_csv(d / "shock_aligned_retrieval_quality.csv"),
        shock_step=_shock_step(d),
    )


# ── save helper ────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, outdir: Path, stem: str) -> list[str]:
    """Save figure as both PDF and PNG; return list of written paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for ext in ("pdf", "png"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        saved.append(str(p))
    plt.close(fig)
    return saved


# ── common drawing helpers ─────────────────────────────────────────────────


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _shock_line(
    ax: plt.Axes, step: Optional[int], annotate: bool = False
) -> None:
    """Draw a vertical dotted line at the shock step."""
    if step is None:
        return
    ax.axvline(step, color="0.55", linestyle=":", linewidth=1.0, zorder=0)
    if annotate:
        ax.text(
            step,
            1.02,
            "shock",
            transform=ax.get_xaxis_transform(),
            fontsize=6,
            color="0.5",
            va="bottom",
            ha="center",
        )


def _fval(df_slice: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Safely extract a scalar float from the first row of a column."""
    if col in df_slice.columns and not df_slice.empty:
        v = df_slice[col].iloc[0]
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return default
            try:
                return float(v)
            except (ValueError, TypeError):
                return default
        try:
            return float(v) if not pd.isna(v) else default
        except (ValueError, TypeError):
            return default
    return default


def _grouped_bars(
    ax: plt.Axes,
    groups: list[str],
    series_labels: list[str],
    series_values: list[list[float]],
    colors: list,
    hatches: Optional[list[str]] = None,
) -> None:
    """
    Draw a grouped bar chart.

    groups        – x-tick labels (one per metric group)
    series_labels – legend labels (one per data series)
    series_values – list of value-lists, one list per series
    colors        – face colors, one per series
    hatches       – optional hatch patterns, one per series
    """
    n_s = len(series_labels)
    if n_s == 0:
        return
    bar_w = min(0.75 / n_s, 0.32)
    x = np.arange(len(groups))
    for si in range(n_s):
        offset = bar_w * (si - (n_s - 1) / 2)
        h = hatches[si] if hatches else ""
        ax.bar(
            x + offset,
            series_values[si],
            bar_w,
            color=colors[si],
            edgecolor="black",
            linewidth=0.5,
            hatch=h,
            label=series_labels[si],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)


# ── Figure 1: Network Structure Over Time ─────────────────────────────────


def fig1_network_structure(runs: list[dict], outdir: Path) -> list[str]:
    """
    Panel A: Network density over time.
    Panel B: Average clustering coefficient (skipped if data absent).
    Panel C: Louvain modularity (skipped if data absent).
    Vertical dashed line at shock step; multiple runs overlaid.
    """
    _PANELS = [
        ("density_cumulative", "Network density"),
        ("avg_clustering", "Avg. clustering coeff."),
        ("modularity_louvain", "Modularity (Louvain)"),
    ]

    active = [
        (col, lbl)
        for col, lbl in _PANELS
        if any(
            r["network"] is not None
            and col in r["network"].columns
            and r["network"][col].notna().any()
            for r in runs
        )
    ]
    if not active:
        print("  [skip] Fig 1: no usable network data", file=sys.stderr)
        return []

    np_ = len(active)
    fig, ax_row = plt.subplots(
        1, np_, figsize=(7 if np_ > 1 else 3.5, 2.8), squeeze=False
    )
    axes = ax_row[0]

    ls_cycle = ["-", "--", "-.", ":"]

    for ri, run in enumerate(runs):
        df = run["network"]
        if df is None:
            continue
        for ai, (col, _) in enumerate(active):
            if col not in df.columns or df[col].isna().all():
                continue
            axes[ai].plot(
                df["step"],
                df[col],
                color=COLORS[ri % len(COLORS)],
                linestyle=ls_cycle[ri % len(ls_cycle)],
                linewidth=1.5,
                marker="o",
                markersize=3,
                label=run["label"],
            )

    # Shock lines – annotate text only on first panel
    shock_steps: set[int] = set()
    for run in runs:
        if run["shock_step"] is not None:
            for ai, ax in enumerate(axes):
                _shock_line(ax, run["shock_step"], annotate=(ai == 0))
            shock_steps.add(run["shock_step"])

    for ax, (_, ylbl) in zip(axes, active):
        ax.set_xlabel("Simulation step", fontsize=10)
        ax.set_ylabel(ylbl, fontsize=10)
        ax.tick_params(labelsize=8)
        _despine(ax)

    axes[0].set_title("Network structure over time", fontsize=11, loc="left")

    # Build legend with optional shock handle
    handles, labels_ = axes[0].get_legend_handles_labels()
    if shock_steps:
        handles.append(Line2D([0], [0], color="0.55", linestyle=":", linewidth=1))
        labels_.append("Shock")
    if handles:
        axes[0].legend(handles, labels_, fontsize=7, frameon=False)

    fig.tight_layout()
    return _save(fig, outdir, "fig1_network_structure")


# ── Figure 2: Micro-Tie Perception Accuracy Over Time ─────────────────────


def fig2_micro_tie_accuracy(runs: list[dict], outdir: Path) -> list[str]:
    """
    Panel A: Precision by wave, one line per truth layer.
    Panel B: Recall by wave.
    Panel C: False positive rate by wave.
    Shaded bands = ±1 SE across respondents.
    Vertical dashed line at shock step; multiple runs with distinct line styles.
    """
    _METRICS = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("fpr", "False positive rate"),
    ]
    _LAYERS = list(TRUTH_LABELS.keys())
    layer_col = {ly: COLORS[i] for i, ly in enumerate(_LAYERS)}
    ls_cycle = ["-", "--", "-.", ":"]

    fig, ax_row = plt.subplots(1, 3, figsize=(7, 3.0), squeeze=True)
    any_plotted = False

    for ri, run in enumerate(runs):
        df = run["micro_tie"]
        if df is None:
            continue
        ls = ls_cycle[ri % len(ls_cycle)]
        first_run = ri == 0

        for ly in _LAYERS:
            ly_df = df[df["truth_layer"] == ly]
            overall = ly_df[ly_df["respondent"] == "__overall__"].sort_values("step")
            indiv = ly_df[ly_df["respondent"] != "__overall__"]
            if overall.empty:
                continue

            steps = overall["step"].values
            color = layer_col[ly]

            for ax, (col, _) in zip(ax_row, _METRICS):
                if col not in overall.columns:
                    continue
                mean_v = overall[col].to_numpy(dtype=float)

                # SE = std(individual respondent values) / sqrt(n)
                se_v = np.zeros(len(steps))
                for si, s in enumerate(steps):
                    vals = indiv.loc[indiv["step"] == s, col].dropna()
                    if len(vals) > 1:
                        se_v[si] = vals.std() / np.sqrt(len(vals))

                lbl = TRUTH_LABELS.get(ly, ly) if first_run else None
                ax.plot(
                    steps,
                    mean_v,
                    color=color,
                    linestyle=ls,
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                    label=lbl,
                )
                ax.fill_between(
                    steps, mean_v - se_v, mean_v + se_v, color=color, alpha=0.15
                )
                any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print("  [skip] Fig 2: no micro-tie data", file=sys.stderr)
        return []

    shock_steps: set[int] = set()
    for run in runs:
        if run["shock_step"] is not None:
            for ai, ax in enumerate(ax_row):
                _shock_line(ax, run["shock_step"], annotate=(ai == 0))
            shock_steps.add(run["shock_step"])

    for ax, (_, ylbl) in zip(ax_row, _METRICS):
        ax.set_xlabel("Simulation step", fontsize=10)
        ax.set_ylabel(ylbl, fontsize=10)
        ax.tick_params(labelsize=8)
        _despine(ax)

    ax_row[0].set_title("Micro-tie perception accuracy", fontsize=11, loc="left")

    handles, labels_ = ax_row[0].get_legend_handles_labels()
    if shock_steps:
        handles.append(Line2D([0], [0], color="0.55", linestyle=":", linewidth=1))
        labels_.append("Shock")
    if handles:
        ax_row[0].legend(handles, labels_, fontsize=7, frameon=False)

    fig.tight_layout()
    return _save(fig, outdir, "fig2_micro_tie_accuracy")


# ── Figure 3: Centrality and Bridge Perception ────────────────────────────


def fig3_centrality_bridge(runs: list[dict], outdir: Path) -> list[str]:
    """
    Panel A: Centrality top-1 and top-3 hit rates, pre vs post (grouped bar).
    Panel B: Bridge top-1 and top-3 hit rates, overall (grouped bar).
             Skipped if no run has analysis_bridge_rank_metrics.csv.
    Panel C: Self-position mean absolute rank error, pre vs post.
    Panels A and C require shock_aligned_centrality_self_position.csv.
    """
    has_cent = any(r["centrality"] is not None for r in runs)
    has_brdg = any(r["bridge"] is not None for r in runs)
    if not has_cent and not has_brdg:
        print("  [skip] Fig 3: no centrality or bridge data", file=sys.stderr)
        return []

    panels: list[str] = []
    if has_cent:
        panels.append("centrality")
    if has_brdg:
        panels.append("bridge")
    if has_cent:
        panels.append("self_position")

    np_ = len(panels)
    fig_w = min(7.0, 2.5 * np_)
    fig, ax_row = plt.subplots(1, np_, figsize=(fig_w, 3.2), squeeze=False)
    panel_ax = dict(zip(panels, ax_row[0]))

    pre_pal = sns.color_palette("pastel", n_colors=max(len(runs), 1))
    post_pal = sns.color_palette("colorblind", n_colors=max(len(runs), 1))
    multi = len(runs) > 1

    # ── Panel A: centrality top-k hit rates ──────────────────────────────
    if "centrality" in panel_ax:
        ax = panel_ax["centrality"]
        grps = ["Top-1 hit rate", "Top-3 hit rate"]
        lbls: list[str] = []
        vals: list[list[float]] = []
        cols_: list = []

        for ri, run in enumerate(runs):
            df = run["centrality"]
            if df is None:
                continue
            pre = df[df["phase"] == "pre"]
            post = df[df["phase"] == "post"]
            if pre.empty or post.empty:
                continue
            lbls += [
                f"{run['label']} pre" if multi else "Pre-shock",
                f"{run['label']} post" if multi else "Post-shock",
            ]
            vals += [
                [
                    _fval(pre, "centrality_top1_hit_rate"),
                    _fval(pre, "centrality_top3_hit_rate"),
                ],
                [
                    _fval(post, "centrality_top1_hit_rate"),
                    _fval(post, "centrality_top3_hit_rate"),
                ],
            ]
            cols_ += [pre_pal[ri], post_pal[ri]]

        if lbls:
            _grouped_bars(ax, grps, lbls, vals, cols_)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Hit rate", fontsize=10)
            ax.set_title("Centrality perception", fontsize=11, loc="left")
            ax.legend(fontsize=7, frameon=False)
            _despine(ax)

    # ── Panel B: bridge hit rates ─────────────────────────────────────────
    if "bridge" in panel_ax:
        ax = panel_ax["bridge"]
        grps = ["Top-1 hit rate", "Top-3 hit rate"]
        lbls_b: list[str] = []
        vals_b: list[list[float]] = []
        cols_b: list = []

        for ri, run in enumerate(runs):
            df = run["bridge"]
            if df is None:
                continue
            ov = df[df["respondent"] == "__overall__"]
            if ov.empty:
                continue
            lbls_b.append(run["label"] if multi else "Overall")
            vals_b.append(
                [_fval(ov, "bridge_top1_hit"), _fval(ov, "bridge_top3_hit")]
            )
            cols_b.append(post_pal[ri])

        if lbls_b:
            _grouped_bars(ax, grps, lbls_b, vals_b, cols_b)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Hit rate", fontsize=10)
            ax.set_title("Bridge perception", fontsize=11, loc="left")
            if len(lbls_b) > 1:
                ax.legend(fontsize=7, frameon=False)
            _despine(ax)

    # ── Panel C: self-position rank error ─────────────────────────────────
    if "self_position" in panel_ax:
        ax = panel_ax["self_position"]
        grps_sp = ["Self-position"]
        lbls_sp: list[str] = []
        vals_sp: list[list[float]] = []
        cols_sp: list = []

        for ri, run in enumerate(runs):
            df = run["centrality"]
            if df is None:
                continue
            pre = df[df["phase"] == "pre"]
            post = df[df["phase"] == "post"]
            if pre.empty or post.empty:
                continue
            lbls_sp += [
                f"{run['label']} pre" if multi else "Pre-shock",
                f"{run['label']} post" if multi else "Post-shock",
            ]
            vals_sp += [
                [_fval(pre, "self_position_mean_abs_rank_error")],
                [_fval(post, "self_position_mean_abs_rank_error")],
            ]
            cols_sp += [pre_pal[ri], post_pal[ri]]

        if lbls_sp:
            _grouped_bars(ax, grps_sp, lbls_sp, vals_sp, cols_sp)
            ax.set_ylabel("Mean abs. rank error", fontsize=10)
            ax.set_title("Self-position accuracy", fontsize=11, loc="left")
            ax.legend(fontsize=7, frameon=False)
            _despine(ax)

    fig.tight_layout()
    return _save(fig, outdir, "fig3_centrality_bridge")


# ── Figure 4: Pre/Post Shock Delta Summary ────────────────────────────────

_DELTA_METRICS: list[tuple[str, str, str]] = [
    # (csv_column,                              display_name,                 improve_dir)
    ("union_delta_precision", "Precision (union)", "+"),
    ("union_delta_recall", "Recall (union)", "+"),
    ("union_delta_fpr", "False positive rate", "-"),
    ("delta_centrality_top1_hit_rate", "Centrality top-1 hit", "+"),
    ("delta_self_position_mean_abs_rank_error", "Self-position rank error", "-"),
]


def fig4_shock_delta_summary(runs: list[dict], outdir: Path) -> list[str]:
    """
    Horizontal diverging bar chart of post-minus-pre deltas for key metrics.
    Filled bars = improvement direction; hatched bars = degradation.
    Multiple runs shown side-by-side (different colors).
    """
    valid = [r for r in runs if r["prepost"] is not None]
    if not valid:
        print("  [skip] Fig 4: no prepost_summary data", file=sys.stderr)
        return []

    n_m = len(_DELTA_METRICS)
    n_r = len(valid)
    bh = min(0.50 / max(n_r, 1), 0.22)

    fig_h = max(3.0, 0.7 * n_m + 1.2)
    fig, ax = plt.subplots(figsize=(3.5, fig_h))

    run_colors = sns.color_palette("colorblind", n_colors=max(n_r, 1))

    for ri, run in enumerate(valid):
        row = run["prepost"].iloc[0]
        y_off = (ri - (n_r - 1) / 2) * bh * 1.1
        color = run_colors[ri]

        for mi, (col, name, direction) in enumerate(_DELTA_METRICS):
            if col not in row.index or pd.isna(row[col]):
                continue
            delta = float(row[col])
            y = float(mi) + y_off

            good = (direction == "+" and delta >= 0) or (
                direction == "-" and delta <= 0
            )
            face = color if good else "white"
            hatch = "" if good else "///"

            ax.barh(
                y,
                delta,
                bh,
                color=face,
                edgecolor=color,
                linewidth=0.8,
                hatch=hatch,
                label=run["label"] if mi == 0 else None,
            )

            # Inline value annotation
            x_off = 0.008 if delta >= 0 else -0.008
            ax.text(
                delta + x_off,
                y,
                f"{delta:+.3f}",
                va="center",
                ha="left" if delta >= 0 else "right",
                fontsize=6,
                color="0.35",
            )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([m[1] for m in _DELTA_METRICS], fontsize=8)
    ax.set_xlabel("Delta (post \u2212 pre)", fontsize=10)
    ax.set_title("Pre/post shock change", fontsize=11, loc="left")
    ax.tick_params(labelsize=8)
    ax.margins(x=0.20)
    _despine(ax)

    if n_r > 1:
        ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax.text(
        0.99,
        -0.11,
        "Filled = improvement; hatched = degradation",
        transform=ax.transAxes,
        fontsize=6,
        ha="right",
        color="0.5",
    )

    fig.tight_layout()
    return _save(fig, outdir, "fig4_shock_delta_summary")


# ── Figure 5: Retrieval Diagnostics ───────────────────────────────────────


def fig5_retrieval_diagnostics(runs: list[dict], outdir: Path) -> list[str]:
    """
    Panel A: Mean n_returned, pre vs post.
    Panel B: Query–target mention rate, pre vs post.
    Panel C: Fail-safe rate, pre vs post.
    Source: shock_aligned_retrieval_quality.csv (pre/post rows).
    """
    _METRICS = [
        ("mean_n_returned", "Mean n returned"),
        ("micro_query_target_mention_rate", "Query–target mention rate"),
        ("micro_tie_fail_safe_rate", "Fail-safe rate"),
    ]
    valid = [r for r in runs if r["retrieval"] is not None]
    if not valid:
        print("  [skip] Fig 5: no retrieval data", file=sys.stderr)
        return []

    fig, ax_row = plt.subplots(1, 3, figsize=(7, 2.8), squeeze=True)

    pre_pal = sns.color_palette("pastel", n_colors=max(len(valid), 1))
    post_pal = sns.color_palette("colorblind", n_colors=max(len(valid), 1))
    bw = 0.30
    multi = len(valid) > 1

    for ri, run in enumerate(valid):
        df = run["retrieval"]
        pre = df[df["phase"] == "pre"]
        post = df[df["phase"] == "post"]
        if pre.empty or post.empty:
            continue

        off = (ri - (len(valid) - 1) / 2) * bw

        for ai, (col, _) in enumerate(_METRICS):
            ax = ax_row[ai]
            if col not in df.columns:
                continue
            pv = _fval(pre, col)
            ptv = _fval(post, col)

            ax.bar(
                0 + off,
                pv,
                bw,
                color=pre_pal[ri],
                edgecolor="black",
                linewidth=0.5,
                label=(
                    (f"{run['label']} pre" if multi else "Pre-shock")
                    if ai == 0
                    else None
                ),
            )
            ax.bar(
                1 + off,
                ptv,
                bw,
                color=post_pal[ri],
                edgecolor="black",
                linewidth=0.5,
                label=(
                    (f"{run['label']} post" if multi else "Post-shock")
                    if ai == 0
                    else None
                ),
            )

    for ai, (_, ylbl) in enumerate(_METRICS):
        ax = ax_row[ai]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"], fontsize=8)
        ax.set_ylabel(ylbl, fontsize=10)
        ax.tick_params(labelsize=8)
        _despine(ax)

    ax_row[0].set_title("Retrieval diagnostics", fontsize=11, loc="left")
    handles, labels_ = ax_row[0].get_legend_handles_labels()
    if handles:
        ax_row[0].legend(handles, labels_, fontsize=7, frameon=False)

    fig.tight_layout()
    return _save(fig, outdir, "fig5_retrieval_diagnostics")


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate publication-quality figures from network cognition survey data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single run:\n"
            "  python generate_figures.py path/to/survey/\n\n"
            "  # Two conditions overlaid, custom output dir:\n"
            "  python generate_figures.py hub/survey/ broker/survey/ --output-dir ./figs/"
        ),
    )
    ap.add_argument(
        "survey_dirs",
        nargs="+",
        metavar="SURVEY_DIR",
        help="One or more survey/ directories containing analysis CSVs.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory for figures (default: <first_survey_dir>/figures/).",
    )
    args = ap.parse_args()

    _apply_style(Path(__file__).parent)

    runs: list[dict] = []
    for sd in args.survey_dirs:
        p = Path(sd)
        if not p.exists():
            sys.exit(f"[error] directory not found: {p}")
        print(f"Loading {p} ...", flush=True)
        run = load_run(p)
        print(f"  label={run['label']}  shock_step={run['shock_step']}")
        runs.append(run)

    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(args.survey_dirs[0]) / "figures")
    )
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput → {outdir}\n")

    generators = [
        ("Fig 1 – Network Structure", fig1_network_structure),
        ("Fig 2 – Micro-Tie Accuracy", fig2_micro_tie_accuracy),
        ("Fig 3 – Centrality & Bridge", fig3_centrality_bridge),
        ("Fig 4 – Shock Delta Summary", fig4_shock_delta_summary),
        ("Fig 5 – Retrieval Diagnostics", fig5_retrieval_diagnostics),
    ]

    all_paths: list[str] = []
    for name, fn in generators:
        print(f"  {name} ...", end=" ", flush=True)
        try:
            paths = fn(runs, outdir)
        except Exception:
            print("ERROR")
            traceback.print_exc()
            continue
        if paths:
            print(f"→ {len(paths)} file(s)")
            all_paths.extend(paths)
        else:
            print("(skipped — insufficient data)")

    print(f"\n{'─' * 52}")
    print(f"{len(all_paths)} file(s) written to {outdir}:")
    for p in all_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
