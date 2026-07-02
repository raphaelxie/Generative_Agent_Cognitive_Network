#!/usr/bin/env python3
"""
Publication figures for the NCN identification experiment.

Reads summary CSVs from the baseline survey directory and network structure
logs from each identification fork, then writes PDF + PNG figures.

Usage (from reverie/backend_server/):
  python figures/ncn_id_figures.py
  python figures/ncn_id_figures.py --survey-dir path/to/survey --output-dir path/to/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
FS_STORAGE = BACKEND_DIR / "../../environment/frontend_server/storage"
DEFAULT_BASELINE = "ncn_baseline_n25_full"
SHOCK_STEP = 2100

FORKS = [
    ("control", 1, "ncn_id_control_r1"),
    ("control", 2, "ncn_id_control_r2"),
    ("hub", 1, "ncn_id_hub_r1"),
    ("hub", 2, "ncn_id_hub_r2"),
    ("broker", 1, "ncn_id_broker_r1"),
    ("broker", 2, "ncn_id_broker_r2"),
]

TREATMENT_COLORS = {
    "control": "#999999",
    "hub": "#E69F00",
    "broker": "#0072B2",
}

PRIMARY_OUTCOMES = [
    (
        "micro_tie",
        "observed_interaction_recent",
        "pair_accuracy",
        "Micro-tie accuracy",
    ),
    (
        "bridge_rank",
        "observed_interaction_recent_betweenness_rank",
        "bridge_mean_abs_rank_error",
        "Bridge rank error",
    ),
    (
        "community_nmi",
        "observed_interaction_recent_louvain_communities",
        "nmi",
        "Community NMI",
    ),
    (
        "shocked_agent_rank_hub",
        "John Lin",
        "mean_perceived_bridge_rank",
        "Hub rank (John Lin)",
    ),
    (
        "shocked_agent_rank_broker",
        "Eddy Lin",
        "mean_perceived_bridge_rank",
        "Broker rank (Eddy Lin)",
    ),
]

METRIC_FILES = {
    "micro_tie": "analysis_micro_tie_metrics_by_truth.csv",
    "bridge_rank": "analysis_bridge_rank_metrics.csv",
    "community_nmi": "analysis_community_group_metrics.csv",
}

METRIC_FILTER_COL = {
    "micro_tie": "truth_layer",
    "bridge_rank": "truth_definition",
    "community_nmi": "truth_definition",
}


def _apply_style() -> None:
    style_path = SCRIPT_DIR / "style.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


def _save(fig: plt.Figure, outdir: Path, stem: str) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for ext in ("pdf", "png"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        saved.append(str(p))
    plt.close(fig)
    return saved


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _num(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _did_row(did_df: pd.DataFrame, instrument: str, truth: str) -> pd.Series | None:
    mask = (did_df["instrument"] == instrument) & (did_df["truth_layer"] == truth)
    rows = did_df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _delta_subset(delta_df: pd.DataFrame, instrument: str, truth: str) -> pd.DataFrame:
    return delta_df[
        (delta_df["instrument"] == instrument)
        & (delta_df["truth_layer"] == truth)
        & delta_df["delta"].notna()
    ].copy()


def _noise_rep_values(
    survey_dir: Path, instrument: str, truth: str, metric: str, noise_waves: list[str]
) -> list[float]:
    if instrument.startswith("shocked_agent_rank_"):
        agent = truth
        vals = []
        for wave in noise_waves:
            path = survey_dir / f"perception_survey_{wave}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            ranks = df[
                (df["question_type"] == "bridge_rank")
                & (df["target"] == agent)
                & (df["is_fail_safe"] != 1)
            ]["value"]
            if len(ranks):
                vals.append(float(ranks.astype(float).mean()))
        return vals

    filename = METRIC_FILES.get(instrument)
    if not filename:
        return []
    df = _load_csv(survey_dir / filename)
    if df.empty:
        return []
    filter_col = METRIC_FILTER_COL[instrument]
    vals = []
    for wave in noise_waves:
        row = df[
            (df["wave_id"] == wave)
            & (df["respondent"] == "__overall__")
            & (df[filter_col] == truth)
        ]
        if not row.empty and metric in row.columns:
            v = _num(row.iloc[0][metric])
            if v is not None:
                vals.append(v)
    return vals


def fig_id1_did_forest(
    did_df: pd.DataFrame, outdir: Path
) -> list[str]:
    """Forest plot of replicated DiD vs ±2×noise_std threshold."""
    entries = []
    for instrument, truth, _metric, label in PRIMARY_OUTCOMES:
        row = _did_row(did_df, instrument, truth)
        if row is None:
            continue
        noise_std = _num(row.get("noise_std"))
        threshold = 2.0 * noise_std if noise_std is not None else None

        if instrument == "shocked_agent_rank_hub":
            entries.append(
                dict(
                    label=label,
                    treatment="hub",
                    did=_num(row.get("hub_did_mean")),
                    r1=_num(row.get("hub_did_r1")),
                    r2=_num(row.get("hub_did_r2")),
                    threshold=threshold,
                )
            )
        elif instrument == "shocked_agent_rank_broker":
            entries.append(
                dict(
                    label=label,
                    treatment="broker",
                    did=_num(row.get("broker_did_mean")),
                    r1=_num(row.get("broker_did_r1")),
                    r2=_num(row.get("broker_did_r2")),
                    threshold=threshold,
                )
            )
        else:
            entries.append(
                dict(
                    label=label,
                    treatment="hub",
                    did=_num(row.get("hub_did_mean")),
                    r1=_num(row.get("hub_did_r1")),
                    r2=_num(row.get("hub_did_r2")),
                    threshold=threshold,
                )
            )
            entries.append(
                dict(
                    label=label,
                    treatment="broker",
                    did=_num(row.get("broker_did_mean")),
                    r1=_num(row.get("broker_did_r1")),
                    r2=_num(row.get("broker_did_r2")),
                    threshold=threshold,
                )
            )

    if not entries:
        return []

    # Unique y positions per outcome label
    labels = []
    for inst, truth, _, lbl in PRIMARY_OUTCOMES:
        if lbl not in labels:
            labels.append(lbl)

    fig_h = max(3.5, 0.55 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))

    y_base = {lbl: i for i, lbl in enumerate(reversed(labels))}
    offsets = {"hub": 0.12, "broker": -0.12}

    for e in entries:
        if e["did"] is None:
            continue
        y = y_base[e["label"]] + offsets[e["treatment"]]
        color = TREATMENT_COLORS[e["treatment"]]
        ax.plot(
            e["did"],
            y,
            "D",
            color=color,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.4,
            zorder=3,
        )
        for rv in (e["r1"], e["r2"]):
            if rv is not None:
                ax.plot(rv, y, "o", color=color, markersize=4, alpha=0.55, zorder=2)

        if e["threshold"] is not None:
            ax.plot(
                [-e["threshold"], e["threshold"]],
                [y, y],
                color="0.75",
                linewidth=4,
                solid_capstyle="butt",
                zorder=1,
            )

    ax.axvline(0, color="black", linewidth=0.8, zorder=0)
    ax.set_yticks(list(y_base.values()))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel("Difference-in-differences (treatment Δ − control Δ)")
    ax.set_title("Replicated treatment effects vs measurement noise band", loc="left")

    legend_handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=TREATMENT_COLORS["hub"],
               markeredgecolor="black", markersize=7, label="Hub DiD (mean)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=TREATMENT_COLORS["broker"],
               markeredgecolor="black", markersize=7, label="Broker DiD (mean)"),
        Line2D([0], [0], marker="o", color="0.5", linestyle="None", markersize=4,
               label="Replicate DiD"),
        Patch(facecolor="0.75", edgecolor="none", label="±2× noise std"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right")
    _despine(ax)
    fig.tight_layout()
    return _save(fig, outdir, "fig_id1_did_forest")


def fig_id2_treatment_deltas(
    delta_df: pd.DataFrame, outdir: Path
) -> list[str]:
    """Grouped pre→post deltas by treatment arm with replicate points."""
    panels = []
    for instrument, truth, _metric, label in PRIMARY_OUTCOMES:
        sub = _delta_subset(delta_df, instrument, truth)
        if sub.empty:
            continue
        panels.append((label, sub))

    if not panels:
        return []

    n_p = len(panels)
    fig, axes = plt.subplots(1, n_p, figsize=(2.2 * n_p, 3.2), squeeze=False)
    treatments = ["control", "hub", "broker"]

    for ax, (title, sub) in zip(axes[0], panels):
        means = []
        for trt in treatments:
            vals = sub.loc[sub["treatment"] == trt, "delta"].astype(float).tolist()
            means.append(np.mean(vals) if vals else np.nan)
            x = treatments.index(trt)
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                color=TREATMENT_COLORS[trt],
                s=22,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.3,
                zorder=3,
            )

        bars = ax.bar(
            range(3),
            means,
            color=[TREATMENT_COLORS[t] for t in treatments],
            edgecolor="black",
            linewidth=0.5,
            width=0.65,
            zorder=2,
        )
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    m + (0.02 if m >= 0 else -0.06),
                    f"{m:+.3f}",
                    ha="center",
                    va="bottom" if m >= 0 else "top",
                    fontsize=6,
                    color="0.35",
                )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Control", "Hub", "Broker"], fontsize=7)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        _despine(ax)

    axes[0, 0].set_ylabel("Pre→post Δ")
    fig.suptitle("Treatment-arm deltas (primary outcomes)", fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, outdir, "fig_id2_treatment_deltas")


def fig_id3_noise_floor(
    survey_dir: Path, noise_waves: list[str], outdir: Path
) -> list[str]:
    """Measurement noise at frozen step 2100 (K independent surveys)."""
    rows = []
    for instrument, truth, metric, label in PRIMARY_OUTCOMES:
        if instrument.startswith("shocked_agent_rank_"):
            continue
        vals = _noise_rep_values(survey_dir, instrument, truth, metric, noise_waves)
        if not vals:
            continue
        rows.append(dict(label=label, values=vals, pre=None))

    # Add shocked-agent noise from perception surveys
    for instrument, truth, _metric, label in PRIMARY_OUTCOMES:
        if not instrument.startswith("shocked_agent_rank_"):
            continue
        vals = _noise_rep_values(survey_dir, instrument, truth, "", noise_waves)
        if vals:
            rows.append(dict(label=label, values=vals, pre=None))

    if not rows:
        return []

    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.55 * len(rows) + 1.0)))
    y_positions = list(range(len(rows)))

    for yi, row in enumerate(rows):
        vals = row["values"]
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        ax.hlines(yi, mean_v - 2 * std_v, mean_v + 2 * std_v, color="0.75", linewidth=5, zorder=1)
        ax.scatter(
            range(len(vals)),
            [yi] * len(vals),
            c=[0.4 + 0.1 * i for i in range(len(vals))],
            s=28,
            zorder=2,
            edgecolors="black",
            linewidths=0.3,
        )
        ax.plot(mean_v, yi, "k|", markersize=12, markeredgewidth=2, zorder=3)
        ax.text(
            len(vals) + 0.15,
            yi,
            f"μ={mean_v:.3f}, σ={std_v:.3f}",
            va="center",
            fontsize=7,
            color="0.4",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Noise replicate index")
    ax.set_title(
        f"Measurement noise floor (K={len(noise_waves)} surveys @ step {SHOCK_STEP})",
        loc="left",
    )
    _despine(ax)
    fig.tight_layout()
    return _save(fig, outdir, "fig_id3_noise_floor")


def fig_id4_network_structure(storage: Path, outdir: Path) -> list[str]:
    """Network density over time by treatment arm (replicate mean ± range)."""
    frames = []
    for treatment, _rep, code in FORKS:
        path = storage / code / "survey" / "network_structure_over_time.csv"
        df = _load_csv(path)
        if df.empty:
            continue
        df = df.groupby("step", as_index=False).last()
        df["treatment"] = treatment
        df["fork"] = code
        frames.append(df)

    if not frames:
        return []

    all_df = pd.concat(frames, ignore_index=True)
    panels = [
        ("density_cumulative", "Cumulative density"),
        ("density_recent", "Recent-window density"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, (col, ylabel) in zip(axes, panels):
        if col not in all_df.columns:
            continue
        for treatment in ("control", "hub", "broker"):
            sub = all_df[all_df["treatment"] == treatment]
            if sub.empty:
                continue
            agg = sub.groupby("step")[col].agg(["mean", "min", "max"])
            steps = agg.index.to_numpy()
            ax.plot(
                steps,
                agg["mean"],
                color=TREATMENT_COLORS[treatment],
                linewidth=1.8,
                label=treatment.capitalize(),
            )
            ax.fill_between(
                steps,
                agg["min"],
                agg["max"],
                color=TREATMENT_COLORS[treatment],
                alpha=0.15,
            )
        ax.axvline(SHOCK_STEP, color="0.55", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Simulation step")
        ax.set_ylabel(ylabel)
        _despine(ax)

    axes[0].set_title("Network saturation by treatment arm", loc="left")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].text(
        SHOCK_STEP,
        1.02,
        "shock",
        transform=axes[1].get_xaxis_transform(),
        fontsize=6,
        color="0.5",
        ha="center",
    )
    fig.tight_layout()
    return _save(fig, outdir, "fig_id4_network_structure")


def fig_id5_prepost_primary(
    delta_df: pd.DataFrame, outdir: Path
) -> list[str]:
    """Pre vs post values for primary outcomes (control vs treatments)."""
    panels = []
    for instrument, truth, _metric, label in PRIMARY_OUTCOMES:
        sub = _delta_subset(delta_df, instrument, truth)
        if sub.empty:
            continue
        panels.append((label, sub))

    if not panels:
        return []

    n_p = len(panels)
    fig, axes = plt.subplots(1, n_p, figsize=(2.0 * n_p, 3.4), squeeze=False)

    for ax, (title, sub) in zip(axes[0], panels):
        for trt in ("control", "hub", "broker"):
            trt_sub = sub[sub["treatment"] == trt]
            if trt_sub.empty:
                continue
            pre_mean = trt_sub["pre"].astype(float).mean()
            post_mean = trt_sub["post"].astype(float).mean()
            color = TREATMENT_COLORS[trt]
            ax.plot(
                [0, 1],
                [pre_mean, post_mean],
                color=color,
                linewidth=1.5,
                marker="o",
                markersize=5,
                label=trt.capitalize(),
            )
            for _, row in trt_sub.iterrows():
                ax.plot(
                    [0, 1],
                    [float(row["pre"]), float(row["post"])],
                    color=color,
                    alpha=0.25,
                    linewidth=0.8,
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"], fontsize=7)
        ax.set_title(title, fontsize=9, loc="left")
        _despine(ax)

    axes[0, 0].set_ylabel("Metric value")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Pre/post trajectories (replicate thin lines)", fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, outdir, "fig_id5_prepost_primary")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate NCN identification experiment figures.")
    ap.add_argument(
        "--survey-dir",
        default=None,
        help="Baseline survey directory (default: storage/ncn_baseline_n25_full/survey)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <survey-dir>/figures_id/)",
    )
    args = ap.parse_args()

    _apply_style()

    survey_dir = (
        Path(args.survey_dir)
        if args.survey_dir
        else (FS_STORAGE / DEFAULT_BASELINE / "survey").resolve()
    )
    outdir = Path(args.output_dir) if args.output_dir else survey_dir / "figures_id"
    storage = survey_dir.parent.parent  # frontend_server/storage

    did_path = survey_dir / "ncn_id_did_summary.csv"
    delta_path = survey_dir / "ncn_id_replicate_deltas.csv"
    manifest_path = survey_dir / "ncn_id_manifest.json"

    for p in (did_path, delta_path):
        if not p.exists():
            sys.exit(f"[error] missing required file: {p}")

    did_df = _load_csv(did_path)
    delta_df = _load_csv(delta_path)

    noise_waves: list[str] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        noise_waves = sorted(
            w for w in manifest.get("noise_floor", {}).get("completed", [])
            if str(w).startswith("noise_r")
        )

    print(f"Survey dir: {survey_dir}")
    print(f"Output   → {outdir}\n")

    generators = [
        ("Fig ID1 – DiD forest", lambda: fig_id1_did_forest(did_df, outdir)),
        ("Fig ID2 – Treatment deltas", lambda: fig_id2_treatment_deltas(delta_df, outdir)),
        ("Fig ID3 – Noise floor", lambda: fig_id3_noise_floor(survey_dir, noise_waves, outdir)),
        ("Fig ID4 – Network structure", lambda: fig_id4_network_structure(storage, outdir)),
        ("Fig ID5 – Pre/post primary", lambda: fig_id5_prepost_primary(delta_df, outdir)),
    ]

    all_paths: list[str] = []
    for name, fn in generators:
        print(f"  {name} ...", end=" ", flush=True)
        try:
            paths = fn()
        except Exception:
            print("ERROR")
            import traceback
            traceback.print_exc()
            continue
        if paths:
            print(f"→ {len(paths)} file(s)")
            all_paths.extend(paths)
        else:
            print("(skipped)")

    print(f"\n{'─' * 52}")
    print(f"{len(all_paths)} file(s) written to {outdir}:")
    for p in all_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
