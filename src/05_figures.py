#!/usr/bin/env python3
"""
plot_metric_groups_by_stage_hist_supervised.py
==============================================
Histogram + KDE ridge plots comparing CNCF (actual stage labels) vs CURIOSS
(stages assigned by supervised centroid method, Case 1: unique_contributors).

CURIOSS filtered to 32,133 repos with ≥1 contributor in last 5 years.
dependency_count dropped (not available in CURIOSS).
Outputs go to metric_plots_by_stage_supervised/ — originals untouched.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

BASE        = Path(__file__).parent.parent
TEMP_DIR    = Path(__file__).parent
CURIOSS_CSV = TEMP_DIR / "curioss_metrics.csv"
CNCF_CSV    = BASE     / "cncf_sustainability_metrics.csv"
ASSIGN_CSV  = TEMP_DIR / "curioss_supervised_clusters.csv"
OUT_DIR     = TEMP_DIR / "metric_plots_by_stage_supervised_v7"
OUT_DIR.mkdir(exist_ok=True)

CNCF_IQR_HI = "#1B6CA8"
CUR_IQR_HI  = "#A83010"
CNCF_BAR    = "#A8CCEA"
CUR_BAR     = "#F5C89A"

STAGES = [
    ("Sandbox",    "sandbox",    "Sandbox"),
    ("Incubating", "incubating", "Incubating"),
    ("Graduated",  "graduated",  "Graduated"),
]


def load_data():
    curioss    = pd.read_csv(CURIOSS_CSV)
    cncf       = pd.read_csv(CNCF_CSV)
    cncf       = cncf[cncf["stage"].isin(["sandbox","incubating","graduated"])].copy()
    assignments = pd.read_csv(ASSIGN_CSV)

    curioss["html_url"] = "https://github.com/" + curioss["project"].astype(str)
    # Use Case 1 (unique_contributors) for stage assignment; keep only assigned rows
    curioss = curioss.merge(
        assignments[["html_url","assigned_1feat"]].rename(columns={"assigned_1feat":"stage_name"}),
        on="html_url", how="inner"   # inner = only the 32,133 filtered repos
    )
    print(f"CURIOSS rows (filtered + assigned): {len(curioss):,}")
    print(f"CNCF rows: {len(cncf):,}")
    for sl, cs, us in STAGES:
        print(f"  {sl:12s}: CNCF={( cncf['stage']==cs).sum():4d}  CURIOSS={( curioss['stage_name']==us).sum():6,}")
    return curioss, cncf


def compute_kde(values, x_grid, min_bw=0.20, exponent=-0.20):
    if len(values) < 5:
        return np.zeros_like(x_grid)
    std = np.std(values)
    if std < 1e-8:
        values = values + np.random.default_rng(0).normal(0, 0.05, len(values))
        std = np.std(values)
    scott = std * len(values) ** exponent
    bw    = max(scott, min_bw)
    kde   = gaussian_kde(values, bw_method=bw/std if std > 0 else min_bw)
    return kde(x_grid)


def draw_ridge_hist(ax, cncf_vals, curioss_vals, log_scale, x_min, x_max,
                    allow_negative=False, n_bins=35, show_median=False):
    bins    = np.linspace(x_min, x_max, n_bins + 1)
    widths  = np.diff(bins)
    centers = (bins[:-1] + bins[1:]) / 2
    x_grid  = np.linspace(x_min, x_max, 500)

    for vals, bar_color, line_color, zorder in [
        (curioss_vals, CUR_BAR,  CUR_IQR_HI,  2),
        (cncf_vals,    CNCF_BAR, CNCF_IQR_HI, 3),
    ]:
        v = np.array(vals, dtype=float)
        v = v[np.isfinite(v)] if allow_negative else v[np.isfinite(v) & (v >= 0)]
        if len(v) < 5:
            continue
        tv = np.log1p(v) if log_scale else v
        counts, _ = np.histogram(tv, bins=bins)
        props = counts / len(tv)
        ax.bar(centers, props, width=widths*0.88,
               color=bar_color, alpha=0.75, zorder=zorder,
               linewidth=0.2, edgecolor="white")
        density = compute_kde(tv, x_grid)
        hist_area = np.sum(props * widths)
        kde_area  = np.trapz(density, x_grid)
        if kde_area > 0:
            density = density * (hist_area / kde_area)
            ax.fill_between(x_grid, density, color=line_color,
                            alpha=0.10, linewidth=0, zorder=zorder+1)
            ax.plot(x_grid, density, color=line_color,
                    linewidth=1.6, alpha=0.90, zorder=zorder+2)
        if show_median:
            med = np.median(tv)
            if x_min <= med <= x_max:
                ax.vlines(med, 0, props.max(), color=line_color,
                          linewidth=1.5, linestyle=":", alpha=0.9, zorder=zorder+1)

    ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)
    ax.axhline(0, color="#cccccc", linewidth=0.7)


def _x_range(vals_list, log_scale, lo_pct=1.0, hi_pct=97.0, allow_negative=False):
    mins, maxs = [], []
    for v in vals_list:
        v = np.array(v, dtype=float)
        v = v[np.isfinite(v)] if allow_negative else v[np.isfinite(v) & (v >= 0)]
        if len(v) < 5:
            continue
        tv = np.log1p(v) if log_scale else v
        tv = tv[np.isfinite(tv)]
        if len(tv) < 5:
            continue
        mins.append(float(np.percentile(tv, lo_pct)))
        maxs.append(float(np.percentile(tv, hi_pct)))
    return (min(mins), max(maxs)) if mins else (0.0, 1.0)


def _log_ticks(x_min, x_max):
    candidates = [0,1,2,3,5,10,20,50,100,200,500,
                  1000,2000,5000,10000,50000,100000,
                  365,730,1825,3650,36500]
    ticks = sorted(set(candidates))
    ticks = [t for t in ticks
             if np.log1p(t) >= x_min-0.05 and np.log1p(t) <= x_max+0.05]
    if len(ticks) > 7:
        step = max(1, len(ticks)//6)
        ticks = ticks[::step]
    return ticks


def _add_legend(fig):
    fig.legend(handles=[
        mpatches.Patch(facecolor=CNCF_BAR, edgecolor=CNCF_IQR_HI, label="CNCF"),
        mpatches.Patch(facecolor=CUR_BAR,  edgecolor=CUR_IQR_HI,
                       label="CURIOSS (supervised — unique contributors)"),
    ], loc="upper right", frameon=True, fontsize=10, bbox_to_anchor=(0.99, 0.99))


def _style(ax, col_j, row_i, stage_label, col_label,
           log_scale, x_min, x_max, nc, nr,
           tick_fs=5.5, count_fs=6.0, label_fs=7.5, stage_fs=11):
    if log_scale:
        ticks = _log_ticks(x_min, x_max)
        ax.set_xticks([np.log1p(t) for t in ticks])
        ax.set_xticklabels([str(t) for t in ticks], fontsize=tick_fs, rotation=45, ha="right")
    else:
        ax.tick_params(axis="x", labelsize=tick_fs)
    ax.set_xlim(x_min, x_max)
    ax.tick_params(axis="x", pad=4)
    ax.set_title(f"CNCF n={nc}  CUR n={nr}", fontsize=count_fs, pad=3, color="#444444")
    if col_j == 0:
        ax.set_ylabel(stage_label, fontsize=stage_fs, fontweight="bold",
                      rotation=0, labelpad=65, va="center")
    if row_i == 2:
        ax.set_xlabel(col_label, fontsize=label_fs, labelpad=10)


def stage_ridge_hist_figure(metric_defs, curioss, cncf, out_path, title="",
                             figsize=None, left_margin=0.08,
                             hspace=0.85, wspace=0.35,
                             tick_fs=5.5, count_fs=6.0, label_fs=7.5, stage_fs=11,
                             show_median=False):
    n_m = len(metric_defs)
    if figsize is None:
        figsize = (max(12, n_m*3.4), 9.5)
    fig, axes = plt.subplots(3, n_m, figsize=figsize,
                              gridspec_kw={"hspace":hspace,"wspace":wspace})
    if n_m == 1:
        axes = axes[:, np.newaxis]

    col_ranges = []
    for row in metric_defs:
        c_col, cncf_col, log_scale = row[1], row[2], row[3]
        lo_pct    = row[4] if len(row) > 4 else 1.0
        hi_pct    = row[5] if len(row) > 5 else 97.0
        allow_neg = row[6] if len(row) > 6 else False
        c_all    = curioss[c_col].dropna().values  if c_col    in curioss.columns else np.array([])
        cncf_all = cncf[cncf_col].dropna().values  if cncf_col in cncf.columns    else np.array([])
        col_ranges.append(_x_range([c_all, cncf_all], log_scale,
                                    lo_pct=lo_pct, hi_pct=hi_pct, allow_negative=allow_neg))

    for row_i, (stage_label, cncf_stage, curioss_stage) in enumerate(STAGES):
        cncf_s    = cncf[cncf["stage"] == cncf_stage]
        curioss_s = curioss[curioss["stage_name"] == curioss_stage] if "stage_name" in curioss.columns else pd.DataFrame()
        for col_j, row in enumerate(metric_defs):
            col_label, c_col, cncf_col, log_scale = row[0], row[1], row[2], row[3]
            allow_neg = row[6] if len(row) > 6 else False
            x_min, x_max = col_ranges[col_j]
            ax = axes[row_i, col_j]
            c_v    = curioss_s[c_col].dropna().values if c_col    in curioss_s.columns and len(curioss_s) > 0 else np.array([])
            cncf_v = cncf_s[cncf_col].dropna().values if cncf_col in cncf_s.columns    and len(cncf_s)    > 0 else np.array([])
            draw_ridge_hist(ax, cncf_v, c_v, log_scale, x_min, x_max,
                            allow_negative=allow_neg, show_median=show_median)
            nc = int(np.isfinite(np.array(cncf_v, dtype=float)).sum())
            nr = int(np.isfinite(np.array(c_v,    dtype=float)).sum())
            _style(ax, col_j, row_i, stage_label, col_label, log_scale, x_min, x_max, nc, nr,
                   tick_fs=tick_fs, count_fs=count_fs, label_fs=label_fs, stage_fs=stage_fs)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)
    _add_legend(fig)
    fig.subplots_adjust(left=left_margin, right=0.97, bottom=0.10, top=0.94)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved → {out_path}")


def stage_general_hist_figure(curioss, cncf, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5),
                              gridspec_kw={"hspace":0.85,"wspace":0.40})
    ridge_cols = [
        ("Project age (days)",  "project_age_days",  "project_age_days",  True),
        ("Community interest",  "community_interest", "community_interest", True),
    ]
    ranges = [_x_range([curioss[c].dropna().values if c in curioss.columns else np.array([]),
                         cncf[cc].dropna().values  if cc in cncf.columns   else np.array([])], ls)
              for _, c, cc, ls in ridge_cols]
    bar_h = 0.32
    for row_i, (stage_label, cncf_stage, curioss_stage) in enumerate(STAGES):
        cncf_s    = cncf[cncf["stage"] == cncf_stage]
        curioss_s = curioss[curioss["stage_name"] == curioss_stage] if "stage_name" in curioss.columns else pd.DataFrame()
        for col_j, (col_label, c_col, cncf_col, log_scale) in enumerate(ridge_cols):
            ax = axes[row_i, col_j]
            x_min, x_max = ranges[col_j]
            c_v    = curioss_s[c_col].dropna().values if c_col    in curioss_s.columns and len(curioss_s) > 0 else np.array([])
            cncf_v = cncf_s[cncf_col].dropna().values if cncf_col in cncf_s.columns    and len(cncf_s)    > 0 else np.array([])
            draw_ridge_hist(ax, cncf_v, c_v, log_scale, x_min, x_max)
            nc = int(np.isfinite(np.array(cncf_v, dtype=float)).sum())
            nr = int(np.isfinite(np.array(c_v,    dtype=float)).sum())
            _style(ax, col_j, row_i, stage_label, col_label, log_scale, x_min, x_max, nc, nr)
        ax = axes[row_i, 2]
        c_v = curioss_s["financial_support"].dropna() if "financial_support" in curioss_s.columns and len(curioss_s) > 0 else pd.Series([], dtype=float)
        c_val = 100 * c_v.astype(bool).mean() if len(c_v) > 0 else 0
        ax.barh([0.5-bar_h/2], [c_val], bar_h, color=CUR_IQR_HI, alpha=0.85)
        ax.text(c_val+0.8, 0.5-bar_h/2, f"{c_val:.0f}%", va="center", ha="left", fontsize=6.5, color=CUR_IQR_HI)
        ax.set_title(f"CNCF: N/A  CUR n={len(c_v)}", fontsize=6.0, pad=3, color="#444444")
        ax.set_xlim(0,115); ax.set_ylim(0,1.0); ax.set_yticks([])
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=5.5)
        if row_i == 2:
            ax.set_xlabel("Financial support\n(% repos, CURIOSS only)", fontsize=7.5)
    fig.suptitle("General — by Maturity Stage | CNCF (blue) vs. CURIOSS supervised (orange)\n"
                 "[CURIOSS: supervised centroid, unique_contributors; n=32,133]",
                 fontsize=11, fontweight="bold", y=0.99)
    _add_legend(fig)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.93)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    curioss, cncf = load_data()
    SUB = "[CURIOSS: supervised centroid, unique_contributors, n=32,133]"

    stage_ridge_hist_figure([
        ("Unique\ncontributors",             "unique_contributors",           "unique_contributors",           True),
        ("Active contributors\n(W90)",       "contributor_count_90d",         "contributor_count_90d",         True),
        ("Org. diversity\n(W90)",            "organizational_diversity_90d",  "organizational_diversity_90d",  True),
        ("Contributor retention\n(W90, %)",  "contributor_retention_90d_pct", "contributor_retention_90d_pct", False),
        ("Contributor growth\n(W90, %)",     "contributor_growth_90d_pct",    "contributor_growth_90d_pct",    False),
        ("Bus factor",                       "bus_factor",                    "bus_factor",                    True),
    ], curioss, cncf,
    out_path=OUT_DIR/"community_hist_supervised.png",
    title=f"Community — by Maturity Stage | CNCF (blue) vs. CURIOSS (orange)\n{SUB}",
    figsize=(21, 9.5))

    stage_ridge_hist_figure([
        ("Social activity\n(W90)",            "social_activity_90d",              "social_activity_90d",              True,  1.0, 97.0),
        ("Commit frequency\n(W90)",           "commit_frequency_90d",             "commit_frequency_90d",             True,  1.0, 97.0),
        ("Commit growth\n(W90, %)",           "commit_growth_90d_pct",            "commit_growth_90d_pct",            False, 2.0, 95.0, True),
        ("Issue closed\nratio (W90)",         "issue_closed_ratio_90d",           "issue_closed_ratio_90d",           False, 1.0, 99.0),
        ("Issue response\ntime (W90, days)",  "issue_response_time_90d",          "issue_response_time_90d",          True,  1.0, 97.0),
        ("PR response\ntime (W90, days)",     "pr_response_time_90d",             "pr_response_time_90d",             True,  1.0, 97.0),
        ("PR merge\ntime (W90, days)",        "pr_merge_time_90d",                "pr_merge_time_90d",                True,  1.0, 97.0),
        ("Code review\nratio (W90)",          "code_review_practices_90d",        "code_review_practices_90d",        False, 1.0, 99.0),
        ("Release\nfrequency (W90)",          "release_frequency_90d",            "release_frequency_90d",            True,  1.0, 97.0),
        ("Contributors\ndev. activity (W90)", "contributors_dev_activity_90d",    "contributors_dev_activity_90d",    True,  1.0, 97.0),
        ("Maintainers\ndev. activity (W90)",  "maintainers_dev_activity_90d",     "maintainers_dev_activity_90d",     True,  1.0, 97.0),
        ("Maintainer\nrecency (days)",        "maintainer_activity_recency_days", "maintainer_activity_recency_days", True,  1.0, 97.0),
    ], curioss, cncf,
    out_path=OUT_DIR/"activity_hist_supervised.png",
    title=f"Activity — by Maturity Stage | CNCF (blue) vs. CURIOSS (orange)\n{SUB}",
    figsize=(34, 9.5))

    stage_general_hist_figure(curioss, cncf, OUT_DIR/"general_hist_supervised.png")

    stage_ridge_hist_figure([
        ("Contributor growth\n(W90, %)", "contributor_growth_90d_pct", "contributor_growth_90d_pct", False, 1.0, 99.0, True),
        ("Bus factor",                   "bus_factor",                 "bus_factor",                 True),
        ("Community interest",           "community_interest",          "community_interest",          True),
    ], curioss, cncf,
    out_path=OUT_DIR/"community_trio_hist_supervised.png",
    title="", figsize=(12, 9.5),
    hspace=0.50, wspace=0.18,
    tick_fs=9, count_fs=7.5, label_fs=14, stage_fs=14, show_median=True)

    stage_ridge_hist_figure([
        ("Social activity\n(W90)",           "social_activity_90d",     "social_activity_90d",     True,  1.0, 97.0),
        ("Commit growth\n(W90, %)",          "commit_growth_90d_pct",   "commit_growth_90d_pct",   False, 2.0, 95.0, True),
        ("Issue response\ntime (W90, days)", "issue_response_time_90d", "issue_response_time_90d", True,  1.0, 97.0),
        ("PR response\ntime (W90, days)",    "pr_response_time_90d",    "pr_response_time_90d",    True,  1.0, 97.0),
    ], curioss, cncf,
    out_path=OUT_DIR/"activity_quartet_hist_supervised.png",
    title="", figsize=(15, 9.5),
    hspace=0.50, wspace=0.18,
    tick_fs=9, count_fs=7.5, label_fs=14, stage_fs=14, show_median=True)


if __name__ == "__main__":
    main()
