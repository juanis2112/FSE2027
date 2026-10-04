#!/usr/bin/env python3
"""
classify_maturity_cncf.py
==========================
Classifies CURIOSS repositories into maturity stages using a
Nearest Centroid classifier trained on 75 CNCF projects (25 per stage).

Pipeline:
  1. Fit a StandardScaler on the CNCF training data (log1p-transformed)
  2. Apply sqrt(Lumbard-weight) scaling
  3. Compute per-stage centroids in that space
  4. Assign each CURIOSS repo to its nearest centroid (Euclidean distance)

Also prints the k=3 unsupervised cluster table for comparison.

Outputs:
  curioss_clusters_cncf.csv         — per-repo stage assignments
  cluster_plots/cncf_k3_scatter.png — scatter coloured by CNCF-based stage
  cluster_plots/cncf_k3_violin.png  — violin plots per feature by stage
  cluster_plots/cncf_centroids.png  — CNCF stage profiles (training data)
"""

import sqlite3, csv, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TEMP_DIR    = Path(__file__).parent
CNCF_DB     = TEMP_DIR.parent / "cncf" / "cncf_raw.db"
CNCF_CSV    = TEMP_DIR.parent / "cncf_sustainability_metrics.csv"
CURIOSS_DB  = TEMP_DIR / "curioss_raw.db"
METRICS_CSV = TEMP_DIR / "curioss_metrics.csv"
OUT_DIR     = TEMP_DIR / "cluster_plots"
OUT_DIR.mkdir(exist_ok=True)

FEATURES = ["unique_contributors", "stargazers_count", "pr_avg_commits"]

# Lumbard feature importances → relative weights among our 3 features
# unique_contributors replaces unique_contributors
WEIGHTS_RAW = {"unique_contributors": 0.561794,
               "stargazers_count":    0.073238,
               "pr_avg_commits":      0.064346}
_total = sum(WEIGHTS_RAW.values())
WEIGHTS_SQRT = np.array([(WEIGHTS_RAW[f] / _total) ** 0.5 for f in FEATURES])

STAGES   = ["sandbox", "incubating", "graduated"]
COLORS   = {"sandbox": "#4C72B0", "incubating": "#DD8452", "graduated": "#55A868"}
STAGE_CAP = {"sandbox": "Sandbox", "incubating": "Incubating", "graduated": "Graduated"}


# ── Data loaders ──────────────────────────────────────────────────────────────

BROAD_SQL = """
    SELECT repo_id, author_login   AS contributor FROM commits       WHERE author_login   IS NOT NULL
    UNION ALL
    SELECT repo_id, author_login   AS contributor FROM pull_requests WHERE author_login   IS NOT NULL
    UNION ALL
    SELECT repo_id, author_login   AS contributor FROM issues        WHERE author_login   IS NOT NULL
    UNION ALL
    SELECT repo_id, reviewer_login AS contributor FROM pr_reviews    WHERE reviewer_login IS NOT NULL
"""


def load_cncf() -> pd.DataFrame:
    conn = sqlite3.connect(CNCF_DB)
    df = pd.read_sql_query(f"""
        SELECT r.html_url, r.cncf_stage AS stage, r.stargazers_count,
               COALESCE(AVG(p.commits_count), 0)    AS pr_avg_commits,
               COUNT(DISTINCT src.contributor)       AS unique_contributors
        FROM repositories r
        LEFT JOIN pull_requests p ON p.repo_id = r.id
        LEFT JOIN ({BROAD_SQL}) src ON src.repo_id = r.id
        WHERE r.cncf_stage IN ('sandbox','incubating','graduated')
        GROUP BY r.id
    """, conn)
    conn.close()
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0).clip(lower=0)
    log.info("CNCF rows: %d", len(df))
    return df


def load_curioss() -> pd.DataFrame:
    conn = sqlite3.connect(CURIOSS_DB)
    df = pd.read_sql_query(f"""
        SELECT r.html_url, r.stargazers_count,
               COALESCE(AVG(p.commits_count), 0)    AS pr_avg_commits,
               COUNT(DISTINCT src.contributor)       AS unique_contributors
        FROM repositories r
        LEFT JOIN pull_requests p ON p.repo_id = r.id
        LEFT JOIN ({BROAD_SQL}) src ON src.repo_id = r.id
        GROUP BY r.id
    """, conn)
    conn.close()
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0).clip(lower=0)
    log.info("CURIOSS rows: %d", len(df))
    return df


# ── Feature transform ─────────────────────────────────────────────────────────

def make_X(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = False):
    X = np.log1p(df[FEATURES].values.astype(float))
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X * WEIGHTS_SQRT, scaler
    else:
        X = scaler.transform(X)
        return X * WEIGHTS_SQRT


# ── Nearest centroid classifier ───────────────────────────────────────────────

def fit_centroids(cncf: pd.DataFrame, scaler: StandardScaler) -> dict:
    centroids = {}
    for stage in STAGES:
        sub = cncf[cncf["stage"] == stage]
        X   = make_X(sub, scaler)
        centroids[stage] = X.mean(axis=0)
        log.info("  Centroid [%s]: %s", stage, np.round(centroids[stage], 3))
    return centroids


def nearest_centroid(X: np.ndarray, centroids: dict) -> np.ndarray:
    dists = np.stack([np.linalg.norm(X - c, axis=1)
                      for c in [centroids[s] for s in STAGES]], axis=1)
    idx   = dists.argmin(axis=1)
    return np.array(STAGES)[idx]


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cncf_profiles(cncf: pd.DataFrame):
    """Bar chart of median feature values per CNCF stage (training data)."""
    feat_labels = ["Unique\nContributors", "Stars", "Avg Commits/PR"]
    x = np.arange(3)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, stage in enumerate(STAGES):
        sub     = cncf[cncf["stage"] == stage]
        medians = [sub["unique_contributors"].median(),
                   sub["stargazers_count"].median(),
                   sub["pr_avg_commits"].median()]
        # Normalise to 0-1 per feature for visual comparison
        bars = ax.bar(x + i * width, medians, width,
                      label=STAGE_CAP[stage], color=COLORS[stage], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(feat_labels, fontsize=11)
    ax.set_ylabel("Median (raw value)", fontsize=11)
    ax.set_title("CNCF Stage Profiles — Training Data Medians\n"
                 "(used as reference centroids for CURIOSS classification)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_yscale("symlog")   # log scale handles wide range stars vs commits
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cncf_centroids.png", dpi=150)
    plt.close(fig)
    log.info("Saved cncf_centroids.png")


def plot_scatter(df: pd.DataFrame, stage_col: str, title_suffix: str, fname: str):
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.log1p(df["unique_contributors"])
    y = np.log1p(df["stargazers_count"])
    for stage in STAGES:
        mask = df[stage_col] == stage
        ax.scatter(x[mask], y[mask], c=COLORS[stage],
                   label=f"{STAGE_CAP[stage]} (n={mask.sum():,})",
                   alpha=0.35, s=10, linewidths=0)
    ax.set_xlabel("log(1 + new contributors, 90d)", fontsize=12)
    ax.set_ylabel("log(1 + stars)", fontsize=12)
    ax.set_title(f"CURIOSS Maturity Classification — {title_suffix}", fontsize=13)
    ax.legend(title="Stage", markerscale=3, fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fname)


def plot_violin(df: pd.DataFrame, stage_col: str, title_suffix: str, fname: str):
    feat_labels = {"unique_contributors": "Unique Contributors",
                   "stargazers_count":     "Stars",
                   "pr_avg_commits":       "Avg Commits / PR"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"Feature Distributions by Stage — {title_suffix}", fontsize=14)
    for ax, feat in zip(axes, FEATURES):
        data = [np.log1p(df.loc[df[stage_col] == s, feat].values) for s in STAGES]
        parts = ax.violinplot(data, positions=range(3), showmedians=True, showextrema=False)
        for pc, stage in zip(parts["bodies"], STAGES):
            pc.set_facecolor(COLORS[stage])
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([STAGE_CAP[s] for s in STAGES], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("log(1 + value)", fontsize=9)
        ax.set_title(feat_labels[feat], fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    log.info("Saved %s", fname)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_table(df: pd.DataFrame, stage_col: str, label: str):
    total = len(df)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Stage':<14} {'n':>7}  {'%':>6}  {'New Contrib (med)':>17}  {'Stars (med)':>11}  {'PR Commits (med)':>16}")
    print(f"  {'-'*14} {'-'*7}  {'-'*6}  {'-'*17}  {'-'*11}  {'-'*16}")
    for stage in STAGES:
        sub = df[df[stage_col] == stage]
        print(f"  {STAGE_CAP[stage]:<14} {len(sub):>7,}  {100*len(sub)/total:>5.1f}%"
              f"  {sub['unique_contributors'].median():>17.0f}"
              f"  {sub['stargazers_count'].median():>11.0f}"
              f"  {sub['pr_avg_commits'].median():>16.2f}")
    print(f"  {'TOTAL':<14} {total:>7,}  {'100.0%':>6}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cncf    = load_cncf()
    curioss = load_curioss()

    # ── Fit scaler on CNCF data, build centroids ──
    log.info("Fitting scaler on CNCF training data…")
    X_cncf, scaler = make_X(cncf, fit=True)
    log.info("Computing CNCF stage centroids…")
    centroids = fit_centroids(cncf, scaler)

    # ── Classify CURIOSS repos ────────────────────
    log.info("Classifying %d CURIOSS repos…", len(curioss))
    X_curioss = make_X(curioss, scaler)
    curioss["stage_cncf"] = nearest_centroid(X_curioss, centroids)

    # ── Print summary tables ──────────────────────
    print_table(curioss, "stage_cncf", "CNCF Nearest-Centroid Classification")

    # ── Print CNCF training profiles ──────────────
    print(f"\n{'='*70}")
    print("  CNCF Training Data — Stage Profiles (medians, raw values)")
    print(f"{'='*70}")
    for stage in STAGES:
        sub = cncf[cncf["stage"] == stage]
        print(f"  {STAGE_CAP[stage]:<12} (n={len(sub):2d})"
              f"  new_contrib={sub['unique_contributors'].median():5.0f}"
              f"  stars={sub['stargazers_count'].median():7.0f}"
              f"  pr_commits={sub['pr_avg_commits'].median():5.2f}")

    # ── Plots ──────────────────────────────────────
    plot_cncf_profiles(cncf)
    plot_scatter(curioss, "stage_cncf", "CNCF Nearest-Centroid", "cncf_k3_scatter.png")
    plot_violin(curioss,  "stage_cncf", "CNCF Nearest-Centroid", "cncf_k3_violin.png")

    # ── Save CSV ───────────────────────────────────
    out = TEMP_DIR / "curioss_clusters_cncf.csv"
    curioss[["html_url", "university", "unique_contributors",
             "stargazers_count", "pr_avg_commits", "stage_cncf"]].to_csv(out, index=False)
    log.info("Saved %s", out.name)


if __name__ == "__main__":
    main()
