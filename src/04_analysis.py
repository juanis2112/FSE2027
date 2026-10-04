#!/usr/bin/env python3
"""
compare_cncf_curioss.py
=======================
Comprehensive CNCF vs CURIOSS comparison plots for sustainability metrics,
organised by category. Produces one set of plots per category:
  - Ridge plots  (KDE density overlays, CNCF blue vs CURIOSS orange)
  - Violin plots (side-by-side per stage)
  - Box plots    (side-by-side per stage)
  - Binary bar charts (proportion-True per stage × dataset)

CURIOSS maturity stages
    Computed with k-means k=3 on log1p(unique_contributors), where
    unique_contributors = COUNT DISTINCT(login) across commits + pull_requests
    + issues + pr_reviews.  Clusters are labelled by ascending mean:
    Sandbox (lowest) → Incubating → Graduated (highest).

CNCF maturity stages
    Real cncf_stage labels from the repositories table
    (sandbox / incubating / graduated).

All metrics are recomputed fresh from each raw SQLite database using the same
SQL patterns as compute_curioss_metrics.py, so this script is self-contained
and fully reproducible without depending on any pre-computed CSV.
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
CURIOSS_DB  = BASE / "curioss_raw.db"
CNCF_DB     = BASE.parent / "cncf" / "cncf_raw.db"
OUT_DIR     = BASE / "cluster_plots" / "compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Date windows (match compute_curioss_metrics.py) ───────────────────────────
TODAY = datetime(2026, 8, 11, tzinfo=timezone.utc)
D30   = TODAY - timedelta(days=30)
D90   = TODAY - timedelta(days=90)
D180  = TODAY - timedelta(days=180)
D365  = TODAY - timedelta(days=365)
def iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# ── Visual constants ───────────────────────────────────────────────────────────
STAGES       = ["Sandbox", "Incubating", "Graduated"]
CNCF_COL     = "#1B6CA8"
CUR_COL      = "#A83010"
CNCF_LIGHT   = "#C5DCF0"
CUR_LIGHT    = "#F5D5B0"
CNCF_MED     = "#5B9EC9"
CUR_MED      = "#E08040"
STAGE_Y_GAP  = 0.55   # vertical separation between ridge rows

PERSONAL = ("'gmail.com','yahoo.com','hotmail.com','outlook.com',"
            "'users.noreply.github.com','noreply.github.com'")

# ── Metric catalogue ───────────────────────────────────────────────────────────
# Each entry: (column_name, axis_label, use_log1p, x_max)
CONTINUOUS = {
    "Community": [
        ("unique_contributors",        "Unique contributors\n(all sources)",  True,  2000),
        ("contributor_count_90d",      "Active contributors\n(90 d)",         True,  300),
        ("new_contributors_90d",       "New contributors\n(90 d)",            True,  100),
        ("organizational_diversity_90d","Org diversity\n(domains, 90 d)",     True,   30),
        ("bus_factor",                 "Bus factor",                          False,  20),
    ],
    "Activity": [
        ("commit_frequency_90d",       "Commit frequency\n(commits/week)",    True,  200),
        ("release_frequency_6m",       "Release frequency\n(rel./month)",     True,    5),
        ("issue_closed_ratio_90d",     "Issue closed ratio\n(90 d)",          False,   1),
        ("code_review_practices_90d",  "Code-review rate\n(90 d)",            False,   1),
    ],
    "Response Times": [
        ("issue_response_time_90d",    "Issue response\n(days, 90 d)",        True,  365),
        ("pr_response_time_90d",       "PR response\n(days, 90 d)",           True,  365),
        ("pr_merge_time_90d",          "PR merge time\n(days, 90 d)",         True,  365),
    ],
}

BINARY = {
    "Documentation": [
        ("readme",         "README"),
        ("contributing",   "CONTRIBUTING"),
        ("code_of_conduct","Code of Conduct"),
        ("governance",     "Governance"),
        ("changelog",      "Changelog"),
    ],
    "Dev Practices & Security": [
        ("ci_cd",              "CI/CD"),
        ("test_presence",      "Tests"),
        ("branch_protection",  "Branch Protection"),
        ("security_policy",    "Security Policy"),
        ("dependency_updates", "Dep. Updates"),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _unique_contrib_query():
    return """
        SELECT repo_id, COUNT(DISTINCT contributor) AS unique_contributors
        FROM (
            SELECT repo_id, author_login   AS contributor FROM commits       WHERE author_login   IS NOT NULL
            UNION ALL
            SELECT repo_id, author_login   AS contributor FROM pull_requests WHERE author_login   IS NOT NULL
            UNION ALL
            SELECT repo_id, author_login   AS contributor FROM issues        WHERE author_login   IS NOT NULL
            UNION ALL
            SELECT repo_id, reviewer_login AS contributor FROM pr_reviews    WHERE reviewer_login IS NOT NULL
        )
        GROUP BY repo_id
    """


def compute_metrics(conn) -> pd.DataFrame:
    """
    Compute the full set of comparable sustainability metrics for a single DB.
    Returns a DataFrame indexed by repo_id with all metric columns, plus
    html_url, stage_raw (cncf_stage or NULL), and unique_contributors.
    """
    repos = pd.read_sql(
        "SELECT id AS repo_id, html_url, cncf_stage AS stage_raw, "
        "stargazers_count, forks_count, created_at, license_key FROM repositories",
        conn)

    # ── Repo files (wide) ──────────────────────────────────────────────────────
    rf = pd.read_sql("SELECT repo_id, file_key, exists_flag FROM repo_files", conn)
    rf_wide = rf.pivot_table(index="repo_id", columns="file_key",
                              values="exists_flag", fill_value=0).reset_index()

    # ── Repo security ─────────────────────────────────────────────────────────
    sec = pd.read_sql(
        "SELECT repo_id, requires_pr_review, has_branch_protection, "
        "has_signed_releases, has_dependabot, has_renovate, open_advisories_count "
        "FROM repo_security", conn)

    # ── Unique contributors (broad, all sources) ──────────────────────────────
    uc = pd.read_sql(_unique_contrib_query(), conn)

    # ── Commit metrics ────────────────────────────────────────────────────────
    commit_agg = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(*) FILTER (WHERE authored_at >= '{iso(D90)}') AS commits_90d,
               COUNT(*) FILTER (WHERE authored_at >= '{iso(D30)}') AS commits_30d,
               COUNT(*) FILTER (WHERE authored_at >= '{iso(D180)}') AS commits_last6,
               COUNT(*) FILTER (WHERE authored_at >= '{iso(D365)}'
                                AND authored_at < '{iso(D180)}') AS commits_prior6
        FROM commits WHERE authored_at IS NOT NULL
        GROUP BY repo_id
    """, conn)

    contrib_agg = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(DISTINCT CASE WHEN authored_at >= '{iso(D90)}' THEN author_login END)
                   AS contributor_count_90d
        FROM commits WHERE authored_at IS NOT NULL AND author_login IS NOT NULL
        GROUP BY repo_id
    """, conn)

    # new contributors 90d (seen in last 90d but not before)
    a90  = pd.read_sql(
        f"SELECT repo_id, author_login FROM commits "
        f"WHERE authored_at >= '{iso(D90)}' AND author_login IS NOT NULL "
        f"GROUP BY repo_id, author_login", conn)
    apre = pd.read_sql(
        f"SELECT repo_id, author_login FROM commits "
        f"WHERE authored_at < '{iso(D90)}' AND author_login IS NOT NULL "
        f"GROUP BY repo_id, author_login", conn)
    a90["in_before"] = a90.set_index(["repo_id","author_login"]).index.isin(
        apre.set_index(["repo_id","author_login"]).index)
    new_contrib = (a90[~a90["in_before"]]
                   .groupby("repo_id").size()
                   .reset_index(name="new_contributors_90d"))

    # organisational diversity 90d
    org_div = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(DISTINCT author_email_domain) AS organizational_diversity_90d
        FROM commits
        WHERE authored_at >= '{iso(D90)}'
          AND author_email_domain IS NOT NULL AND author_email_domain != ''
          AND author_email_domain NOT IN ({PERSONAL})
        GROUP BY repo_id
    """, conn)

    # ── Issue metrics ─────────────────────────────────────────────────────────
    issue_agg = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(*) FILTER (WHERE created_at >= '{iso(D30)}') AS issues_30d,
               COUNT(*) FILTER (WHERE closed_at >= '{iso(D90)}' AND state='closed')
                   AS closed_issues_90d,
               COUNT(*) FILTER (WHERE created_at >= '{iso(D90)}'
                                AND (closed_at IS NULL OR closed_at > '{iso(TODAY)}'))
                   AS open_issues_90d
        FROM issues WHERE created_at IS NOT NULL
        GROUP BY repo_id
    """, conn)

    issue_resp = pd.read_sql(f"""
        SELECT repo_id,
               ROUND(AVG(julianday(first_response_at) - julianday(created_at)), 1)
                   AS issue_response_time_90d
        FROM issues
        WHERE created_at >= '{iso(D90)}'
          AND first_response_at IS NOT NULL AND first_response_at >= created_at
        GROUP BY repo_id
    """, conn)

    # ── PR metrics ────────────────────────────────────────────────────────────
    pr_agg = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(*) FILTER (WHERE created_at >= '{iso(D30)}') AS prs_30d,
               COUNT(*) FILTER (WHERE merged_at >= '{iso(D90)}') AS total_merged_prs,
               COUNT(*) FILTER (WHERE merged_at >= '{iso(D90)}' AND review_count > 0)
                   AS reviewed_prs
        FROM pull_requests WHERE created_at IS NOT NULL
        GROUP BY repo_id
    """, conn)

    pr_resp = pd.read_sql(f"""
        SELECT repo_id,
               ROUND(AVG(julianday(first_review_at) - julianday(created_at)), 1)
                   AS pr_response_time_90d
        FROM pull_requests
        WHERE created_at >= '{iso(D90)}'
          AND first_review_at IS NOT NULL AND first_review_at >= created_at
        GROUP BY repo_id
    """, conn)

    pr_merge = pd.read_sql(f"""
        SELECT repo_id,
               ROUND(AVG(julianday(merged_at) - julianday(created_at)), 1)
                   AS pr_merge_time_90d
        FROM pull_requests
        WHERE merged_at >= '{iso(D90)}' AND merged_at IS NOT NULL
          AND merged_at >= created_at
        GROUP BY repo_id
    """, conn)

    # ── Release metrics ───────────────────────────────────────────────────────
    release_agg = pd.read_sql(f"""
        SELECT repo_id, COUNT(*) AS releases_6m
        FROM releases WHERE published_at >= '{iso(D180)}'
        GROUP BY repo_id
    """, conn)

    # ── Comments (social) ─────────────────────────────────────────────────────
    try:
        comment_agg = pd.read_sql(f"""
            SELECT repo_id, COUNT(*) AS comments_30d
            FROM issue_comments WHERE created_at >= '{iso(D30)}'
            GROUP BY repo_id
        """, conn)
    except Exception:
        comment_agg = pd.DataFrame(columns=["repo_id","comments_30d"])

    # ── Bus factor ────────────────────────────────────────────────────────────
    bf_raw = pd.read_sql("""
        SELECT repo_id, author_login, COUNT(*) AS cnt
        FROM commits WHERE author_login IS NOT NULL
        GROUP BY repo_id, author_login ORDER BY repo_id, cnt DESC
    """, conn)

    def calc_bus_factor(grp):
        total = grp["cnt"].sum()
        if total == 0: return np.nan
        running = 0
        for i, cnt in enumerate(grp["cnt"], 1):
            running += cnt
            if running / total > 0.5: return i
        return i

    bus_factor = (bf_raw.groupby("repo_id")
                  .apply(calc_bus_factor)
                  .reset_index(name="bus_factor"))

    # ── Merge ─────────────────────────────────────────────────────────────────
    base = repos.copy()
    for df_j in [rf_wide, sec, uc, commit_agg, contrib_agg, new_contrib,
                 org_div, issue_agg, issue_resp, pr_agg, pr_resp, pr_merge,
                 release_agg, comment_agg, bus_factor]:
        base = base.merge(df_j, on="repo_id", how="left")

    def f(col):
        return base.get(col, pd.Series(0, index=base.index)).fillna(0).astype(bool)

    # ── Assemble output columns ───────────────────────────────────────────────
    out = pd.DataFrame()
    out["repo_id"]   = base["repo_id"]
    out["html_url"]  = base["html_url"]
    out["stage_raw"] = base["stage_raw"]

    # Community
    out["unique_contributors"]         = base.get("unique_contributors", pd.Series(0, index=base.index)).fillna(0).astype(int)
    out["contributor_count_90d"]       = base.get("contributor_count_90d", pd.Series(0, index=base.index)).fillna(0).astype(int)
    out["new_contributors_90d"]        = base.get("new_contributors_90d", pd.Series(0, index=base.index)).fillna(0).astype(int)
    out["organizational_diversity_90d"]= base.get("organizational_diversity_90d", pd.Series(0, index=base.index)).fillna(0).astype(int)
    out["bus_factor"]                  = base.get("bus_factor")

    # Activity
    out["commit_frequency_90d"]        = (base.get("commits_90d", pd.Series(0, index=base.index)).fillna(0) / 12.857).round(2)
    out["release_frequency_6m"]        = (base.get("releases_6m", pd.Series(0, index=base.index)).fillna(0) / 6).round(2)
    closed  = base.get("closed_issues_90d", pd.Series(0, index=base.index)).fillna(0)
    open90  = base.get("open_issues_90d",   pd.Series(0, index=base.index)).fillna(0)
    total90 = closed + open90
    out["issue_closed_ratio_90d"]      = (closed / total90.replace(0, np.nan)).round(3)
    merged  = base.get("total_merged_prs", pd.Series(0, index=base.index)).fillna(0)
    reviewed= base.get("reviewed_prs",     pd.Series(0, index=base.index)).fillna(0)
    out["code_review_practices_90d"]   = (reviewed / merged.replace(0, np.nan)).round(3)

    # Response times
    out["issue_response_time_90d"]     = base.get("issue_response_time_90d")
    out["pr_response_time_90d"]        = base.get("pr_response_time_90d")
    out["pr_merge_time_90d"]           = base.get("pr_merge_time_90d")

    # Social
    c30 = base.get("commits_30d",  pd.Series(0, index=base.index)).fillna(0)
    i30 = base.get("issues_30d",   pd.Series(0, index=base.index)).fillna(0)
    p30 = base.get("prs_30d",      pd.Series(0, index=base.index)).fillna(0)
    m30 = base.get("comments_30d", pd.Series(0, index=base.index)).fillna(0)
    out["social_activity_30d"]         = (c30 + i30 + p30 + m30).astype(int)

    # Documentation (boolean)
    out["readme"]          = f("readme")
    out["contributing"]    = f("contributing")
    out["code_of_conduct"] = f("coc")
    out["governance"]      = f("governance")
    out["changelog"]       = f("changelog")

    # Dev Practices (boolean)
    out["ci_cd"]               = f("ci_any")
    out["test_presence"]       = f("test_dir")
    out["branch_protection"]   = base.get("has_branch_protection", pd.Series(0, index=base.index)).fillna(0).astype(bool)
    out["security_policy"]     = f("security")
    out["dependency_updates"]  = (base.get("has_dependabot", pd.Series(0, index=base.index)).fillna(0).astype(bool) |
                                  base.get("has_renovate",   pd.Series(0, index=base.index)).fillna(0).astype(bool) |
                                  f("dependabot") | f("renovate"))

    return out


def load_cncf(conn) -> pd.DataFrame:
    """Load + stage CNCF repos. Stage comes from cncf_stage column."""
    df = compute_metrics(conn)
    df = df[df["stage_raw"].isin(["sandbox","incubating","graduated"])].copy()
    df["stage"] = df["stage_raw"].str.capitalize()
    return df


def load_curioss(conn) -> pd.DataFrame:
    """
    Load CURIOSS repos + assign maturity stage via k-means k=3
    on log1p(unique_contributors).
    """
    df = compute_metrics(conn)

    # Filter: keep repos with any activity (unique_contributors > 0)
    df = df[df["unique_contributors"] > 0].copy()

    X       = np.log1p(df["unique_contributors"].values).reshape(-1, 1)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    km      = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=500)
    df      = df.copy()
    df["cluster"] = km.fit_predict(X_sc)

    order = (df.groupby("cluster")["unique_contributors"]
               .apply(lambda s: np.log1p(s).mean())
               .sort_values().index)
    label_map = {old: ["Sandbox","Incubating","Graduated"][new]
                 for new, old in enumerate(order)}
    df["stage"] = df["cluster"].map(label_map)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  KDE HELPER  (log1p space, same approach as ridge_nc5yr.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_kde(values, x_grid, log_transform=True, min_bw=0.20):
    vals = np.log1p(values) if log_transform else np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.zeros_like(x_grid)
    if np.std(vals) < 1e-8:
        vals = vals + np.random.default_rng(0).normal(0, 0.05, len(vals))
    std   = np.std(vals)
    scott = std * len(vals) ** (-0.20)
    bw    = max(scott, min_bw)
    kde   = gaussian_kde(vals, bw_method=bw / std if std > 0 else min_bw)
    return kde(x_grid)


# ══════════════════════════════════════════════════════════════════════════════
#  RIDGE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_ridge_category(cncf, curioss, metrics, category_name):
    """
    One figure: 3 rows (stages) × N columns (metrics).
    Each cell: KDE for CNCF (blue) + CURIOSS (orange) overlaid.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(3, n_metrics,
                             figsize=(4.5 * n_metrics, 7),
                             squeeze=False)

    fig.suptitle(f"Ridge Plots — {category_name}\nCNCF (blue) vs. CURIOSS (orange) by Maturity Stage",
                 fontsize=13, fontweight="bold", y=1.01)

    for col, (col_name, label, use_log, x_max) in enumerate(metrics):
        for row, stage in enumerate(STAGES):
            ax = axes[row][col]

            c_raw  = cncf.loc[cncf["stage"]   == stage, col_name].dropna().values
            cu_raw = curioss.loc[curioss["stage"] == stage, col_name].dropna().values

            # x grid
            x_max_g = np.log1p(x_max) if use_log else x_max
            x_min_g = np.log1p(0)     if use_log else 0
            x_grid  = np.linspace(x_min_g, x_max_g, 600)

            for vals, fill_c, line_c, lbl in [
                (c_raw,  CNCF_LIGHT, CNCF_COL, "CNCF"),
                (cu_raw, CUR_LIGHT,  CUR_COL,  "CURIOSS"),
            ]:
                if len(vals) < 2:
                    continue
                if np.unique(vals).size == 1:
                    v = vals[0]
                    vpos = np.log1p(v) if use_log else v
                    ax.axvline(vpos, color=line_c, linewidth=2.0, alpha=0.8)
                    continue
                dens = compute_kde(vals, x_grid, log_transform=use_log)
                if dens.max() == 0:
                    continue
                dens = dens / dens.max()
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                ax.fill_between(x_grid, dens, color=fill_c, alpha=0.85, linewidth=0)
                ax.plot(x_grid, dens, color=line_c, linewidth=1.3)
                for qv in [q1, med, q3]:
                    qpos = np.log1p(qv) if use_log else qv
                    ax.axvline(qpos, color=line_c, linestyle="--", linewidth=0.7, alpha=0.7)

            # x-axis ticks
            if use_log:
                raw_ticks = [t for t in [0,1,2,5,10,20,50,100,200,500,1000,2000]
                             if t <= x_max]
                tick_pos  = [np.log1p(t) for t in raw_ticks]
                ax.set_xticks(tick_pos)
                ax.set_xticklabels([str(t) for t in raw_ticks], fontsize=6, rotation=45)
            else:
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax.tick_params(axis="x", labelsize=6)

            ax.set_xlim(x_min_g, x_max_g)
            ax.set_yticks([])
            ax.spines[["top","right","left"]].set_visible(False)
            ax.axhline(0, color="#cccccc", linewidth=0.6)

            n_c  = len(c_raw)
            n_cu = len(cu_raw)
            ax.set_title(f"CNCF n={n_c}  CUR n={n_cu:,}", fontsize=6.5, pad=2)

            if col == 0:
                ax.set_ylabel(stage, fontsize=10, fontweight="bold",
                              rotation=0, labelpad=60, va="center")
            if row == 2:
                ax.set_xlabel(label, fontsize=8)

    # Legend
    legend_els = [
        mpatches.Patch(facecolor=CNCF_LIGHT, edgecolor=CNCF_COL, label="CNCF"),
        mpatches.Patch(facecolor=CUR_LIGHT,  edgecolor=CUR_COL,  label="CURIOSS"),
    ]
    fig.legend(handles=legend_els, loc="upper right", fontsize=9,
               framealpha=0.9, bbox_to_anchor=(1.0, 1.00))

    plt.tight_layout()
    out = OUT_DIR / f"ridge_{category_name.lower().replace(' ','_')}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  VIOLIN PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_violin_category(cncf, curioss, metrics, category_name):
    """
    One subplot per metric; within each: grouped violin by stage,
    CNCF (left, blue) and CURIOSS (right, orange).
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    fig.suptitle(f"Violin Plots — {category_name}\nCNCF (blue) vs. CURIOSS (orange) by Maturity Stage",
                 fontsize=13, fontweight="bold")

    for col, (col_name, label, use_log, x_max) in enumerate(metrics):
        ax = axes[0][col]
        positions_c  = [1, 4, 7]
        positions_cu = [2, 5, 8]
        tick_pos     = [1.5, 4.5, 7.5]

        for pos_c, pos_cu, stage in zip(positions_c, positions_cu, STAGES):
            c_raw  = cncf.loc[cncf["stage"]   == stage, col_name].dropna().values
            cu_raw = curioss.loc[curioss["stage"] == stage, col_name].dropna().values

            if use_log:
                c_raw  = np.log1p(c_raw)
                cu_raw = np.log1p(cu_raw)

            for vals, pos, color, light in [
                (c_raw,  pos_c,  CNCF_COL, CNCF_LIGHT),
                (cu_raw, pos_cu, CUR_COL,  CUR_LIGHT),
            ]:
                vals = vals[np.isfinite(vals)]
                if len(vals) < 3:
                    ax.scatter([pos], [vals.mean() if len(vals) else 0],
                               color=color, s=30, zorder=3)
                    continue
                vp = ax.violinplot(vals, positions=[pos], widths=0.7,
                                   showmedians=False, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(light)
                    body.set_edgecolor(color)
                    body.set_alpha(0.85)
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                ax.vlines(pos, q1, q3, color=color, linewidth=2.5)
                ax.scatter([pos], [med], color=color, s=20, zorder=3)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(STAGES, fontsize=8)
        ax.set_xlim(0, 9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        y_lbl = f"log(x+1)" if use_log else ""
        ax.set_ylabel(y_lbl, fontsize=7, color="#555")
        ax.set_title(label, fontsize=9, fontweight="bold")

    legend_els = [
        mpatches.Patch(facecolor=CNCF_LIGHT, edgecolor=CNCF_COL, label="CNCF"),
        mpatches.Patch(facecolor=CUR_LIGHT,  edgecolor=CUR_COL,  label="CURIOSS"),
    ]
    fig.legend(handles=legend_els, loc="upper right", fontsize=9,
               framealpha=0.9, bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    out = OUT_DIR / f"violin_{category_name.lower().replace(' ','_')}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BOX PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_box_category(cncf, curioss, metrics, category_name):
    """
    One subplot per metric; side-by-side box plots per stage, CNCF vs CURIOSS.
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    fig.suptitle(f"Box Plots — {category_name}\nCNCF (blue) vs. CURIOSS (orange) by Maturity Stage",
                 fontsize=13, fontweight="bold")

    for col, (col_name, label, use_log, x_max) in enumerate(metrics):
        ax = axes[0][col]
        data_c, data_cu, labels = [], [], []

        for stage in STAGES:
            c_raw  = cncf.loc[cncf["stage"]   == stage, col_name].dropna().values
            cu_raw = curioss.loc[curioss["stage"] == stage, col_name].dropna().values
            if use_log:
                c_raw  = np.log1p(c_raw)
                cu_raw = np.log1p(cu_raw)
            data_c.append(c_raw[np.isfinite(c_raw)])
            data_cu.append(cu_raw[np.isfinite(cu_raw)])
            labels.append(stage)

        positions_c  = [1, 4, 7]
        positions_cu = [2, 5, 8]

        bp_c = ax.boxplot(data_c, positions=positions_c, widths=0.7,
                          patch_artist=True, showfliers=False,
                          medianprops=dict(color="white", linewidth=2))
        bp_cu = ax.boxplot(data_cu, positions=positions_cu, widths=0.7,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color="white", linewidth=2))
        for patch in bp_c["boxes"]:
            patch.set_facecolor(CNCF_MED); patch.set_edgecolor(CNCF_COL); patch.set_alpha(0.85)
        for patch in bp_cu["boxes"]:
            patch.set_facecolor(CUR_MED); patch.set_edgecolor(CUR_COL); patch.set_alpha(0.85)
        for w in bp_c["whiskers"] + bp_c["caps"]:
            w.set_color(CNCF_COL)
        for w in bp_cu["whiskers"] + bp_cu["caps"]:
            w.set_color(CUR_COL)

        ax.set_xticks([1.5, 4.5, 7.5])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(0, 9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        y_lbl = "log(x+1)" if use_log else ""
        ax.set_ylabel(y_lbl, fontsize=7, color="#555")
        ax.set_title(label, fontsize=9, fontweight="bold")

    legend_els = [
        mpatches.Patch(facecolor=CNCF_MED, edgecolor=CNCF_COL, label="CNCF"),
        mpatches.Patch(facecolor=CUR_MED,  edgecolor=CUR_COL,  label="CURIOSS"),
    ]
    fig.legend(handles=legend_els, loc="upper right", fontsize=9,
               framealpha=0.9, bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    out = OUT_DIR / f"box_{category_name.lower().replace(' ','_')}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  BINARY / BOOLEAN BAR CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_binary_category(cncf, curioss, metrics, category_name):
    """
    For each boolean metric, grouped bar chart: proportion True per stage,
    CNCF (blue) vs CURIOSS (orange).
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), squeeze=False)
    fig.suptitle(f"Adoption Rate — {category_name}\nProportion of repos with feature enabled",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(STAGES))
    width = 0.35

    for col, (col_name, label) in enumerate(metrics):
        ax = axes[0][col]

        vals_c, vals_cu = [], []
        for stage in STAGES:
            c_sub  = cncf.loc[cncf["stage"]   == stage, col_name]
            cu_sub = curioss.loc[curioss["stage"] == stage, col_name]
            vals_c.append( c_sub.astype(bool).mean()  if len(c_sub)  > 0 else 0)
            vals_cu.append(cu_sub.astype(bool).mean() if len(cu_sub) > 0 else 0)

        ax.bar(x - width/2, vals_c,  width, color=CNCF_MED, edgecolor=CNCF_COL,
               alpha=0.85, label="CNCF")
        ax.bar(x + width/2, vals_cu, width, color=CUR_MED,  edgecolor=CUR_COL,
               alpha=0.85, label="CURIOSS")

        for i, (vc, vcu) in enumerate(zip(vals_c, vals_cu)):
            ax.text(i - width/2, vc  + 0.02, f"{vc:.0%}",  ha="center", fontsize=7)
            ax.text(i + width/2, vcu + 0.02, f"{vcu:.0%}", ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(STAGES, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Proportion True", fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    legend_els = [
        mpatches.Patch(facecolor=CNCF_MED, edgecolor=CNCF_COL, label="CNCF"),
        mpatches.Patch(facecolor=CUR_MED,  edgecolor=CUR_COL,  label="CURIOSS"),
    ]
    fig.legend(handles=legend_els, loc="upper right", fontsize=9,
               framealpha=0.9, bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    out = OUT_DIR / f"binary_{category_name.lower().replace(' ','_')}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(cncf, curioss):
    print(f"\n{'='*80}")
    print(f"  Dataset sizes:")
    for ds, df in [("CNCF", cncf), ("CURIOSS", curioss)]:
        counts = df["stage"].value_counts()
        print(f"    {ds}: {counts.to_dict()}")

    print(f"\n  Continuous metrics Q1 / Median / Q3:")
    print(f"  {'Stage':<12} {'Dataset':<10} {'Metric':<35} {'Q1':>8}  {'Med':>8}  {'Q3':>8}")
    print(f"  {'-'*90}")
    for cat, mlist in CONTINUOUS.items():
        for col_name, label, *_ in mlist:
            for stage in STAGES:
                for lbl, df in [("CNCF", cncf), ("CURIOSS", curioss)]:
                    sub = df.loc[df["stage"] == stage, col_name].dropna()
                    if len(sub) == 0:
                        continue
                    q1, med, q3 = np.percentile(sub, [25, 50, 75])
                    print(f"  {stage:<12} {lbl:<10} {col_name:<35} {q1:>8.2f}  {med:>8.2f}  {q3:>8.2f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading CNCF data from DB…")
    with sqlite3.connect(CNCF_DB) as conn_c:
        cncf = load_cncf(conn_c)

    print("Loading CURIOSS data from DB…")
    with sqlite3.connect(CURIOSS_DB) as conn_cu:
        curioss = load_curioss(conn_cu)

    print(f"\nCNCF:   {len(cncf):,} repos  {cncf['stage'].value_counts().to_dict()}")
    print(f"CURIOSS:{len(curioss):,} repos  {curioss['stage'].value_counts().to_dict()}\n")

    print_summary(cncf, curioss)

    # ── Ridge plots ────────────────────────────────────────────────────────────
    print("Generating ridge plots…")
    for cat, mlist in CONTINUOUS.items():
        plot_ridge_category(cncf, curioss, mlist, cat)

    # ── Violin plots ───────────────────────────────────────────────────────────
    print("\nGenerating violin plots…")
    for cat, mlist in CONTINUOUS.items():
        plot_violin_category(cncf, curioss, mlist, cat)

    # ── Box plots ──────────────────────────────────────────────────────────────
    print("\nGenerating box plots…")
    for cat, mlist in CONTINUOUS.items():
        plot_box_category(cncf, curioss, mlist, cat)

    # ── Binary bar charts ──────────────────────────────────────────────────────
    print("\nGenerating binary/boolean bar charts…")
    for cat, mlist in BINARY.items():
        plot_binary_category(cncf, curioss, mlist, cat)

    print(f"\n✅  All plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
