"""
compute_curioss_metrics.py
==========================
Computes all 42 sustainability metrics for CURIOSS repositories.
Uses SQL aggregations to handle 34k repos / 2.6M commits efficiently.

Output: curioss_temp/curioss_metrics.csv
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB      = Path(__file__).parent / "curioss_raw.db"
SC_CSV  = Path(__file__).parent / "scorecard_results.csv"
OUT_CSV = Path(__file__).parent / "curioss_metrics.csv"

TODAY = datetime(2026, 8, 11, tzinfo=timezone.utc)
D90   = TODAY - timedelta(days=90)
D30   = TODAY - timedelta(days=30)
D180  = TODAY - timedelta(days=180)
D365  = TODAY - timedelta(days=365)

def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def pct_change(a, b):
    if pd.isna(b) or b == 0:
        return np.nan
    return round((float(a) - float(b)) / float(b) * 100, 1)


def main():
    print("Connecting…")
    conn = sqlite3.connect(DB)

    # ── 1. Base repos ──────────────────────────────────────────────────────────
    print("Loading repositories…")
    repos = pd.read_sql("SELECT id AS repo_id, full_name, cncf_stage, stargazers_count, "
                        "forks_count, watchers_count, created_at, license_key "
                        "FROM repositories", conn)

    # ── 2. Repo files (pivot to wide) ─────────────────────────────────────────
    print("Loading repo_files…")
    rf = pd.read_sql("SELECT repo_id, file_key, exists_flag FROM repo_files", conn)
    rf_wide = rf.pivot_table(index="repo_id", columns="file_key",
                              values="exists_flag", fill_value=0).reset_index()

    # ── 3. Repo security ──────────────────────────────────────────────────────
    print("Loading repo_security…")
    sec = pd.read_sql("SELECT repo_id, requires_pr_review, has_branch_protection, "
                      "has_signed_releases, has_dependabot, has_renovate, "
                      "open_advisories_count FROM repo_security", conn)

    # ── 4. Scorecard ──────────────────────────────────────────────────────────
    sc_db = pd.read_sql("SELECT repo_id, overall_score, vulnerabilities "
                        "FROM scorecard_results", conn)
    if len(sc_db) == 0 and SC_CSV.exists():
        print("Scorecard DB empty — loading from CSV…")
        sc_csv = pd.read_csv(SC_CSV, low_memory=False)
        url_map = pd.read_sql("SELECT id AS repo_id, html_url FROM repositories", conn)
        # Rename CSV columns to snake_case to match DB schema
        col_map = {
            "html_url":              "html_url_sc",
            "Vulnerabilities":       "vulnerabilities",
            "Branch-Protection":     "sc_branch_protection",
            "Security-Policy":       "sc_security_policy",
            "Signed-Releases":       "sc_signed_releases",
            "Dependency-Update-Tool":"sc_dependency_update_tool",
        }
        sc_csv = sc_csv.rename(columns={k: v for k, v in col_map.items() if k in sc_csv.columns})
        sc_db = sc_csv.merge(url_map, left_on="html_url_sc", right_on="html_url", how="inner")
        extra_cols = [c for c in ["vulnerabilities","sc_branch_protection","sc_security_policy",
                                   "sc_signed_releases","sc_dependency_update_tool"]
                      if c in sc_db.columns]
        sc_db = sc_db[["repo_id", "overall_score"] + extra_cols].copy()
        # Convert scorecard numeric columns (may have stray strings from CSV export)
        for col in extra_cols:
            sc_db[col] = pd.to_numeric(sc_db[col], errors="coerce")

    # ── 5. SQL-aggregated metrics ──────────────────────────────────────────────
    print("Computing commit metrics via SQL…")

    # commit counts
    commit_agg = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(*) FILTER (WHERE authored_at >= '{iso(D90)}') AS commits_90d,
               COUNT(*) FILTER (WHERE authored_at >= '{iso(D90)}' AND authored_at < '{iso(TODAY)}') AS commits_last90,
               COUNT(*) FILTER (WHERE authored_at >= '{iso(D180)}' AND authored_at < '{iso(D90)}') AS commits_prior90
        FROM commits
        WHERE authored_at IS NOT NULL
        GROUP BY repo_id
    """, conn)

    # distinct contributors (no correlated subquery)
    contrib_agg = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(DISTINCT CASE WHEN authored_at >= '{iso(D90)}' THEN author_login END) AS contributor_count_90d,
               COUNT(DISTINCT CASE WHEN authored_at >= '{iso(D180)}' THEN author_login END) AS authors_last6,
               COUNT(DISTINCT CASE WHEN authored_at >= '{iso(D365)}' AND authored_at < '{iso(D180)}' THEN author_login END) AS authors_prior6
        FROM commits
        WHERE authored_at IS NOT NULL AND author_login IS NOT NULL
        GROUP BY repo_id
    """, conn)

    # new_contributors_90d — authors in 90d window not seen before it (two passes)
    print("Computing new contributors…")
    a90_set    = pd.read_sql(f"SELECT repo_id, author_login FROM commits "
                              f"WHERE authored_at >= '{iso(D90)}' AND author_login IS NOT NULL "
                              f"GROUP BY repo_id, author_login", conn)
    abefore_set = pd.read_sql(f"SELECT repo_id, author_login FROM commits "
                               f"WHERE authored_at < '{iso(D90)}' AND author_login IS NOT NULL "
                               f"GROUP BY repo_id, author_login", conn)
    a90_set["in_before"] = a90_set.set_index(["repo_id","author_login"]).index.isin(
        abefore_set.set_index(["repo_id","author_login"]).index)
    new_contrib_agg = (a90_set[~a90_set["in_before"]]
                       .groupby("repo_id").size()
                       .reset_index(name="new_contributors_90d"))

    # org diversity (non-personal domains, 90d)
    PERSONAL = "('gmail.com','yahoo.com','hotmail.com','outlook.com','users.noreply.github.com','noreply.github.com')"
    org_div = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(DISTINCT author_email_domain) AS organizational_diversity_90d
        FROM commits
        WHERE authored_at >= '{iso(D90)}'
          AND author_email_domain IS NOT NULL
          AND author_email_domain != ''
          AND author_email_domain NOT IN {PERSONAL}
        GROUP BY repo_id
    """, conn)

    print("Computing issue metrics via SQL…")
    issue_agg = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(*) FILTER (WHERE created_at >= '{iso(D90)}') AS issues_90d,
               COUNT(*) FILTER (WHERE created_at >= '{iso(D90)}' AND state='CLOSED') AS closed_issues_90d,
               COUNT(*) FILTER (WHERE created_at >= '{iso(D90)}' AND state='OPEN')  AS open_issues_90d
        FROM issues
        WHERE created_at IS NOT NULL
        GROUP BY repo_id
    """, conn)

    issue_resp = pd.read_sql(f"""
        SELECT repo_id,
               ROUND(AVG(
                   (julianday(first_response_at) - julianday(created_at))
               ), 1) AS issue_response_time_90d
        FROM issues
        WHERE created_at >= '{iso(D90)}'
          AND first_response_at IS NOT NULL
          AND first_response_at >= created_at
        GROUP BY repo_id
    """, conn)

    print("Computing PR metrics via SQL…")
    pr_agg = pd.read_sql(f"""
        SELECT repo_id,
               COUNT(*) FILTER (WHERE created_at >= '{iso(D90)}') AS prs_90d,
               COUNT(*) FILTER (WHERE merged_at >= '{iso(D90)}') AS total_merged_prs,
               COUNT(*) FILTER (WHERE merged_at >= '{iso(D90)}' AND review_count > 0) AS reviewed_prs
        FROM pull_requests
        WHERE created_at IS NOT NULL
        GROUP BY repo_id
    """, conn)

    pr_resp = pd.read_sql(f"""
        SELECT repo_id,
               ROUND(AVG(julianday(first_review_at) - julianday(created_at)), 1) AS pr_response_time_90d
        FROM pull_requests
        WHERE created_at >= '{iso(D90)}'
          AND first_review_at IS NOT NULL
          AND first_review_at >= created_at
        GROUP BY repo_id
    """, conn)

    pr_merge = pd.read_sql(f"""
        SELECT repo_id,
               ROUND(AVG(julianday(merged_at) - julianday(created_at)), 1) AS pr_merge_time_90d
        FROM pull_requests
        WHERE merged_at >= '{iso(D90)}'
          AND merged_at IS NOT NULL
          AND merged_at >= created_at
        GROUP BY repo_id
    """, conn)

    print("Computing comment/release metrics via SQL…")
    comment_agg = pd.read_sql(f"""
        SELECT repo_id, COUNT(*) AS comments_90d
        FROM issue_comments
        WHERE created_at >= '{iso(D90)}'
        GROUP BY repo_id
    """, conn)

    release_agg = pd.read_sql(f"""
        SELECT repo_id, COUNT(*) AS releases_90d
        FROM releases
        WHERE published_at >= '{iso(D90)}'
        GROUP BY repo_id
    """, conn)

    print("Computing platform OS support via SQL…")
    wf_raw = pd.read_sql("SELECT repo_id, os_targets FROM repo_workflows "
                         "WHERE os_targets IS NOT NULL", conn)
    os_support = wf_raw.groupby("repo_id")["os_targets"].apply(
        lambda x: len(set(o.strip() for val in x for o in str(val).split(",") if o.strip()))
    ).reset_index().rename(columns={"os_targets": "platform_os_support"})

    print("Computing broad unique contributors (commits+PRs+issues+PR reviews)…")
    unique_contrib = pd.read_sql("""
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
    """, conn)

    print("Computing bus factor via SQL…")
    bf_raw = pd.read_sql("""
        SELECT repo_id, author_login, COUNT(*) AS cnt
        FROM commits
        WHERE author_login IS NOT NULL
        GROUP BY repo_id, author_login
        ORDER BY repo_id, cnt DESC
    """, conn)

    def calc_bus_factor(grp):
        total = grp["cnt"].sum()
        if total == 0:
            return np.nan
        running = 0
        for i, cnt in enumerate(grp["cnt"], 1):
            running += cnt
            if running / total > 0.5:
                return i
        return i

    bus_factor = bf_raw.groupby("repo_id").apply(calc_bus_factor).reset_index()
    bus_factor.columns = ["repo_id", "bus_factor"]

    print("Computing maintainer metrics via SQL…")
    # Core = top-quartile contributors by contribution count
    rc_raw = pd.read_sql("SELECT repo_id, contributor_login, contributions "
                         "FROM repo_contributors", conn)
    q75 = rc_raw.groupby("repo_id")["contributions"].transform(lambda x: x.quantile(0.75))
    core_set = (rc_raw[rc_raw["contributions"] >= q75]
                .groupby("repo_id")["contributor_login"]
                .apply(set).reset_index()
                .rename(columns={"contributor_login": "core_logins"}))

    # Last activity date by core contributors across all event types (maintainer_activity_recency)
    last_activity_sql = pd.read_sql(f"""
        SELECT repo_id, login, MAX(last_at) AS last_at FROM (
            SELECT repo_id, author_login AS login, MAX(authored_at)   AS last_at FROM commits        WHERE author_login  IS NOT NULL GROUP BY repo_id, author_login
            UNION ALL
            SELECT repo_id, author_login AS login, MAX(created_at)    AS last_at FROM issues         WHERE author_login  IS NOT NULL GROUP BY repo_id, author_login
            UNION ALL
            SELECT repo_id, author_login AS login, MAX(created_at)    AS last_at FROM pull_requests  WHERE author_login  IS NOT NULL GROUP BY repo_id, author_login
            UNION ALL
            SELECT repo_id, author_login AS login, MAX(created_at)    AS last_at FROM issue_comments WHERE author_login  IS NOT NULL GROUP BY repo_id, author_login
            UNION ALL
            SELECT repo_id, merged_by    AS login, MAX(merged_at)     AS last_at FROM pull_requests  WHERE merged_by     IS NOT NULL GROUP BY repo_id, merged_by
        ) GROUP BY repo_id, login
    """, conn)
    last_activity_sql = last_activity_sql.merge(core_set, on="repo_id", how="left")
    last_activity_sql["is_core"] = last_activity_sql.apply(
        lambda r: r["login"] in r["core_logins"] if isinstance(r.get("core_logins"), set) else False,
        axis=1)
    last_core = (last_activity_sql[last_activity_sql["is_core"]]
                 .groupby("repo_id")["last_at"].max()
                 .reset_index().rename(columns={"last_at": "last_core_at"}))

    last_core["last_core_at"] = pd.to_datetime(last_core["last_core_at"], utc=True, errors="coerce")
    last_core["maintainer_activity_recency_days"] = (TODAY - last_core["last_core_at"]).dt.days

    # Activity by login in W90: commits, issues, PRs, comments, merges
    def events_by_login(sql, login_col="author_login"):
        df = pd.read_sql(sql, conn)
        df = df.merge(core_set, on="repo_id", how="left")
        df["is_core"] = df.apply(
            lambda r: r[login_col] in r["core_logins"] if isinstance(r.get("core_logins"), set) else False,
            axis=1)
        return df

    c90 = events_by_login(f"""
        SELECT repo_id, author_login, COUNT(*) AS cnt FROM commits
        WHERE authored_at >= '{iso(D90)}' AND author_login IS NOT NULL
        GROUP BY repo_id, author_login""")

    i90 = events_by_login(f"""
        SELECT repo_id, author_login, COUNT(*) AS cnt FROM issues
        WHERE created_at >= '{iso(D90)}' AND author_login IS NOT NULL
        GROUP BY repo_id, author_login""")

    p90 = events_by_login(f"""
        SELECT repo_id, author_login, COUNT(*) AS cnt FROM pull_requests
        WHERE created_at >= '{iso(D90)}' AND author_login IS NOT NULL
        GROUP BY repo_id, author_login""")

    com90 = events_by_login(f"""
        SELECT repo_id, author_login, COUNT(*) AS cnt FROM issue_comments
        WHERE created_at >= '{iso(D90)}' AND author_login IS NOT NULL
        GROUP BY repo_id, author_login""")

    merge90 = events_by_login(f"""
        SELECT repo_id, merged_by AS author_login, COUNT(*) AS cnt FROM pull_requests
        WHERE merged_at >= '{iso(D90)}' AND merged_by IS NOT NULL
        GROUP BY repo_id, merged_by""")

    def sum_by_core(df, is_core):
        return (df[df["is_core"] == is_core]
                .groupby("repo_id")["cnt"].sum().reset_index())

    # Maintainers: commits + issues + PRs + comments + merges by core
    maint_parts = [sum_by_core(df, True) for df in [c90, i90, p90, com90, merge90]]
    maint_activity = maint_parts[0].copy().rename(columns={"cnt": "total"})
    for part in maint_parts[1:]:
        maint_activity = maint_activity.merge(part, on="repo_id", how="outer")
        maint_activity["total"] = maint_activity["total"].fillna(0) + maint_activity["cnt"].fillna(0)
        maint_activity = maint_activity.drop(columns=["cnt"])
    maint_activity.columns = ["repo_id", "maint_activity_90"]

    # Contributors: commits + issues + PRs + comments by non-core
    contrib_parts = [sum_by_core(df, False) for df in [c90, i90, p90, com90]]
    contrib_activity = contrib_parts[0].copy().rename(columns={"cnt": "total"})
    for part in contrib_parts[1:]:
        contrib_activity = contrib_activity.merge(part, on="repo_id", how="outer")
        contrib_activity["total"] = contrib_activity["total"].fillna(0) + contrib_activity["cnt"].fillna(0)
        contrib_activity = contrib_activity.drop(columns=["cnt"])
    contrib_activity.columns = ["repo_id", "contributors_dev_activity_90d"]

    # Keep maint_commits separately for other uses (recency etc.)
    maint_commits = sum_by_core(c90, True).rename(columns={"cnt": "maint_commits_90"})

    # Contributor retention 90d (current 90d vs prior 90d, i.e. t-180d to t-90d)
    authors_last90 = pd.read_sql(f"""
        SELECT repo_id, author_login
        FROM commits
        WHERE authored_at >= '{iso(D90)}' AND author_login IS NOT NULL
        GROUP BY repo_id, author_login
    """, conn)
    authors_prior90 = pd.read_sql(f"""
        SELECT repo_id, author_login
        FROM commits
        WHERE authored_at >= '{iso(D180)}' AND authored_at < '{iso(D90)}'
          AND author_login IS NOT NULL
        GROUP BY repo_id, author_login
    """, conn)
    authors_last90_set  = authors_last90.groupby("repo_id")["author_login"].apply(set).reset_index()
    authors_prior90_set = authors_prior90.groupby("repo_id")["author_login"].apply(set).reset_index()
    retention = authors_prior90_set.merge(authors_last90_set, on="repo_id", how="left",
                                           suffixes=("_prior","_last"))
    retention["contributor_retention_90d_pct"] = retention.apply(
        lambda r: round(len(r["author_login_prior"] & (r["author_login_last"] if isinstance(r["author_login_last"], set) else set())) /
                        len(r["author_login_prior"]) * 100, 1)
        if len(r["author_login_prior"]) > 0 else np.nan, axis=1)
    # contributor growth: current 90d vs prior 90d (same windows as retention)
    growth = retention[["repo_id","author_login_prior","author_login_last"]].copy()
    growth["contributor_growth_90d_pct"] = growth.apply(
        lambda r: pct_change(
            len(r["author_login_last"]) if isinstance(r["author_login_last"], set) else 0,
            len(r["author_login_prior"])),
        axis=1)
    retention_out = retention[["repo_id","contributor_retention_90d_pct"]].merge(
        growth[["repo_id","contributor_growth_90d_pct"]], on="repo_id", how="outer")

    conn.close()

    # ── Merge everything onto base repos ───────────────────────────────────────
    print("Merging all aggregations…")
    base = repos.copy()

    for df_join, key in [
        (rf_wide,         "repo_id"),
        (sec,             "repo_id"),
        (sc_db,           "repo_id"),
        (commit_agg,      "repo_id"),
        (contrib_agg,     "repo_id"),
        (new_contrib_agg, "repo_id"),
        (org_div,         "repo_id"),
        (issue_agg,       "repo_id"),
        (issue_resp,      "repo_id"),
        (pr_agg,          "repo_id"),
        (pr_resp,         "repo_id"),
        (pr_merge,        "repo_id"),
        (comment_agg,     "repo_id"),
        (release_agg,     "repo_id"),
        (os_support,      "repo_id"),
        (bus_factor,      "repo_id"),
        (last_core[["repo_id","maintainer_activity_recency_days"]], "repo_id"),
        (maint_commits,    "repo_id"),
        (maint_activity,   "repo_id"),
        (contrib_activity, "repo_id"),
        (retention_out,   "repo_id"),
        (unique_contrib,  "repo_id"),
    ]:
        base = base.merge(df_join, on=key, how="left")

    def f(col):
        return base.get(col, pd.Series(0, index=base.index)).fillna(0).astype(bool)

    # ── Assemble final output ──────────────────────────────────────────────────
    print("Assembling output…")
    out = pd.DataFrame()
    out["project"]    = base["full_name"]
    out["university"] = base["cncf_stage"]

    # Activity
    out["issue_response_time_90d"]          = base.get("issue_response_time_90d")
    out["pr_response_time_90d"]             = base.get("pr_response_time_90d")
    out["pr_merge_time_90d"]                = base.get("pr_merge_time_90d")
    out["social_activity_90d"]              = (base.get("commits_90d",  pd.Series(0,index=base.index)).fillna(0) +
                                               base.get("issues_90d",   pd.Series(0,index=base.index)).fillna(0) +
                                               base.get("prs_90d",      pd.Series(0,index=base.index)).fillna(0) +
                                               base.get("comments_90d", pd.Series(0,index=base.index)).fillna(0)).astype(int)
    out["commit_frequency_90d"]             = (base.get("commits_90d", pd.Series(0,index=base.index)).fillna(0) / 12.857).round(2)
    out["commit_growth_90d_pct"]            = base.apply(
        lambda r: pct_change(r.get("commits_last90",0) or 0, r.get("commits_prior90",0) or 0), axis=1)
    total90 = base.get("closed_issues_90d", pd.Series(0,index=base.index)).fillna(0) + \
              base.get("open_issues_90d",   pd.Series(0,index=base.index)).fillna(0)
    out["issue_closed_ratio_90d"]           = (base.get("closed_issues_90d", pd.Series(0,index=base.index)).fillna(0) /
                                               total90.replace(0, np.nan)).round(3)
    out["contributors_dev_activity_90d"]    = base.get("contributors_dev_activity_90d", pd.Series(0,index=base.index)).fillna(0).astype(int)
    out["maintainers_dev_activity_90d"]     = base.get("maint_activity_90", pd.Series(0,index=base.index)).fillna(0).astype(int)
    out["code_review_practices_90d"]        = (base.get("reviewed_prs",      pd.Series(0,index=base.index)).fillna(0) /
                                               base.get("total_merged_prs",  pd.Series(0,index=base.index)).fillna(0).replace(0, np.nan)).round(3)
    out["maintainer_activity_recency_days"] = base.get("maintainer_activity_recency_days")
    out["release_frequency_90d"]            = (base.get("releases_90d", pd.Series(0,index=base.index)).fillna(0) / 3).round(2)

    # Development Practices
    out["ci_cd"]                 = f("ci_any")
    build_keys = ["build_cargo","build_cmake","build_gemfile","build_gradle",
                  "build_makefile","build_package_json","build_pom",
                  "build_pyproject","build_setup_py"]
    out["build_environment"]     = sum(f(k).astype(int) for k in build_keys)
    out["test_presence"]         = f("test_dir")
    out["platform_os_support"]   = base.get("platform_os_support", pd.Series(0,index=base.index)).fillna(0).astype(int)

    # Docs
    out["readme"]                     = f("readme")
    out["contributing"]               = f("contributing")
    out["code_of_conduct"]            = f("coc")
    out["governance"]                 = f("governance")
    out["issue_template"]             = f("issue_template")
    out["pr_template"]                = f("pr_template")
    out["code_review_policy"]         = f("codeowners") | base.get("requires_pr_review", pd.Series(0,index=base.index)).fillna(0).astype(bool)
    out["changelog"]                  = f("changelog")
    out["documentation_completeness"] = f("readme").astype(int) + f("contributing").astype(int) + f("governance").astype(int)

    # Funding
    out["financial_support"] = f("funding")

    # Community
    out["organizational_diversity_90d"]  = base.get("organizational_diversity_90d", pd.Series(0,index=base.index)).fillna(0).astype(int)
    out["contributor_count_90d"]         = base.get("contributor_count_90d", pd.Series(0,index=base.index)).fillna(0).astype(int)
    out["contributor_retention_90d_pct"]  = base.get("contributor_retention_90d_pct")
    out["contributor_growth_90d_pct"]     = base.get("contributor_growth_90d_pct")
    out["bus_factor"]                    = base.get("bus_factor")
    out["new_contributors_90d"]          = base["new_contributors_90d"].fillna(0).astype(int)
    out["unique_contributors"]           = base.get("unique_contributors", pd.Series(0, index=base.index)).fillna(0).astype(int)

    # Popularity
    out["community_interest"]    = (base["stargazers_count"].fillna(0) + base["forks_count"].fillna(0) + base["watchers_count"].fillna(0)).astype(int)
    out["downstream_dependents"] = "N/A"

    # General
    created = pd.to_datetime(base["created_at"], utc=True, errors="coerce")
    out["project_age_days"] = (TODAY - created).dt.days
    out["license"]          = base["license_key"].notna() & (base["license_key"].astype(str) != "")

    # Security — prefer OpenSSF Scorecard values (from CSV); fall back to repo_security / repo_files
    # Scorecard uses 0-10 scale; -1 means "not applicable". Treat score > 0 as True.
    def sc_flag(col):
        """Returns a boolean Series: True where scorecard col > 0 (ignores -1 / NaN)."""
        if col in base.columns:
            v = pd.to_numeric(base[col], errors="coerce").fillna(-1)
            return v > 0
        return pd.Series(False, index=base.index)

    out["security_policy"]    = f("security") | sc_flag("sc_security_policy")
    out["branch_protection"]  = sc_flag("sc_branch_protection")
    out["signed_releases"]    = base["has_signed_releases"].fillna(0).astype(bool) | sc_flag("sc_signed_releases")
    out["trusted_publishing"] = "N/A"
    # Scorecard Vulnerabilities: 0-10 where 10=clean, -1=N/A.
    # Invert so >0 means "has known vulnerabilities"; -1 and NaN → NaN.
    sc_v = pd.to_numeric(base.get("vulnerabilities", pd.Series(dtype=float)), errors="coerce")
    out["vulnerability_count"] = sc_v.where(sc_v >= 0, other=np.nan).apply(
        lambda x: max(0, 10 - x) if pd.notna(x) else np.nan
    )
    out["dependency_updates"] = (base["has_dependabot"].fillna(0).astype(bool) |
                                 base["has_renovate"].fillna(0).astype(bool) |
                                 f("dependabot") | f("renovate") |
                                 sc_flag("sc_dependency_update_tool"))

    out.to_csv(OUT_CSV, index=False)
    print(f"\n✅  Saved {len(out):,} rows → {OUT_CSV}")
    print(f"   Universities: {out['university'].nunique()}")
    na = out.isnull().sum()
    print("N/A counts per metric:")
    print(na[na > 0].to_string())


if __name__ == "__main__":
    main()
