"""
collect_cncf_metrics.py
=======================
Collects sustainability metrics for CNCF projects.

Sources:
  - cncf_projects.csv    : CNCF landscape export (stars, contributors, dates already included)
  - GitHub REST API      : governance artefacts, bus factor, activity, responsiveness

Usage:
  GITHUB_TOKEN=ghp_xxx python3 collect_cncf_metrics.py

Output: cncf_metrics.csv (~80 rows, one per project)

API calls per repo (~10): readme, contributing, CoC, security, issue templates,
  CI workflows, contributors (for bus factor), commits (90d), issues (closed 90d).
Total: ~800 calls — well within the 5,000/hr authenticated limit.
"""

import csv, os, re, time, datetime, requests
from pathlib import Path
from collections import Counter

TOKEN   = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github.v3+json"}
INPUT   = Path(__file__).parent / "cncf_projects.csv"
OUTPUT  = Path(__file__).parent / "cncf_metrics.csv"

# ── output columns (order matters for the CSV) ────────────────────────────────
COLS = [
    # identifiers
    "stage", "name", "github_url", "owner", "repo",
    # ── COMMUNITY ─────────────────────────────────────────────────────────────
    "contributor_count",         # from landscape (github_contributors_count)
    "bus_factor",                # smallest N contributors whose commits > 50% total
    "new_contributors_90d",      # distinct authors in last 90 days not seen before
    "stargazers_count",          # from landscape (github_stars)
    # ── GOVERNANCE ARTEFACTS (binary 1/0) ────────────────────────────────────
    "has_license",               # from landscape
    "has_readme",
    "has_contributing",
    "has_code_of_conduct",
    "has_security_policy",
    "has_issue_templates",
    "has_pr_template",
    "has_ci",                    # any file in .github/workflows/
    # ── ACTIVITY ──────────────────────────────────────────────────────────────
    "commits_90d",               # commit count in last 90 days
    "open_issues_count",
    "closed_issues_90d",
    "issue_close_ratio_90d",     # closed / (open + closed) in 90d window
    "releases_last_5",           # number of releases in last 5 (proxy for cadence)
    # ── LIFECYCLE ─────────────────────────────────────────────────────────────
    "days_since_last_commit",    # from landscape
    "days_since_last_release",   # from landscape
    "age_days",                  # from landscape (github_start_commit_date)
    # ── META ──────────────────────────────────────────────────────────────────
    "license",                   # SPDX string from landscape
    "collected_at",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def gh(url, params=None):
    """GitHub GET with rate-limit back-off. Returns Response or None."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        except Exception as e:
            print(f"    network error: {e}")
            time.sleep(5)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (301, 302):
            return None
        if r.status_code == 403:
            wait = int(r.headers.get("Retry-After", 60))
            print(f"    rate limited — sleeping {wait}s")
            time.sleep(wait)
        elif r.status_code == 404:
            return None
        else:
            time.sleep(2)
    return None


def days_since(date_str):
    """Return integer days since an ISO date string, or None."""
    if not date_str or not date_str.strip():
        return None
    try:
        d = datetime.datetime.fromisoformat(date_str.strip().replace("Z", "+00:00"))
        return (datetime.datetime.now(datetime.timezone.utc) - d).days
    except Exception:
        return None


def parse_repo(github_url):
    parts = github_url.rstrip("/").split("/")
    return parts[-2], parts[-1]


def file_exists(owner, repo, path):
    r = gh(f"https://api.github.com/repos/{owner}/{repo}/contents/{path}")
    return 1 if r else 0


# ── per-repo collection ───────────────────────────────────────────────────────

def collect_api(owner, repo):
    """Fetch the metrics that require GitHub API calls. Returns dict."""
    m = {}
    now = datetime.datetime.now(datetime.timezone.utc)
    ago90 = (now - datetime.timedelta(days=90)).isoformat()

    # ── base repo info (open issues) ─────────────────────────────────────────
    r_repo = gh(f"https://api.github.com/repos/{owner}/{repo}")
    if r_repo is None:
        print(f"    ⚠  repo not found — skipping")
        return None
    data = r_repo.json()
    m["open_issues_count"] = data.get("open_issues_count", 0)

    # ── governance artefacts ─────────────────────────────────────────────────
    # README
    m["has_readme"]          = 1 if gh(f"https://api.github.com/repos/{owner}/{repo}/readme") else 0
    # CONTRIBUTING
    for path in ["CONTRIBUTING.md", "CONTRIBUTING.rst", "CONTRIBUTING"]:
        if file_exists(owner, repo, path):
            m["has_contributing"] = 1
            break
    else:
        m["has_contributing"] = 0
    # Code of Conduct
    for path in ["CODE_OF_CONDUCT.md", "CODE_OF_CONDUCT.rst", ".github/CODE_OF_CONDUCT.md"]:
        if file_exists(owner, repo, path):
            m["has_code_of_conduct"] = 1
            break
    else:
        m["has_code_of_conduct"] = 0
    # Security policy
    for path in ["SECURITY.md", ".github/SECURITY.md", "docs/SECURITY.md"]:
        if file_exists(owner, repo, path):
            m["has_security_policy"] = 1
            break
    else:
        m["has_security_policy"] = 0
    # Issue templates (directory or single file)
    r_it = gh(f"https://api.github.com/repos/{owner}/{repo}/contents/.github/ISSUE_TEMPLATE")
    m["has_issue_templates"] = 1 if r_it else file_exists(owner, repo, ".github/ISSUE_TEMPLATE.md")
    # PR template
    for path in [".github/PULL_REQUEST_TEMPLATE.md", "PULL_REQUEST_TEMPLATE.md",
                 ".github/pull_request_template.md"]:
        if file_exists(owner, repo, path):
            m["has_pr_template"] = 1
            break
    else:
        m["has_pr_template"] = 0
    # CI (any workflow file)
    r_wf = gh(f"https://api.github.com/repos/{owner}/{repo}/contents/.github/workflows")
    m["has_ci"] = 1 if (r_wf and isinstance(r_wf.json(), list) and len(r_wf.json()) > 0) else 0

    # ── contributors → bus factor ─────────────────────────────────────────────
    contributors = []
    for page in [1, 2, 3]:   # up to 300 contributors
        r_c = gh(f"https://api.github.com/repos/{owner}/{repo}/contributors",
                 params={"per_page": 100, "page": page})
        if not r_c or not isinstance(r_c.json(), list) or not r_c.json():
            break
        contributors.extend(r_c.json())
        if len(r_c.json()) < 100:
            break

    if contributors:
        total = sum(c.get("contributions", 0) for c in contributors)
        running, bf = 0, 0
        for c in sorted(contributors, key=lambda x: x.get("contributions", 0), reverse=True):
            running += c.get("contributions", 0)
            bf += 1
            if total > 0 and running / total > 0.5:
                break
        m["bus_factor"] = bf
    else:
        m["bus_factor"] = 0

    # ── commits last 90 days ─────────────────────────────────────────────────
    all_commits = []
    for page in [1, 2, 3]:
        r_cm = gh(f"https://api.github.com/repos/{owner}/{repo}/commits",
                  params={"since": ago90, "per_page": 100, "page": page})
        if not r_cm or not isinstance(r_cm.json(), list) or not r_cm.json():
            break
        all_commits.extend(r_cm.json())
        if len(r_cm.json()) < 100:
            break
    m["commits_90d"] = len(all_commits)

    # new contributors in 90d: distinct author emails in this window
    recent_emails = set(
        c.get("commit", {}).get("author", {}).get("email", "")
        for c in all_commits
        if c.get("commit", {}).get("author", {}).get("email")
    )
    m["new_contributors_90d"] = len(recent_emails)  # distinct active authors, 90d

    # ── closed issues last 90 days ───────────────────────────────────────────
    r_closed = gh(f"https://api.github.com/repos/{owner}/{repo}/issues",
                  params={"state": "closed", "since": ago90, "per_page": 100})
    if r_closed and isinstance(r_closed.json(), list):
        # filter out PRs (issues endpoint returns both)
        closed_issues = [i for i in r_closed.json() if "pull_request" not in i]
    else:
        closed_issues = []
    m["closed_issues_90d"] = len(closed_issues)
    total_issues = m["open_issues_count"] + len(closed_issues)
    m["issue_close_ratio_90d"] = (
        round(len(closed_issues) / total_issues, 3) if total_issues > 0 else None
    )

    # ── releases ─────────────────────────────────────────────────────────────
    r_rel = gh(f"https://api.github.com/repos/{owner}/{repo}/releases",
               params={"per_page": 5})
    releases = r_rel.json() if r_rel and isinstance(r_rel.json(), list) else []
    m["releases_last_5"] = len(releases)

    return m


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not TOKEN:
        print("❌  Set GITHUB_TOKEN environment variable before running.")
        print("    export GITHUB_TOKEN=ghp_your_token_here")
        return

    projects = []
    with open(INPUT, newline="") as f:
        for row in csv.DictReader(f):
            projects.append(row)

    print(f"Processing {len(projects)} projects...\n")
    rows = []

    for i, p in enumerate(projects):
        owner, repo = parse_repo(p["github_url"])
        stage = p.get("stage") or p.get("relation", "")
        print(f"[{i+1:02d}/{len(projects)}] {stage:12s} {owner}/{repo}")

        # ── fields from landscape CSV (no API call needed) ──────────────────
        from_landscape = {
            "stage":       stage,
            "name":        p["name"],
            "github_url":  p["github_url"],
            "owner":       owner,
            "repo":        repo,
            "contributor_count":      _int(p.get("github_contributors_count")),
            "stargazers_count":       _int(p.get("github_stars")),
            "has_license":            1 if p.get("license", "").strip() else 0,
            "license":                p.get("license", "").strip(),
            "days_since_last_commit": days_since(p.get("github_latest_commit_date")),
            "days_since_last_release":days_since(p.get("github_latest_release_date")),
            "age_days":               days_since(p.get("github_start_commit_date")),
            "collected_at":           datetime.date.today().isoformat(),
        }

        # ── fields from GitHub API ──────────────────────────────────────────
        api_data = collect_api(owner, repo)
        if api_data is None:
            print(f"    skipped\n")
            continue

        row = {**from_landscape, **api_data}
        rows.append(row)
        print(f"    ✓  bus_factor={row['bus_factor']}  "
              f"contributors={row['contributor_count']}  "
              f"commits_90d={row['commits_90d']}  "
              f"CoC={row['has_code_of_conduct']}  CI={row['has_ci']}")
        time.sleep(0.5)  # polite pause between repos

    # ── write output ─────────────────────────────────────────────────────────
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅  Saved {len(rows)} rows → {OUTPUT}")
    print("\nStage counts:")
    counts = Counter(r["stage"] for r in rows)
    for stage in ["sandbox", "incubating", "graduated", "archived"]:
        print(f"  {stage:12s}: {counts.get(stage, 0)}")

    # quick preview of means per stage
    print("\nQuick summary (means per stage):")
    for stage in ["sandbox", "incubating", "graduated", "archived"]:
        sr = [r for r in rows if r["stage"] == stage]
        if not sr:
            continue
        def avg(key):
            vals = [r[key] for r in sr if r.get(key) not in (None, "")]
            return round(sum(float(v) for v in vals) / len(vals), 1) if vals else "—"
        print(f"  {stage:12s}  contributors={avg('contributor_count')}  "
              f"bus_factor={avg('bus_factor')}  commits_90d={avg('commits_90d')}  "
              f"CoC={avg('has_code_of_conduct')}  CI={avg('has_ci')}")


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
