#!/usr/bin/env python3
"""
collect_curioss.py
==================
Collects raw GitHub sustainability data for all CURIOSS repos.

Reads repos from the per-university SQLite databases in Data/db/,
applies the CURIOSS corpus filters, and stores everything in
curioss_temp/curioss_raw.db using the same schema as cncf/collect_raw_data.py.

Token is read from githubtokens.txt (MINE: entry), same as the CNCF script.

Run (full corpus):
    python3 curioss_temp/collect_curioss.py

Run (missing repos only, skips iterating all 35k):
    python3 curioss_temp/collect_curioss.py --from-csv curioss_temp/missing_repos.csv
"""

import sys, argparse
from pathlib import Path

# ── Import shared collection logic from cncf/collect_raw_data.py ──────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "cncf"))

import collect_raw_data as crd   # noqa: E402 — must come after sys.path insert

import sqlite3, glob, os, csv, logging

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_DIR     = REPO_ROOT / "Data" / "db"
OUT_DB     = Path(__file__).parent / "curioss_raw.db"
DAYS       = 1825   # 5-year window
SOURCE     = "curioss"

# Lero has multiple DB variants; keep only the canonical one
SKIP_SUFFIX = ("previous", "recovered", "recovered2")

# CURIOSS corpus filters (same as used to produce the 35k repo corpus)
FILTER_SQL = """
    SELECT full_name, html_url
    FROM   repositories
    WHERE  fork = 0
      AND  mirror_url IS NULL
      AND  archived   = 0
      AND  size       > 0
      AND  pushed_at  >= '2023-07-27'
      AND  affiliation_prediction_gpt_5_mini >= 0.7
      AND  type_prediction_gpt_5_mini = 'DEV'
    ORDER BY full_name
"""


def university_acronym(db_path):
    """Extract acronym from filename, e.g. repository_data_UCSC_database.db → UCSC."""
    stem = Path(db_path).stem                         # repository_data_UCSC_database
    stem = stem.replace("repository_data_", "")      # UCSC_database
    stem = stem.replace("_database", "")              # UCSC
    return stem


def iter_curioss_repos():
    """Yield (full_name, html_url, university) for every filtered repo."""
    pattern = str(DB_DIR / "repository_data_*_database.db")
    db_files = sorted(glob.glob(pattern))

    for db_path in db_files:
        acr = university_acronym(db_path)
        if any(db_path.endswith(s + ".db") for s in SKIP_SUFFIX):
            log.debug("Skipping variant DB: %s", db_path)
            continue

        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(FILTER_SQL).fetchall()
        except Exception as e:
            log.warning("Could not query %s: %s", db_path, e)
            rows = []
        conn.close()

        for full_name, html_url in rows:
            yield full_name, html_url, acr


def iter_from_csv(csv_path):
    """Yield (full_name, html_url, university) from a missing_repos.csv file."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row["full_name"], row["html_url"], row["university"]


def already_collected(db, full_name):
    row = db.execute(
        "SELECT id FROM repositories WHERE full_name=?", (full_name,)
    ).fetchone()
    return row is not None


def main():
    parser = argparse.ArgumentParser(description="Collect CURIOSS repo data")
    parser.add_argument(
        "--from-csv",
        metavar="CSV",
        help="Path to missing_repos.csv — collect only those repos instead of iterating all DBs",
    )
    args = parser.parse_args()

    if not crd.TOKEN:
        log.error("No token found under MINE: in githubtokens.txt")
        sys.exit(1)

    # Set up output DB with shared schema
    out_db = sqlite3.connect(OUT_DB)
    out_db.executescript(crd.SCHEMA)
    log.info("Output database: %s", OUT_DB)

    if args.from_csv:
        csv_path = Path(args.from_csv)
        log.info("Loading repos from CSV: %s", csv_path)
        repos = list(iter_from_csv(csv_path))
        log.info("Repos to collect from CSV: %d", len(repos))
    else:
        repos = list(iter_curioss_repos())
        log.info("Total repos to collect: %d", len(repos))

    for i, (full_name, html_url, university) in enumerate(repos, 1):
        if already_collected(out_db, full_name):
            log.info("[%d/%d] ↷ skip %s", i, len(repos), full_name)
            continue

        log.info("[%d/%d] %s (%s)", i, len(repos), full_name, university)

        # Inject university as stage field so it's recorded in repositories.cncf_stage
        owner, repo = full_name.split("/", 1)
        try:
            repo_id = crd.collect_repo(out_db, owner, repo,
                                       stage=university, source=SOURCE)
            if not repo_id:
                continue
            crd.collect_contributors(out_db, owner, repo, repo_id)
            crd.collect_commits(out_db, owner, repo, repo_id, days=DAYS)
            crd.collect_issues(out_db, owner, repo, repo_id, days=DAYS)
            crd.collect_pull_requests(out_db, owner, repo, repo_id, days=DAYS)
            has_signed = crd.collect_releases(out_db, owner, repo, repo_id)
            crd.collect_files_and_security(out_db, owner, repo, repo_id, has_signed)
            crd.collect_dependents(out_db, owner, repo, repo_id)
            log.info("  ✓ DONE %s", full_name)
        except Exception as e:
            log.exception("  ✗ FAILED %s — %s", full_name, e)

    out_db.close()
    log.info("Collection complete → %s", OUT_DB)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
