"""
01_filter_repos.py
------------------
Applies the dataset filtering criteria described in Section 3.1 of the paper
to the raw Repofinder output, producing the final set of 34,629 academic OSS
repositories from 32 CURIOSS institutions.

Input:  data/repofinder_raw.csv  (raw Repofinder output — not included in repo
                                   due to size; available from the Repofinder
                                   pipeline: https://github.com/PLACEHOLDER)
Output: data/curioss_repos.csv

Filtering criteria
------------------
  1. Non-fork, non-mirror  (fork == False, mirror_url is null)
  2. Affiliation confidence >= 0.7
  3. Active since July 2023  (pushed_at >= 2023-07-01)
  4. Repository type == 'DEV'
  5. Not archived, not empty  (archived == False, size > 0)
"""

import argparse
import sys
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.7
ACTIVE_SINCE = pd.Timestamp("2023-07-01", tz="UTC")

REQUIRED_COLUMNS = [
    "repo_id", "full_name", "institution", "affiliation_confidence",
    "fork", "mirror_url", "archived", "size", "pushed_at",
    "repo_type", "default_branch", "language", "html_url",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in raw file: {missing}", file=sys.stderr)
        sys.exit(1)
    df["pushed_at"] = pd.to_datetime(df["pushed_at"], utc=True, errors="coerce")
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    print(f"Starting with {n0:,} repositories")

    # 1. Non-fork, non-mirror
    df = df[~df["fork"].astype(bool)]
    df = df[df["mirror_url"].isna()]
    print(f"  After non-fork / non-mirror:       {len(df):,}  (-{n0 - len(df):,})")
    n1 = len(df)

    # 2. Affiliation confidence
    df = df[df["affiliation_confidence"] >= CONFIDENCE_THRESHOLD]
    print(f"  After confidence >= {CONFIDENCE_THRESHOLD}:            {len(df):,}  (-{n1 - len(df):,})")
    n2 = len(df)

    # 3. Active since July 2023
    df = df[df["pushed_at"] >= ACTIVE_SINCE]
    print(f"  After active since {ACTIVE_SINCE.date()}:   {len(df):,}  (-{n2 - len(df):,})")
    n3 = len(df)

    # 4. DEV-type only
    df = df[df["repo_type"].str.upper() == "DEV"]
    print(f"  After DEV-type filter:             {len(df):,}  (-{n3 - len(df):,})")
    n4 = len(df)

    # 5. Not archived, not empty
    df = df[~df["archived"].astype(bool)]
    df = df[df["size"] > 0]
    print(f"  After non-archived / non-empty:    {len(df):,}  (-{n4 - len(df):,})")

    print(f"\nFinal dataset: {len(df):,} repositories")
    return df.reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter raw Repofinder output")
    parser.add_argument(
        "--input", default="data/repofinder_raw.csv",
        help="Path to the raw Repofinder CSV (default: data/repofinder_raw.csv)"
    )
    parser.add_argument(
        "--output", default="data/curioss_repos.csv",
        help="Output path for filtered repos (default: data/curioss_repos.csv)"
    )
    args = parser.parse_args()

    df_raw = load_raw(args.input)
    df_filtered = apply_filters(df_raw)

    output_cols = [
        "repo_id", "full_name", "institution", "affiliation_confidence",
        "default_branch", "language", "html_url", "pushed_at",
        "star_count", "fork_count", "open_issues_count", "size",
    ]
    # keep only columns that exist
    output_cols = [c for c in output_cols if c in df_filtered.columns]
    df_filtered[output_cols].to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
