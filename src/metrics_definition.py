"""
metrics_definition.py
---------------------
Single source of truth for the 43 sustainability metrics used in the paper.
Column names match those in curioss_metrics.csv and cncf_sustainability_metrics.csv.

CNCF note: some columns use slightly shorter names in cncf_sustainability_metrics.csv
(e.g. 'issue_response_time' instead of 'issue_response_time_90d'); the CNCF_COL_MAP
dictionary below documents these aliases so analysis scripts can normalise on load.
"""

# ── Metric catalogue ──────────────────────────────────────────────────────────
# Each entry: name, category, higher_is_better (True/False/None for binary)
METRICS = [
    # Community
    {"name": "unique_contributors",          "category": "Community",                "higher_is": True},
    {"name": "contributor_count_90d",        "category": "Community",                "higher_is": True},
    {"name": "new_contributors_90d",         "category": "Community",                "higher_is": True},
    {"name": "organizational_diversity_90d", "category": "Community",                "higher_is": True},
    {"name": "bus_factor",                   "category": "Community",                "higher_is": True},
    {"name": "contributor_retention_90d_pct","category": "Community",                "higher_is": True},
    {"name": "contributor_growth_90d_pct",   "category": "Community",                "higher_is": True},
    # Activity
    {"name": "commit_frequency_90d",         "category": "Activity",                 "higher_is": True},
    {"name": "social_activity_90d",          "category": "Activity",                 "higher_is": True},
    {"name": "commit_growth_90d_pct",        "category": "Activity",                 "higher_is": True},
    {"name": "issue_response_time_90d",      "category": "Activity",                 "higher_is": False},
    {"name": "pr_response_time_90d",         "category": "Activity",                 "higher_is": False},
    {"name": "pr_merge_time_90d",            "category": "Activity",                 "higher_is": False},
    {"name": "issue_closed_ratio_90d",       "category": "Activity",                 "higher_is": True},
    {"name": "release_frequency_90d",        "category": "Activity",                 "higher_is": True},
    {"name": "maintainer_activity_recency_days", "category": "Activity",             "higher_is": False},
    # General
    {"name": "project_age_days",             "category": "General",                  "higher_is": True},
    {"name": "community_interest",           "category": "General",                  "higher_is": True},
    {"name": "downstream_dependents",        "category": "General",                  "higher_is": True},
    # Development Practices
    {"name": "ci_cd",                        "category": "Development Practices",    "higher_is": True},
    {"name": "build_environment",            "category": "Development Practices",    "higher_is": True},
    {"name": "test_presence",                "category": "Development Practices",    "higher_is": True},
    {"name": "platform_os_support",          "category": "Development Practices",    "higher_is": True},
    {"name": "contributors_dev_activity_90d","category": "Development Practices",    "higher_is": True},
    {"name": "maintainers_dev_activity_90d", "category": "Development Practices",    "higher_is": True},
    {"name": "code_review_practices_90d",    "category": "Development Practices",    "higher_is": True},
    # Documentation & Governance
    {"name": "readme",                       "category": "Documentation & Governance","higher_is": True},
    {"name": "contributing",                 "category": "Documentation & Governance","higher_is": True},
    {"name": "code_of_conduct",              "category": "Documentation & Governance","higher_is": True},
    {"name": "governance",                   "category": "Documentation & Governance","higher_is": True},
    {"name": "issue_template",               "category": "Documentation & Governance","higher_is": True},
    {"name": "pr_template",                  "category": "Documentation & Governance","higher_is": True},
    {"name": "code_review_policy",           "category": "Documentation & Governance","higher_is": True},
    {"name": "changelog",                    "category": "Documentation & Governance","higher_is": True},
    {"name": "documentation_completeness",   "category": "Documentation & Governance","higher_is": True},
    {"name": "financial_support",            "category": "Documentation & Governance","higher_is": True},
    # Security
    {"name": "license",                      "category": "Security",                 "higher_is": True},
    {"name": "security_policy",              "category": "Security",                 "higher_is": True},
    {"name": "branch_protection",            "category": "Security",                 "higher_is": True},
    {"name": "signed_releases",              "category": "Security",                 "higher_is": True},
    {"name": "dependency_updates",           "category": "Security",                 "higher_is": True},
    {"name": "vulnerability_count",          "category": "Security",                 "higher_is": False},
    {"name": "trusted_publishing",           "category": "Security",                 "higher_is": True},
]

METRIC_NAMES = [m["name"] for m in METRICS]
CATEGORIES   = list(dict.fromkeys(m["category"] for m in METRICS))  # preserve order


def metrics_by_category():
    """Return {category: [metric_name, ...]} mapping."""
    from collections import defaultdict
    d = defaultdict(list)
    for m in METRICS:
        d[m["category"]].append(m["name"])
    return dict(d)


def monotone_constraints():
    """
    Return {metric_name: direction} for LightGBM monotone_constraints.
    +1 = higher value → higher maturity stage
    -1 = lower value  → higher maturity stage
     0 = no constraint
    """
    mapping = {True: 1, False: -1, None: 0}
    return {m["name"]: mapping[m["higher_is"]] for m in METRICS}


# ── Column-name aliases: CNCF CSV → CURIOSS CSV ──────────────────────────────
# The CNCF metrics CSV (cncf_sustainability_metrics*.csv) uses slightly
# different column names for some metrics. This mapping lets analysis scripts
# rename CNCF columns to match the canonical CURIOSS names above.
CNCF_COL_MAP = {
    "issue_response_time":   "issue_response_time_90d",
    "pr_response_time":      "pr_response_time_90d",
    "pr_merge_time":         "pr_merge_time_90d",
    "issue_closed_ratio":    "issue_closed_ratio_90d",
    "code_review_practices": "code_review_practices_90d",
    "known_vulnerabilities": "vulnerability_count",
    "code_review_enforced":  "code_review_policy",   # nearest equivalent
}
