# Replication Package: Publish and Perish — Measuring Sustainability in Academic Open Source across Maturity Stages

Replication package for the FSE 2027 submission. This repository contains all code, data, and instructions needed to reproduce the results.

---

## Repository Structure

```
├── data/
│   ├── curioss_repos.csv          # 34,629 CURIOSS academic OSS repositories
│   ├── cncf_repos.csv             # 194 CNCF reference projects (Sandbox/Incubating/Graduated)
│   └── metrics/
│       ├── curioss_metrics.csv    # Computed metric values for CURIOSS repos
│       └── cncf_metrics.csv       # Computed metric values for CNCF repos
├── src/
│   ├── 01_filter_repos.py         # Dataset filtering pipeline (Section 3.1)
│   ├── 02_compute_metrics.py      # Metric computation via GitHub REST API (Section 3.2)
│   ├── 03_classify_maturity.py    # Quantile normalization + ordinal classifier (Section 3.3)
│   └── 04_analysis.py             # Comparative analysis (RQ1–RQ4)
├── notebooks/
│   └── figures.ipynb              # Reproduces all figures in the paper
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- A GitHub personal access token (for metric collection via the GitHub REST API)
- Install dependencies: `pip install -r requirements.txt`

Set your GitHub token as an environment variable before running:

```bash
export GITHUB_TOKEN=your_token_here
```

## Reproducing the Results

### Step 1 — Filter repositories

```bash
python src/01_filter_repos.py
```

Applies the filtering criteria (non-fork, non-mirror, affiliation confidence ≥ 0.7, active since July 2023, DEV-type, non-archived) to produce the 34,629-repository dataset.

### Step 2 — Compute metrics

```bash
python src/02_compute_metrics.py
```

Queries the GitHub REST API to compute the 40 sustainability metrics across 6 categories (Community, Activity, General, Development Practices, Documentation & Governance, Security) for all CURIOSS and CNCF repositories. **Note:** This step makes a large number of API calls and may take several hours depending on rate limits. Pre-computed metric values are available in `data/metrics/`.

### Step 3 — Maturity classification

```bash
python src/03_classify_maturity.py
```

Applies quantile normalization to the CNCF scale and trains the monotone ordinal gradient boosting classifier. Outputs maturity labels (Sandbox / Incubating / Graduated) for all CURIOSS repositories.

### Step 4 — Analysis and figures

```bash
python src/04_analysis.py
jupyter notebook notebooks/figures.ipynb
```

Reproduces the stage-stratified comparative analysis (RQ1–RQ4) and all figures in the paper.

## Data

Raw repository metadata was collected via the [Repofinder](PLACEHOLDER_REPOFINDER_CITE) pipeline. The 32 CURIOSS member institutions and their affiliated repositories are listed in `data/curioss_repos.csv`. CNCF project metadata was collected from the [CNCF landscape](https://landscape.cncf.io/) as of [DATE].

## License

[TO BE ADDED]
