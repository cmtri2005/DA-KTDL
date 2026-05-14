# Phase 6: Reporting and Reproducibility - Setup & Execution Guide

## Overview

Phase 6 aggregates outputs from Phases 1-5 and generates comprehensive reporting artifacts:

- **Table 1**: Clustering evaluation results
- **Table 2**: Classification evaluation results
- **Confusion Matrix Analysis**: Model performance metrics
- **Error Analysis**: Lowest-performing configurations
- **Cluster Analysis**: Cluster purity and composition
- **Visualizations**: Performance comparison charts
- **Final Markdown Report**: Comprehensive reproduction summary
- **Reproducibility Scripts**: Smoke test and build automation

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify key packages
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('✓ All packages installed')"
```

## Quick Start

### Option 1: Direct Python Script

```bash
# Run the main reporting pipeline
python phase6_reporting.py --output-dir reports/

# Expected output:
# ✓ results_table1.csv
# ✓ results_table2.csv
# ✓ confusion_matrix_summary.csv
# ✓ top_errors.csv
# ✓ reproduction_report.md
# ✓ figures/*.png
```

### Option 2: Using Makefile

```bash
# View available targets
make help

# Run Phase 6 reporting
make phase6

# Run all phases
make all

# Quick smoke test
make smoke
```

### Option 3: PowerShell Smoke Test

```powershell
.\run_smoke.ps1
```

## Output Structure

```
reports/
├── results_table1.csv              # Clustering results (all experiments)
├── results_table1.md               # Clustering results (markdown format)
├── results_table1_summary.csv      # Clustering summary statistics
├── results_table2.csv              # Classification results (all experiments)
├── results_table2.md               # Classification results (markdown)
├── results_table2_best_by_pair.csv # Best classification by representation pair
├── confusion_matrix_summary.csv    # Confusion matrix metrics
├── top_errors.csv                  # Lowest-performing configurations
├── cluster_analysis_summary.json   # Cluster purity analysis
├── reproduction_report.md          # Comprehensive final report
├── environment_summary.json        # Project configuration and parameters
└── figures/
    ├── macro_f1_by_model.png               # Model performance comparison
    ├── clustering_metrics_comparison.png   # ARI/NMI by representation
    ├── accuracy_heatmap.png                # Accuracy matrix heatmap
    └── representation_performance.png      # Average accuracy by representation
```

## Key Metrics and Comparisons

### Table 1: Clustering Results

| Metric         | Description                                        |
| -------------- | -------------------------------------------------- |
| ARI            | Adjusted Rand Index (0-1, higher better)           |
| NMI            | Normalized Mutual Information (0-1, higher better) |
| Silhouette     | Silhouette Coefficient (-1 to 1, higher better)    |
| Noise_Fraction | Fraction of noise points (HDBSCAN)                 |

**Expected Performance** (from paper):

- Best ARI: ~0.47
- Best NMI: ~0.55
- Best representation: Hybrid or Abstract+Triples

### Table 2: Classification Results

| Metric        | Description                                           |
| ------------- | ----------------------------------------------------- |
| Accuracy      | Overall accuracy (0-1)                                |
| Macro_F1      | Unweighted average F1 across classes                  |
| Weighted_F1   | Weighted average F1 (by class frequency)              |
| MCC           | Matthews Correlation Coefficient                      |
| Cohen_Kappa   | Inter-rater agreement                                 |
| Top3_Accuracy | Top-3 accuracy (predicted class in top 3 predictions) |
| ROC_AUC_OvR   | Macro-averaged ROC-AUC (One-vs-Rest)                  |

**Expected Performance** (from paper):

- Best Accuracy: 92.6%
- Best Macro-F1: 0.925
- Best configuration: Hybrid/Abstract with SciBERT or SPECTER

## Input Data Dependencies

Phase 6 reads from:

1. **Phase 3 Outputs**: `outputs/outputs/phase3_clustering/`
   - `results_table.csv` - Clustering experiment results
   - Cluster assignments and metrics

2. **Phase 4 Outputs**: `outputs/outputs/phase4_cluster_propagation/`
   - Propagated cluster labels for classification set
   - Cluster-to-document mappings

3. **Phase 5 Outputs**: `outputs/outputs/da-ktdl-phase5-table2/`
   - `results_table_all_runs.csv` - All classification experiment runs
   - `results_table_best_by_pair.csv` - Best configuration per representation pair
   - Classification model checkpoints and predictions

## Customization

### Modify output directory:

```bash
python phase6_reporting.py --output-dir /path/to/custom/reports/
```

### Modify comparison metrics:

Edit `PAPER_EXPECTED` dictionary in `phase6_reporting.py`:

```python
PAPER_EXPECTED = {
    "best_accuracy": 0.926,        # Change if using different paper
    "best_macro_f1": 0.925,
    "clustering_ari": 0.47,
    "clustering_nmi": 0.55,
}
```

### Add custom visualizations:

Extend the `create_visualizations()` function in `phase6_reporting.py` to add plots.

## Troubleshooting

### Issue: "Table 1 source not found"

**Solution:** Run Phase 3 first:

```bash
python -m clustering --output-dir outputs/phase3_clustering/
```

### Issue: "Table 2 source not found"

**Solution:** Run Phase 5 first (typically on Kaggle notebook)

### Issue: Missing pandas/numpy

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
```

### Issue: Permission denied (run_smoke.ps1)

**Solution:** Set execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Pipeline Architecture

```
Phase 1: Data Preparation
    ↓ (5K + 10K documents with 4 representations)
Phase 2: Embedding Generation
    ↓ (4 representations × 4 models = 16 embedding sets)
Phase 3: Clustering
    ↓ (KMeans, GMM, HDBSCAN × 16 embeddings = 100+ configs)
Phase 4: Cluster Propagation
    ↓ (Best cluster configs propagated to classification set)
Phase 5: Classification
    ↓ (16 representation pairs × 3 models = 48+ experiments)
Phase 6: Reporting ✓
    ↓ (Aggregate & analyze results)
Deliverables: Tables, Confusion Matrices, Report, Plots
```

## Expected Runtime

- **Table 1 aggregation**: < 1 second
- **Table 2 aggregation**: < 1 second
- **Visualizations (4 plots)**: < 10 seconds
- **Report generation**: < 2 seconds
- **Total Phase 6**: < 15 seconds

## Quality Assurance

After running Phase 6, verify:

1. ✓ All CSV files generated with data

   ```bash
   ls -lh reports/*.csv
   ```

2. ✓ Markdown reports readable

   ```bash
   head -50 reports/reproduction_report.md
   ```

3. ✓ Figures generated

   ```bash
   ls reports/figures/
   ```

4. ✓ JSON configs created
   ```bash
   python -c "import json; print(json.load(open('reports/environment_summary.json')))"
   ```

## Next Steps

### For Publication:

- Use `reproduction_report.md` as basis for paper supplementary material
- Include figures from `reports/figures/` in paper
- Reference `results_table1.csv` and `results_table2.csv` for exact metrics

### For Further Research (Phase 7-8):

- **Phase 7**: Triple-quality-aware hybrid classification
- **Phase 8**: Graph-aware document representation

## References

- Paper: "Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents"
- Code: This repository
- Dataset: arXiv metadata snapshot (Kaggle)

---

**Phase 6 Status**: ✓ Ready for execution

**Last Updated**: May 14, 2026
