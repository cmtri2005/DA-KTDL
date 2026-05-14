# Phase 6 - Execution Guide

## Overview

Phase 6 generates comprehensive reporting artifacts from the outputs of Phases 1-5. All previous phases are complete, and this phase focuses on data aggregation and visualization without retraining models.

**Status:** ✓ All prerequisites complete  
**Date:** May 14, 2026

---

## Quick Start (Choose One)

### Option A: Run Python Script Directly (Recommended)

```bash
cd d:\Baitapvenha\Khai thác dữ liệu\Do-an\DA-KTDL

# Run the main reporting pipeline
python phase6_reporting.py --output-dir reports/
```

**Expected output:** 15+ files in `reports/` including tables, figures, and markdown report  
**Runtime:** ~15 seconds

### Option B: Use Jupyter Notebook (Interactive)

```bash
# Open Jupyter and run the notebook
jupyter notebook Phase6_Reporting_Reproducibility.ipynb
```

**Advantages:**

- See outputs cell-by-cell
- Inspect intermediate data
- Modify visualizations interactively
- Export plots individually

**Steps:**

1. Cell 1-3: Load and validate outputs
2. Cell 4-6: Build Table 1
3. Cell 7-9: Build Table 2
4. Cell 10-11: Error analysis
5. Cell 12: Cluster analysis
6. Cell 13-17: Generate 4 visualizations
7. Cell 18: Create comprehensive markdown report
8. Cell 19-20: Export reproducibility summary

### Option C: Use PowerShell Script (Windows)

```powershell
# Run smoke test with validation
.\run_smoke.ps1
```

**Validates:**

- Python installation
- Phase 3 outputs exist
- Phase 5 outputs exist
- Runs reporting pipeline
- Reports success/failure

### Option D: Use Makefile (Windows/Mac/Linux)

```bash
# View all available targets
make help

# Run just Phase 6 reporting
make phase6

# Run complete pipeline (all phases)
make all

# Check current status
make status
```

---

## Expected Outputs

### CSV Files (Tabular Data)

| File                              | Rows | Columns | Purpose                                    |
| --------------------------------- | ---- | ------- | ------------------------------------------ |
| `results_table1.csv`              | 900+ | 8       | All clustering experiments                 |
| `results_table1_summary.csv`      | 16   | 10      | Summary statistics by representation+model |
| `results_table2.csv`              | 16   | 10      | All classification configurations          |
| `results_table2_best_by_pair.csv` | 16   | 10      | Best config per representation pair        |
| `confusion_matrix_summary.csv`    | 16   | 7       | Confusion matrix metrics                   |
| `top_errors.csv`                  | 10   | 7       | Worst-performing configurations            |

### Markdown Files

| File                     | Content                         | Use Case                              |
| ------------------------ | ------------------------------- | ------------------------------------- |
| `results_table1.md`      | Formatted Table 1               | Blog posts, reports                   |
| `results_table2.md`      | Formatted Table 2               | Blog posts, reports                   |
| `reproduction_report.md` | 13-section comprehensive report | Supplementary material, documentation |

### JSON Files

| File                            | Content                    |
| ------------------------------- | -------------------------- |
| `cluster_analysis_summary.json` | Cluster purity metrics     |
| `environment_summary.json`      | Full project configuration |

### PNG Figures (300 DPI, Publication Quality)

```
figures/
├── macro_f1_by_model.png                    # Model performance comparison
├── clustering_metrics_comparison.png        # ARI/NMI/Silhouette by representation
├── accuracy_heatmap.png                     # Classification accuracy matrix
└── representation_performance.png           # Average metrics by representation
```

### Reproducibility Scripts

| File            | Purpose                         |
| --------------- | ------------------------------- |
| `run_smoke.ps1` | PowerShell smoke test           |
| `Makefile`      | Build automation (make targets) |

---

## Detailed Instructions

### Step 1: Verify Prerequisites

Before running Phase 6, ensure:

1. **Python 3.8+** installed

   ```bash
   python --version
   ```

2. **Phase 3 outputs exist**

   ```bash
   ls outputs/outputs/phase3_clustering/results_table.csv
   ```

3. **Phase 5 outputs exist**

   ```bash
   ls outputs/outputs/da-ktdl-phase5-table2/results_table_all_runs.csv
   ```

4. **Dependencies installed**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

### Step 2: Run Phase 6

#### Using Python Script:

```bash
cd d:\Baitapvenha\Khai thác dữ liệu\Do-an\DA-KTDL
python phase6_reporting.py --output-dir reports/
```

#### Using Jupyter Notebook:

```bash
jupyter notebook Phase6_Reporting_Reproducibility.ipynb
# Run all cells in order
```

#### Using PowerShell:

```powershell
.\run_smoke.ps1
```

### Step 3: Verify Outputs

Check that all files were created:

```bash
# Verify CSV files
ls reports/*.csv

# Verify markdown files
ls reports/*.md

# Verify figures
ls reports/figures/*.png

# Show summary
ls -lh reports/ | head -15
```

### Step 4: Review Results

#### Check Table 1 (Clustering)

```bash
# View top 10 clustering configurations by ARI
head -11 reports/results_table1.csv | column -t -s,
```

#### Check Table 2 (Classification)

```bash
# View best classification configuration
head -2 reports/results_table2.csv | column -t -s,
```

#### Read Final Report

```bash
# View markdown report (in text editor or browser)
cat reports/reproduction_report.md | less
# or
start reports/reproduction_report.md  # Windows
open reports/reproduction_report.md   # Mac
```

---

## Key Metrics Explained

### Table 1: Clustering Metrics

- **ARI (Adjusted Rand Index)**:
  - Range: -1 to 1
  - 1.0 = perfect clustering
  - 0.47 expected (from paper)

- **NMI (Normalized Mutual Information)**:
  - Range: 0 to 1
  - 1.0 = perfect clustering
  - 0.55 expected (from paper)

- **Silhouette**:
  - Range: -1 to 1
  - Positive = well-separated clusters
  - Higher is better

- **Noise Fraction**:
  - Fraction of noise points (HDBSCAN only)
  - 0.0 = no noise points
  - Lower is better

### Table 2: Classification Metrics

- **Accuracy**: Overall correctness (0-1)
- **Macro F1**: Unweighted average across classes
- **Weighted F1**: Weighted by class frequency
- **MCC**: Matthews Correlation Coefficient (-1 to 1)
- **Cohen Kappa**: Inter-rater agreement (-1 to 1)
- **Top-3 Accuracy**: Correct if predicted in top 3
- **ROC-AUC**: Macro-averaged one-vs-rest

**Paper expected:** Accuracy 0.926, Macro F1 0.925

---

## Customization Options

### Change Output Directory

```bash
python phase6_reporting.py --output-dir /path/to/custom/reports/
```

### Modify Paper Comparison Values

Edit `phase6_reporting.py` line 38-43:

```python
PAPER_EXPECTED = {
    "best_accuracy": 0.926,        # Modify these values
    "best_macro_f1": 0.925,        # to match your paper
    "clustering_ari": 0.47,
    "clustering_nmi": 0.55,
}
```

### Add Custom Visualizations

Extend `create_visualizations()` function in `phase6_reporting.py`:

```python
def create_visualizations(table1, table2, output_dir):
    # ... existing plots ...

    # Add your custom plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # ... plot code ...
    fig.savefig(output_dir / "figures/my_custom_plot.png", dpi=300)
```

---

## Troubleshooting

### Issue: "Table 1 source not found"

**Solution:** Run Phase 3 first

```bash
python -m clustering --output-dir outputs/phase3_clustering/
```

### Issue: "Table 2 source not found"

**Solution:** Run Phase 5 first (typically on Kaggle)

### Issue: "Module not found" errors

**Solution:** Install dependencies

```bash
pip install -r requirements.txt
```

### Issue: Permission denied (run_smoke.ps1)

**Solution:** Set execution policy (Windows)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Out of memory creating figures

**Solution:** Run Jupyter notebook cells individually to control memory

### Issue: Unicode/encoding errors

**Solution:** Ensure UTF-8 encoding

```bash
$ENV:PYTHONIOENCODING = "utf-8"
```

---

## Quality Assurance Checklist

After running Phase 6, verify:

- [ ] `results_table1.csv` has 900+ rows (clustering experiments)
- [ ] `results_table2.csv` has 16 rows (classification configs)
- [ ] Best accuracy in Table 2 > 0.80 (sanity check)
- [ ] Best ARI in Table 1 > 0.40 (sanity check)
- [ ] 4 PNG figures created at 300 DPI
- [ ] `reproduction_report.md` contains 13 sections
- [ ] `environment_summary.json` valid JSON
- [ ] No empty CSV files (would indicate loading error)

---

## File Structure After Phase 6

```
DA-KTDL/
├── reports/
│   ├── results_table1.csv               ✓
│   ├── results_table1.md                ✓
│   ├── results_table1_summary.csv       ✓
│   ├── results_table2.csv               ✓
│   ├── results_table2.md                ✓
│   ├── results_table2_best_by_pair.csv  ✓
│   ├── confusion_matrix_summary.csv     ✓
│   ├── top_errors.csv                   ✓
│   ├── cluster_analysis_summary.json    ✓
│   ├── reproduction_report.md           ✓
│   ├── environment_summary.json         ✓
│   └── figures/
│       ├── macro_f1_by_model.png        ✓
│       ├── clustering_metrics_comparison.png ✓
│       ├── accuracy_heatmap.png         ✓
│       └── representation_performance.png    ✓
├── phase6_reporting.py                  ✓
├── Phase6_Reporting_Reproducibility.ipynb ✓
├── run_smoke.ps1                        ✓
├── Makefile                             ✓
└── PHASE6_README.md                     ✓
```

---

## Next Steps

### For Publication

1. Use `reproduction_report.md` as supplementary material
2. Include figures from `reports/figures/` in paper appendix
3. Reference `results_table1.csv` and `results_table2.csv` for exact metrics
4. Add `environment_summary.json` to GitHub for reproducibility

### For Continued Research (Phase 7-8)

- **Phase 7:** Triple-quality-aware classification
- **Phase 8:** Graph-aware document representations

### For Sharing

1. Commit to GitHub:

   ```bash
   git add reports/ PHASE6_README.md phase6_reporting.py
   git commit -m "Phase 6: Comprehensive reporting and reproducibility"
   ```

2. Create reproducibility package:
   ```bash
   tar czf DA-KTDL-phase6-artifacts.tar.gz reports/ phase6_reporting.py PHASE6_README.md
   ```

---

## Questions & Support

For issues or questions:

1. Check `PHASE6_README.md` for additional documentation
2. Review logs in `phase6_reporting.py` output
3. Inspect intermediate DataFrames in Jupyter notebook
4. Verify file paths are relative to project root

---

## Execution Timeline

**Typical Phase 6 Runtime:**

- Table 1 aggregation: <1 second
- Table 2 aggregation: <1 second
- Visualizations (4 plots): <10 seconds
- Report generation: <2 seconds
- **Total: <15 seconds** (very fast!)

---

**Phase 6 Status:** ✓ Ready for execution

**Last Updated:** May 14, 2026
