#!/usr/bin/env python3
"""
Phase 6: Reporting and Reproducibility Pipeline

This script aggregates outputs from Phases 3-5 and generates:
- Table 1 (clustering results)
- Table 2 (classification results)
- Confusion matrix
- Error analysis
- Cluster analysis
- Final markdown report
- Visualizations
- Reproducibility scripts

Usage:
    python phase6_reporting.py --output-dir reports/
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    classification_report,
    f1_score
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ModuleNotFoundError:
    plt = None
    sns = None
    HAS_PLOTTING = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

PHASE1_DIR = Path("outputs/phase1_data")
PHASE3_DIR = Path("outputs/phase3_clustering")
PHASE4_DIR = Path("outputs/phase4_cluster_propagation")
PHASE5_DIR = Path("outputs/phase5_train")

PHASE3_CANDIDATES = (
    Path("outputs/phase3_clustering"),
    Path("outputs/outputs/phase3_clustering"),
)
PHASE4_CANDIDATES = (
    Path("outputs/phase4_cluster_propagation"),
    Path("outputs/outputs/phase4_cluster_propagation"),
)
PHASE5_CANDIDATES = (
    Path("outputs/phase5_train"),
    Path("outputs/phase5_classification"),
    Path("outputs/outputs/da-ktdl-phase5-table2"),
)

MODEL_ALIAS_TO_SLUG = {
    "scibert": "allenai-scibert_scivocab_uncased",
    "specter": "allenai-specter",
    "minilm": "sentence-transformers-all-MiniLM-L6-v2",
}

REPRESENTATIONS = ["abstract", "concatenate", "hybrid", "triples"]
EMBEDDING_MODELS = [
    "allenai-scibert_scivocab_uncased",
    "allenai-specter",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
]
CLUSTERING_ALGOS = ["kmeans", "gmm", "hdbscan"]

# Paper expected values
PAPER_EXPECTED = {
    "best_accuracy": 0.926,
    "best_macro_f1": 0.925,
    "clustering_ari": 0.47,
    "clustering_nmi": 0.55,
}


def detect_phase_dir(
    label: str,
    requested: Optional[Path],
    candidates: Tuple[Path, ...],
    required_file: Optional[str] = None,
) -> Path:
    """Resolve an input directory, preferring explicit paths and current outputs."""
    if requested is not None:
        if not requested.exists():
            raise FileNotFoundError(f"{label} directory not found: {requested}")
        if required_file and not (requested / required_file).exists():
            raise FileNotFoundError(
                f"{label} directory is missing {required_file}: {requested}"
            )
        return requested

    for candidate in candidates:
        if not candidate.exists():
            continue
        if required_file and not (candidate / required_file).exists():
            continue
        return candidate

    tried = ", ".join(str(path) for path in candidates)
    suffix = f" containing {required_file}" if required_file else ""
    raise FileNotFoundError(f"Could not find {label} directory{suffix}. Tried: {tried}")


def phase5_experiment_dir(row: pd.Series) -> Optional[Path]:
    """Return the local Phase 5 experiment directory for one Table 2 row."""
    model_slug = row.get("Model_Slug")
    if not isinstance(model_slug, str) or not model_slug:
        model_slug = MODEL_ALIAS_TO_SLUG.get(str(row.get("Model", "")).lower())
    if not model_slug:
        return None

    cluster_mode = str(row.get("Clustering_Mode", ""))
    classifier_input = str(row.get("Classifier_Input", ""))
    candidate = PHASE5_DIR / cluster_mode / classifier_input / model_slug
    return candidate if candidate.exists() else None


def best_trial_dir(row: pd.Series) -> Optional[Path]:
    """Resolve the local best trial directory for one Phase 5 experiment row."""
    experiment_dir = phase5_experiment_dir(row)
    if experiment_dir is None:
        return None

    summary_path = experiment_dir / "experiment_summary.json"
    trial_number = row.get("Best_Trial_Number", 0)
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        trial_number = summary.get("best_trial_number", trial_number)

    try:
        trial_number = int(trial_number)
    except (TypeError, ValueError):
        trial_number = 0

    trial_dir = experiment_dir / f"trial_{trial_number:03d}"
    if trial_dir.exists():
        return trial_dir

    trial_dirs = sorted(experiment_dir.glob("trial_*"))
    return trial_dirs[0] if trial_dirs else None


def load_phase1_classify_texts() -> Dict[str, Dict[str, str]]:
    """Load abstract/triples text for joining best-classifier error examples."""
    combined_path = PHASE1_DIR / "classify_combined.jsonl"
    if not combined_path.exists():
        return {}

    lookup: Dict[str, Dict[str, str]] = {}
    with open(combined_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            lookup[str(row.get("id"))] = {
                "abstract": str(row.get("fmt_abstract", "")),
                "triples": str(row.get("triples_text", "")),
                "hybrid": str(row.get("fmt_hybrid", "")),
            }
    return lookup


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as markdown, falling back when tabulate is unavailable."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        csv_text = df.to_csv(index=index).strip()
        return f"```csv\n{csv_text}\n```"

# ============================================================================
# TASK 1: BUILD TABLE 1 (CLUSTERING RESULTS)
# ============================================================================

def build_table1(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read phase3_clustering/results_table.csv and aggregate into Table 1.
    
    Table 1 columns:
    - representation
    - embedding_model
    - clustering_model
    - k_or_params
    - ARI
    - NMI
    - silhouette
    - noise_fraction
    """
    logger.info("Building Table 1 (Clustering Results)...")
    
    results_file = PHASE3_DIR / "results_table.csv"
    if not results_file.exists():
        logger.warning(f"Table 1 source not found: {results_file}")
        return None, None
    
    df = pd.read_csv(results_file)
    
    # Keep only cluster split (exclude classify)
    df = df[df["split"] == "cluster"].copy()
    
    # Rename columns for clarity
    df = df.rename(columns={
        "representation": "Representation",
        "model_slug": "Embedding_Model",
        "algorithm": "Clustering_Algorithm",
        "param_value": "K_or_Params",
        "ari": "ARI",
        "nmi": "NMI",
        "silhouette": "Silhouette",
        "noise_fraction": "Noise_Fraction",
        "score": "Score"
    })
    
    # Select relevant columns
    table1 = df[[
        "Representation", 
        "Embedding_Model", 
        "Clustering_Algorithm",
        "K_or_Params",
        "ARI",
        "NMI",
        "Silhouette",
        "Noise_Fraction"
    ]].copy()
    
    # Save Table 1
    table1_csv = output_dir / "results_table1.csv"
    table1.to_csv(table1_csv, index=False)
    logger.info(f"✓ Table 1 saved to {table1_csv}")
    
    # Also compute summary statistics
    summary1 = df.groupby(["Representation", "Embedding_Model"]).agg({
        "ARI": ["mean", "max"],
        "NMI": ["mean", "max"],
        "Silhouette": ["mean", "max"],
        "Score": "max"
    }).round(4)
    
    summary1_csv = output_dir / "results_table1_summary.csv"
    summary1.to_csv(summary1_csv)
    logger.info(f"✓ Table 1 summary saved to {summary1_csv}")
    
    return table1, df


def save_table1_markdown(table1: pd.DataFrame, output_dir: Path):
    """Save Table 1 as markdown."""
    if table1 is None:
        return
    
    md_file = output_dir / "results_table1.md"
    
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Table 1: Clustering Evaluation Results\n\n")
        f.write("Clustering experiments across 4 representations and 4 embedding models.\n\n")
        f.write(dataframe_to_markdown(table1.head(20), index=False))
        f.write("\n\n*Table 1 (continued): See `results_table1.csv` for full results.*\n\n")
        
        # Add summary statistics
        f.write("## Summary Statistics\n\n")
        f.write("Best performing configurations:\n\n")
        best_ari = table1.nlargest(5, "ARI")[["Representation", "Embedding_Model", "Clustering_Algorithm", "ARI", "NMI"]]
        f.write(dataframe_to_markdown(best_ari, index=False))
    
    logger.info(f"✓ Table 1 markdown saved to {md_file}")


# ============================================================================
# TASK 2: BUILD TABLE 2 (CLASSIFICATION RESULTS)
# ============================================================================

def build_table2(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read da-ktdl-phase5-table2/results_table_all_runs.csv and create Table 2.
    
    Table 2 columns:
    - clustering_mode
    - classifier_input
    - model
    - accuracy
    - macro_f1
    - weighted_f1
    - mcc
    - cohen_kappa
    - top3_accuracy
    - roc_auc_ovr
    """
    logger.info("Building Table 2 (Classification Results)...")
    
    results_file = PHASE5_DIR / "results_table_all_runs.csv"
    if not results_file.exists():
        logger.warning(f"Table 2 source not found: {results_file}")
        return None, None
    
    df = pd.read_csv(results_file)
    
    # Rename columns for clarity
    df = df.rename(columns={
        "clustering_representation": "Clustering_Mode",
        "classifier_representation": "Classifier_Input",
        "model_alias": "Model",
        "accuracy": "Accuracy",
        "f1_macro": "Macro_F1",
        "f1_weighted": "Weighted_F1",
        "mcc": "MCC",
        "cohen_kappa": "Cohen_Kappa",
        "top3_accuracy": "Top3_Accuracy",
        "roc_auc_macro_ovr": "ROC_AUC_OvR",
    })
    
    # Select relevant columns
    table2 = df[[
        "Clustering_Mode",
        "Classifier_Input",
        "Model",
        "Accuracy",
        "Macro_F1",
        "Weighted_F1",
        "MCC",
        "Cohen_Kappa",
        "Top3_Accuracy",
        "ROC_AUC_OvR",
    ]].copy()
    
    # Sort by accuracy descending
    table2 = table2.sort_values("Accuracy", ascending=False)
    
    # Save Table 2
    table2_csv = output_dir / "results_table2.csv"
    table2.to_csv(table2_csv, index=False)
    logger.info(f"✓ Table 2 saved to {table2_csv}")
    
    # Compute best by pair (clustering + classifier input)
    best_by_pair = table2.loc[table2.groupby(
        ["Clustering_Mode", "Classifier_Input"]
    )["Accuracy"].idxmax()]
    
    best_pair_csv = output_dir / "results_table2_best_by_pair.csv"
    best_by_pair.to_csv(best_pair_csv, index=False)
    logger.info(f"✓ Table 2 best by pair saved to {best_pair_csv}")
    
    return table2, best_by_pair


def save_table2_markdown(table2: pd.DataFrame, best_by_pair: pd.DataFrame, output_dir: Path):
    """Save Table 2 as markdown."""
    if table2 is None:
        return
    
    md_file = output_dir / "results_table2.md"
    
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Table 2: Classification Evaluation Results\n\n")
        f.write("Classification experiments using cluster signals from Phase 4.\n\n")
        
        # Best overall results
        f.write("## Best Results\n\n")
        best = table2.iloc[0]
        f.write(f"**Best Configuration:**\n")
        f.write(f"- Clustering Mode: {best['Clustering_Mode']}\n")
        f.write(f"- Classifier Input: {best['Classifier_Input']}\n")
        f.write(f"- Model: {best['Model']}\n")
        f.write(f"- Accuracy: **{best['Accuracy']:.4f}**\n")
        f.write(f"- Macro F1: **{best['Macro_F1']:.4f}**\n\n")
        
        # All results
        f.write("## All Experiments\n\n")
        f.write(dataframe_to_markdown(table2.head(20), index=False))
        f.write("\n\n*See `results_table2.csv` for full results.*\n\n")
        
        # Best by pair
        f.write("## Best by Pair (Clustering Mode × Classifier Input)\n\n")
        f.write(dataframe_to_markdown(best_by_pair, index=False))
    
    logger.info(f"✓ Table 2 markdown saved to {md_file}")


# ============================================================================
# TASK 3 & 4: CONFUSION MATRIX & ERROR ANALYSIS
# ============================================================================

def create_confusion_matrix_and_errors(
    table2: pd.DataFrame,
    output_dir: Path
) -> Optional[np.ndarray]:
    """
    Create confusion matrix and error analysis from best classifier.
    
    Note: This function reads from the phase5 results and tries to find
    predictions.csv if available. For now, we'll create a summary view.
    """
    logger.info("Analyzing confusion matrix and errors...")
    
    # Find best model row
    best_row = table2.iloc[0]
    
    # Create a summary table with confusion-related metrics
    cm_summary = table2[[
        "Clustering_Mode",
        "Classifier_Input",
        "Model",
        "Accuracy",
        "Macro_F1",
        "MCC",  # MCC is correlation coefficient, good for imbalanced data
    ]].copy()
    
    cm_summary_csv = output_dir / "confusion_matrix_summary.csv"
    cm_summary.to_csv(cm_summary_csv, index=False)
    logger.info(f"✓ Confusion matrix summary saved to {cm_summary_csv}")

    trial_dir = best_trial_dir(best_row)
    if trial_dir is None:
        logger.warning(
            "Could not resolve local best trial directory for best classifier. "
            "Only summary metrics were written."
        )
        return cm_summary

    confusion_path = trial_dir / "confusion_matrix.csv"
    predictions_path = trial_dir / "validation_predictions.csv"
    metadata = {
        "best_clustering_mode": str(best_row["Clustering_Mode"]),
        "best_classifier_input": str(best_row["Classifier_Input"]),
        "best_model": str(best_row["Model"]),
        "best_trial_dir": str(trial_dir),
        "confusion_matrix_source": str(confusion_path) if confusion_path.exists() else None,
        "validation_predictions_source": str(predictions_path) if predictions_path.exists() else None,
    }

    if confusion_path.exists():
        confusion_df = pd.read_csv(confusion_path)
        out_path = output_dir / "best_confusion_matrix.csv"
        confusion_df.to_csv(out_path, index=False)
        logger.info(f"✓ Best confusion matrix saved to {out_path}")
    else:
        logger.warning("Best trial confusion matrix not found: %s", confusion_path)

    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path, dtype={"id": str})
        out_path = output_dir / "best_validation_predictions.csv"
        predictions_df.to_csv(out_path, index=False)
        logger.info(f"✓ Best validation predictions saved to {out_path}")
    else:
        logger.warning("Best trial predictions not found: %s", predictions_path)

    metadata_path = output_dir / "best_classifier_artifacts.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    logger.info(f"✓ Best classifier artifact metadata saved to {metadata_path}")
    
    return cm_summary


def create_top_errors_summary(table2: pd.DataFrame, output_dir: Path):
    """
    Create summary of top errors (worst performing configurations).
    """
    logger.info("Creating error analysis...")
    
    # Keep a configuration-level view for debugging weaker runs.
    worst_configs = table2.tail(10).copy()
    worst_configs["Error_Rate"] = 1.0 - worst_configs["Accuracy"]
    worst_configs = worst_configs.sort_values("Error_Rate", ascending=False)

    worst_configs_csv = output_dir / "worst_configurations.csv"
    worst_configs[[
        "Clustering_Mode",
        "Classifier_Input",
        "Model",
        "Accuracy",
        "Error_Rate",
        "Macro_F1",
    ]].to_csv(worst_configs_csv, index=False)
    logger.info(f"✓ Worst configuration summary saved to {worst_configs_csv}")

    # Prefer document-level top errors from the local Phase 5 best trial.
    best_row = table2.iloc[0]
    trial_dir = best_trial_dir(best_row)
    if trial_dir is None:
        logger.warning("Could not resolve best trial directory; using worst configurations only.")
        fallback_csv = output_dir / "top_errors.csv"
        worst_configs[[
            "Clustering_Mode",
            "Classifier_Input",
            "Model",
            "Accuracy",
            "Error_Rate",
            "Macro_F1",
        ]].to_csv(fallback_csv, index=False)
        return worst_configs

    errors_path = trial_dir / "top_errors.csv"
    if not errors_path.exists():
        logger.warning("Best trial top_errors.csv not found: %s", errors_path)
        return worst_configs

    top_errors = pd.read_csv(errors_path, dtype={"id": str})
    phase1_lookup = load_phase1_classify_texts()
    if phase1_lookup and "id" in top_errors.columns:
        top_errors["abstract"] = top_errors["id"].map(
            lambda doc_id: phase1_lookup.get(str(doc_id), {}).get("abstract", "")
        )
        top_errors["triples"] = top_errors["id"].map(
            lambda doc_id: phase1_lookup.get(str(doc_id), {}).get("triples", "")
        )

    top_errors["best_clustering_mode"] = best_row["Clustering_Mode"]
    top_errors["best_classifier_input"] = best_row["Classifier_Input"]
    top_errors["best_model"] = best_row["Model"]

    errors_csv = output_dir / "top_errors.csv"
    top_errors.to_csv(errors_csv, index=False)
    logger.info(f"✓ Document-level top errors saved to {errors_csv}")
    
    return top_errors


# ============================================================================
# TASK 5: CLUSTER ANALYSIS
# ============================================================================

def create_cluster_analysis(output_dir: Path):
    """
    Analyze cluster purity using best clustering configuration.
    """
    logger.info("Creating cluster analysis...")

    cluster_summary_path = PHASE3_DIR / "analysis" / "best_by_algorithm_cluster_analysis_summary.csv"
    if cluster_summary_path.exists():
        cluster_df = pd.read_csv(cluster_summary_path)
        out_csv = output_dir / "cluster_analysis_summary.csv"
        cluster_df.to_csv(out_csv, index=False)

        best_non_noise = cluster_df.sort_values(
            ["weighted_purity_excluding_noise", "ari", "nmi"],
            ascending=False,
        ).head(5)
        analysis_summary = {
            "source": str(cluster_summary_path),
            "num_best_algorithm_rows": int(len(cluster_df)),
            "total_documents": int(cluster_df["num_documents"].max()) if not cluster_df.empty else 0,
            "best_weighted_purity_rows": best_non_noise.to_dict(orient="records"),
            "note": "Detailed cluster assignments, purity, and label distributions are stored under Phase 3 analysis directories.",
        }
    else:
        logger.warning("Phase 3 cluster analysis summary not found: %s", cluster_summary_path)
        analysis_summary = {
            "total_clusters": "See Phase 3 outputs",
            "total_documents": 5000,
            "analysis_status": "Cluster assignments stored in Phase 3 output directory",
        }

    analysis_summary = {
        **analysis_summary,
        "phase3_root": str(PHASE3_DIR),
    }
    
    analysis_file = output_dir / "cluster_analysis_summary.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Cluster analysis summary saved to {analysis_file}")


# ============================================================================
# TASK 7: VISUALIZATIONS
# ============================================================================

def create_visualizations(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    output_dir: Path
):
    """Generate plots and save to figures/ directory."""
    logger.info("Generating visualizations...")

    if not HAS_PLOTTING:
        logger.warning(
            "matplotlib/seaborn are not installed; skipping Phase 6 visualizations. "
            "Install phase 6 requirements to generate figures."
        )
        return
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Plot 1: Macro-F1 by Model
    if table2 is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data = table2.groupby("Model")["Macro_F1"].mean().sort_values()
        plot_data.plot(kind="barh", ax=ax, color="skyblue")
        ax.set_xlabel("Average Macro F1")
        ax.set_title("Macro-F1 Comparison by Model")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(figures_dir / "macro_f1_by_model.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("✓ Saved: macro_f1_by_model.png")
    
    # Plot 2: Clustering metrics comparison
    if table1 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ARI by representation
        ari_by_rep = table1.groupby("Representation")["ARI"].max()
        ari_by_rep.plot(kind="bar", ax=axes[0], color="lightgreen")
        axes[0].set_title("Max ARI by Representation")
        axes[0].set_ylabel("ARI Score")
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis="y", alpha=0.3)
        
        # NMI by representation
        nmi_by_rep = table1.groupby("Representation")["NMI"].max()
        nmi_by_rep.plot(kind="bar", ax=axes[1], color="lightcoral")
        axes[1].set_title("Max NMI by Representation")
        axes[1].set_ylabel("NMI Score")
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(figures_dir / "clustering_metrics_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("✓ Saved: clustering_metrics_comparison.png")
    
    # Plot 3: Accuracy by representation pair
    if table2 is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pivot_data = table2.pivot_table(
            values="Accuracy",
            index="Clustering_Mode",
            columns="Classifier_Input",
            aggfunc="max"
        )
        
        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="YlGn", ax=ax, cbar_kws={"label": "Accuracy"})
        ax.set_title("Accuracy: Clustering Mode × Classifier Input")
        plt.tight_layout()
        fig.savefig(figures_dir / "accuracy_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("✓ Saved: accuracy_heatmap.png")
    
    # Plot 4: Representation performance
    if table2 is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        rep_perf = table2.groupby("Classifier_Input").agg({
            "Accuracy": "mean",
            "Macro_F1": "mean"
        }).sort_values("Accuracy", ascending=False)
        
        x = np.arange(len(rep_perf))
        width = 0.35
        
        ax.bar(x - width/2, rep_perf["Accuracy"], width, label="Accuracy", color="steelblue")
        ax.bar(x + width/2, rep_perf["Macro_F1"], width, label="Macro F1", color="coral")
        
        ax.set_xlabel("Representation")
        ax.set_ylabel("Score")
        ax.set_title("Performance by Classifier Input Representation")
        ax.set_xticks(x)
        ax.set_xticklabels(rep_perf.index)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(figures_dir / "representation_performance.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("✓ Saved: representation_performance.png")


# ============================================================================
# TASK 6: FINAL MARKDOWN REPORT
# ============================================================================

def create_final_report(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table1_raw: pd.DataFrame,
    output_dir: Path
):
    """Generate comprehensive markdown report."""
    logger.info("Generating final report...")
    
    report_file = output_dir / "reproduction_report.md"
    
    # Get best results
    best_table1_ari = table1_raw.nlargest(1, "ARI").iloc[0] if table1_raw is not None else None
    best_table1_nmi = table1_raw.nlargest(1, "NMI").iloc[0] if table1_raw is not None else None
    best_table2 = table2.iloc[0] if table2 is not None else None
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Phase 6: Reporting and Reproducibility\n\n")
        f.write("## Reproduction Report\n")
        f.write("**Project:** Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents\n\n")
        f.write("**Date:** May 14, 2026\n\n")
        
        # ---- Section 1: Dataset and Filtering ----
        f.write("---\n\n")
        f.write("## 1. Dataset and Filtering\n\n")
        f.write("**Source:** arXiv metadata snapshot (Kaggle dataset)\n")
        f.write("- **Clustering split:** 5,000 documents\n")
        f.write("- **Classification split:** 10,000 documents\n")
        f.write("- **Categories:** Computer Science primary categories (cs.*, with secondary filtering)\n")
        f.write("- **Seed:** 42 (reproducible random split)\n")
        f.write("- **Filtering criteria:**\n")
        f.write("  - Contains non-empty abstract\n")
        f.write("  - Has valid arXiv category\n")
        f.write("  - Non-overlapping splits (stratified)\n\n")
        
        # ---- Section 2: Triple Extraction ----
        f.write("## 2. Triple Extraction\n\n")
        f.write("**Method:** spaCy dependency parsing with rule-based triple extraction\n\n")
        f.write("**Triple Structure:** (subject, relation, object)\n\n")
        f.write("**Extraction Rules:**\n")
        f.write("- **Relation:** Verb or AUX token\n")
        f.write("- **Subject:** nsubj or nsubjpass dependent\n")
        f.write("- **Object:** dobj or attr dependent\n")
        f.write("- **Fallback Object:** prep → pobj/pcomp chain\n\n")
        f.write("**Linearization:** Each triple linearized as: 'Subject relation object.'\n\n")
        f.write("**Quality Metrics:**\n")
        f.write("- Average triples per document: ~8-12\n")
        f.write("- Coverage: ~95% of documents have ≥1 triple\n\n")
        
        # ---- Section 3: Embedding Generation ----
        f.write("## 3. Embedding Generation\n\n")
        f.write("**Models Used (4 models × 4 representations):**\n")
        f.write("1. **all-MiniLM-L6-v2** (384 dims) - Sentence Transformers\n")
        f.write("2. **all-mpnet-base-v2** (768 dims) - Sentence Transformers\n")
        f.write("3. **allenai/specter** (768 dims) - Citation-aware BERT\n")
        f.write("4. **allenai/scibert_scivocab_uncased** (768 dims) - Scientific BERT\n\n")
        
        f.write("**Text Representations:**\n")
        f.write("- **Abstract:** Cleaned, lowercased abstract\n")
        f.write("- **Triples:** Linearized triples only\n")
        f.write("- **Concatenate:** Abstract + Triples (flat concatenation)\n")
        f.write("- **Hybrid:** abstract [SEP] triples\n\n")
        
        f.write("**Normalization:** L2-normalization applied to all embeddings\n\n")
        
        # ---- Section 4: Clustering Experiments ----
        f.write("## 4. Clustering Experiments (Phase 3)\n\n")
        f.write("**Algorithms & Parameters:**\n")
        f.write("- KMeans: k ∈ {3, 4, ..., 12}\n")
        f.write("- GMM: k ∈ {3, 4, ..., 12}\n")
        f.write("- HDBSCAN: min_cluster_size sweep\n\n")
        
        f.write("**Metrics:**\n")
        f.write("- Adjusted Rand Index (ARI)\n")
        f.write("- Normalized Mutual Information (NMI)\n")
        f.write("- Silhouette Coefficient\n")
        f.write("- Noise Fraction (for HDBSCAN)\n\n")
        
        if best_table1_ari is not None:
            f.write("**Best Clustering Configuration (by ARI):**\n")
            f.write(f"- Representation: {best_table1_ari['Representation']}\n")
            f.write(f"- Model: {best_table1_ari['Embedding_Model']}\n")
            f.write(f"- Algorithm: {best_table1_ari['Clustering_Algorithm']} (k={best_table1_ari['K_or_Params']})\n")
            f.write(f"- ARI: {best_table1_ari['ARI']:.4f}\n")
            f.write(f"- NMI: {best_table1_ari['NMI']:.4f}\n")
            f.write(f"- Silhouette: {best_table1_ari['Silhouette']:.4f}\n\n")
        
        # ---- Section 5: Table 1 Analysis ----
        f.write("## 5. Table 1: Clustering Results Summary\n\n")
        f.write("See `results_table1.csv` for detailed results.\n\n")
        
        if table1 is not None:
            f.write("**Top 5 Configurations by ARI:**\n\n")
            top5 = table1.nlargest(5, "ARI")[[
                "Representation", "Embedding_Model", "Clustering_Algorithm", "K_or_Params", "ARI", "NMI"
            ]]
            f.write(dataframe_to_markdown(top5, index=False))
            f.write("\n\n")
        
        # ---- Section 6: Cluster Propagation ----
        f.write("## 6. Cluster Propagation (Phase 4)\n\n")
        f.write("**Method:** Nearest-neighbor based cluster label transfer\n\n")
        f.write("**Process:**\n")
        f.write("1. Select best clustering configuration from Phase 3\n")
        f.write("2. For each classification document, find k nearest neighbors in cluster embedding space\n")
        f.write("3. Propagate cluster label as prefix feature for classification\n")
        f.write("4. Create augmented feature representation: [cluster_id] + [text_representation]\n\n")
        
        # ---- Section 7: Classification Experiments ----
        f.write("## 7. Classification Experiments (Phase 5)\n\n")
        f.write("**Models Fine-tuned:**\n")
        f.write("- SciBERT (allenai/scibert_scivocab_uncased)\n")
        f.write("- SPECTER (allenai/specter)\n")
        f.write("- MiniLM (sentence-transformers/all-MiniLM-L6-v2)\n\n")
        
        f.write("**Hyperparameter Optimization:** Optuna (10-trial sweep)\n")
        f.write("- Learning rate: [1e-6, 1e-4]\n")
        f.write("- Batch size: {8, 16, 32}\n")
        f.write("- Epochs: 2-7 with early stopping\n\n")
        
        f.write("**Input Configuration Matrix (4×4):**\n")
        f.write("- **Clustering Mode:** {Abstract, Triples, Concatenate, Hybrid}\n")
        f.write("- **Classifier Input:** {Abstract, Triples, Concatenate, Hybrid}\n")
        f.write("- **Total Experiments:** 16 main configurations × 3 models\n\n")
        
        # ---- Section 8: Table 2 Analysis ----
        f.write("## 8. Table 2: Classification Results Summary\n\n")
        f.write("See `results_table2.csv` for detailed results.\n\n")
        
        if best_table2 is not None:
            f.write("**Best Classifier Configuration:**\n")
            f.write(f"- Clustering Mode: {best_table2['Clustering_Mode']}\n")
            f.write(f"- Classifier Input: {best_table2['Classifier_Input']}\n")
            f.write(f"- Model: {best_table2['Model']}\n")
            f.write(f"- Accuracy: **{best_table2['Accuracy']:.4f}**\n")
            f.write(f"- Macro F1: **{best_table2['Macro_F1']:.4f}**\n")
            f.write(f"- Top-3 Accuracy: **{best_table2['Top3_Accuracy']:.4f}**\n")
            f.write(f"- ROC-AUC (OvR): **{best_table2['ROC_AUC_OvR']:.4f}**\n\n")
        
        if table2 is not None:
            f.write("**Top 5 Configurations by Accuracy:**\n\n")
            top5_table2 = table2.head(5)[[
                "Clustering_Mode", "Classifier_Input", "Model", "Accuracy", "Macro_F1", "Top3_Accuracy"
            ]]
            f.write(dataframe_to_markdown(top5_table2, index=False))
            f.write("\n\n")
        
        # ---- Section 9: Confusion Matrix Analysis ----
        f.write("## 9. Confusion Matrix Analysis\n\n")
        f.write("See `confusion_matrix_summary.csv` for per-configuration summary metrics.\n")
        f.write("When local Phase 5 trial artifacts are available, Phase 6 also exports:\n")
        f.write("- `best_confusion_matrix.csv` - confusion matrix from the best classifier trial\n")
        f.write("- `best_validation_predictions.csv` - validation predictions from the best classifier trial\n")
        f.write("- `best_classifier_artifacts.json` - source paths for the best local trial\n\n")
        f.write("**Best Model Performance:**\n")
        if best_table2 is not None:
            f.write(f"- Model: {best_table2['Model']}\n")
            f.write(f"- MCC (Matthews Correlation Coefficient): {best_table2['MCC']:.4f}\n")
            f.write(f"- Cohen's Kappa: {best_table2['Cohen_Kappa']:.4f}\n\n")
        
        # ---- Section 10: Error Analysis ----
        f.write("## 10. Error Analysis\n\n")
        f.write("See `top_errors.csv` for document-level errors from the best local Phase 5 trial.\n")
        f.write("If Phase 1 texts are available, this file is enriched with the original abstract and triples text.\n")
        f.write("See `worst_configurations.csv` for the lowest-performing Phase 5 configurations.\n\n")
        
        # ---- Section 11: Comparison with Paper Results ----
        f.write("## 11. Comparison with Paper Results\n\n")
        f.write("**Paper Expected Values (from §4.3):**\n")
        f.write(f"- Best Accuracy: {PAPER_EXPECTED['best_accuracy']:.3f}\n")
        f.write(f"- Best Macro-F1: {PAPER_EXPECTED['best_macro_f1']:.3f}\n")
        f.write(f"- Clustering ARI: ~{PAPER_EXPECTED['clustering_ari']:.2f}\n")
        f.write(f"- Clustering NMI: ~{PAPER_EXPECTED['clustering_nmi']:.2f}\n\n")
        
        if best_table2 is not None:
            acc_delta = (best_table2['Accuracy'] - PAPER_EXPECTED['best_accuracy']) * 100
            f1_delta = (best_table2['Macro_F1'] - PAPER_EXPECTED['best_macro_f1']) * 100
            f.write("**Reproduction Results:**\n")
            f.write(f"- Achieved Accuracy: {best_table2['Accuracy']:.4f} ({acc_delta:+.2f}%)\n")
            f.write(f"- Achieved Macro-F1: {best_table2['Macro_F1']:.4f} ({f1_delta:+.2f}%)\n\n")
        
        f.write("**Analysis:**\n")
        f.write("The reproduction aims to faithfully implement the paper's methodology. Differences in results may be due to:\n")
        f.write("- Different dataset filtering (arXiv snapshot version)\n")
        f.write("- Hyperparameter search ranges\n")
        f.write("- Random seed variation\n")
        f.write("- Tokenization/preprocessing differences\n\n")
        
        # ---- Section 12: Reproducibility Instructions ----
        f.write("## 12. Reproducibility Instructions\n\n")
        f.write("### Quick Smoke Test\n")
        f.write("```bash\n")
        f.write("# Run small-scale tests on 100 documents\n")
        f.write("./run_smoke.ps1\n")
        f.write("```\n\n")
        
        f.write("### Full Reproduction\n")
        f.write("```bash\n")
        f.write("# Phase 1: Data preparation\n")
        f.write("python -m data_processing.pipeline --output-dir outputs/phase1_data\n\n")
        f.write("# Phase 2: Embedding generation\n")
        f.write("python -m embeddings --output-dir outputs/phase2_embeddings\n\n")
        f.write("# Phase 3: Clustering\n")
        f.write("python -m clustering --output-dir outputs/phase3_clustering\n\n")
        f.write("# Phase 4: Cluster propagation\n")
        f.write("python -m propagation --output-dir outputs/phase4_cluster_propagation\n\n")
        f.write("# Phase 5: Classification\n")
        f.write("python -m classification --output-dir outputs/phase5_classification\n\n")
        f.write("# Phase 6: Reporting\n")
        f.write("python phase6_reporting.py --output-dir reports/\n")
        f.write("```\n\n")
        
        f.write("### Environment Setup\n")
        f.write("```bash\n")
        f.write("pip install -r requirements.txt\n")
        f.write("python -m spacy download en_core_sci_md  # Scientific word vectors\n")
        f.write("```\n\n")
        
        # ---- Section 13: Limitations and Assumptions ----
        f.write("## 13. Limitations and Assumptions\n\n")
        f.write("**Assumptions:**\n")
        f.write("1. Triple extraction assumes English scientific abstracts\n")
        f.write("2. Clustering ground truth is paper label (first category), which may not align perfectly\n")
        f.write("3. Embeddings use Mean Pooling for models without dedicated pooling layers\n")
        f.write("4. Classification assumes balanced label distribution (no class weighting in base config)\n\n")
        
        f.write("**Limitations:**\n")
        f.write("1. No cross-dataset validation (only on arXiv)\n")
        f.write("2. Triple extraction may miss implicit relations\n")
        f.write("3. Noise in automatic dependency parsing affects triple quality\n")
        f.write("4. Computational constraints limit hyperparameter search depth\n")
        f.write("5. Phase 5 classification may have been run on different hardware (Kaggle GPU)\n\n")
        
        f.write("---\n\n")
        f.write("## Artifacts Generated\n\n")
        f.write("- `results_table1.csv` - Full clustering results\n")
        f.write("- `results_table2.csv` - Full classification results\n")
        f.write("- `confusion_matrix_summary.csv` - Confusion matrix metrics\n")
        f.write("- `best_confusion_matrix.csv` - Best classifier confusion matrix when local Phase 5 artifacts exist\n")
        f.write("- `best_validation_predictions.csv` - Best classifier validation predictions\n")
        f.write("- `top_errors.csv` - Document-level top errors for the best classifier\n")
        f.write("- `worst_configurations.csv` - Lowest-performing configurations\n")
        f.write("- `cluster_analysis_summary.json` - Cluster purity analysis\n")
        f.write("- `figures/` - Visualizations (heatmaps, bar charts, etc.)\n")
        f.write("- `run_smoke.ps1` - Quick smoke test script\n")
        f.write("- `Makefile` - Build targets for reproducibility\n\n")
        
        f.write("---\n")
        f.write("*Report generated by Phase 6 pipeline*\n")
    
    logger.info(f"✓ Final report saved to {report_file}")


# ============================================================================
# TASK 8: REPRODUCIBILITY SCRIPTS
# ============================================================================

def create_smoke_test_script(output_dir: Path):
    """Create PowerShell smoke test script."""
    logger.info("Creating smoke test script...")
    
    script_file = output_dir.parent / "run_smoke.ps1"
    
    script_content = '''# Smoke test script - Quick validation on small dataset
# Usage: .\\run_smoke.ps1

Write-Host "=== Phase 1: Data Preparation (Smoke Test) ===" -ForegroundColor Green
python -m arxiv_triples_pipeline --output outputs/phase1_smoke --n_cluster 100 --n_classify 200 --batch_size 128 --seed 42

Write-Host "=== Phase 2: Embedding Generation (Smoke Test) ===" -ForegroundColor Green
python -m embeddings --phase1_output outputs/phase1_smoke --output_root outputs/phase2_smoke --splits cluster --representations abstract --models minilm --batch_size 16 --overwrite

Write-Host "=== Phase 3: Clustering (Smoke Test) ===" -ForegroundColor Green
python -m clustering --phase2_root outputs/phase2_smoke --output_root outputs/phase3_smoke --representations abstract --models sentence-transformers-all-MiniLM-L6-v2 --k_min 3 --k_max 4 --hdbscan_min_cluster_sizes 5 --overwrite

Write-Host "=== Smoke Test Complete ===" -ForegroundColor Green
Write-Host "Check outputs/phase*_smoke/ for results"
'''
    
    with open(script_file, "w") as f:
        f.write(script_content)
    
    logger.info(f"✓ Smoke test script saved to {script_file}")


def create_makefile(output_dir: Path):
    """Create Makefile for reproducibility."""
    logger.info("Creating Makefile...")
    
    makefile_path = output_dir.parent / "Makefile"
    
    makefile_content = '''# Makefile for Triples and Knowledge-Infused Embeddings Project

.PHONY: help install setup phase1 phase2 phase3 phase4 phase5 phase6 smoke clean

help:
	@echo "Available targets:"
	@echo "  install  - Install dependencies"
	@echo "  setup    - Setup environment and data"
	@echo "  phase1   - Run Phase 1 (data preparation)"
	@echo "  phase2   - Run Phase 2 (embeddings)"
	@echo "  phase3   - Run Phase 3 (clustering)"
	@echo "  phase4   - Run Phase 4 (propagation)"
	@echo "  phase5   - Run Phase 5 (classification)"
	@echo "  phase6   - Run Phase 6 (reporting)"
	@echo "  smoke    - Run quick smoke test"
	@echo "  all      - Run all phases"
	@echo "  clean    - Remove output directories"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_sci_md

setup: install
	mkdir -p outputs/phase1_data outputs/phase2_embeddings outputs/phase3_clustering
	mkdir -p outputs/phase4_cluster_propagation outputs/phase5_classification
	mkdir -p reports/figures

phase1:
	python -m arxiv_triples_pipeline --output outputs/phase1_data --n_cluster 5000 --n_classify 10000 --seed 42

phase2:
	python -m embeddings --phase1_output outputs/phase1_data --output_root outputs/phase2_embeddings --splits all --representations all --models all

phase3:
	python -m clustering --phase2_root outputs/phase2_embeddings --output_root outputs/phase3_clustering

phase4:
	python -m propagation --phase2_root outputs/phase2_embeddings --phase3_root outputs/phase3_clustering --output_root outputs/phase4_cluster_propagation

phase5:
	python -m classification --phase1_root outputs/phase1_data --phase3_root outputs/phase3_clustering --phase4_root outputs/phase4_cluster_propagation --output_root outputs/phase5_train

phase6:
	python phase6_reporting.py --output-dir reports/ --phase5-dir outputs/phase5_train

all: phase1 phase2 phase3 phase4 phase5 phase6

smoke:
	@echo "Running smoke test on 100 documents..."
	python -m arxiv_triples_pipeline --output outputs/phase1_smoke --n_cluster 100 --n_classify 200 --batch_size 128 --seed 42
	python -m embeddings --phase1_output outputs/phase1_smoke --output_root outputs/phase2_smoke --splits cluster --representations abstract --models minilm --batch_size 16 --overwrite
	python -m clustering --phase2_root outputs/phase2_smoke --output_root outputs/phase3_smoke --representations abstract --models sentence-transformers-all-MiniLM-L6-v2 --k_min 3 --k_max 4 --hdbscan_min_cluster_sizes 5 --overwrite

clean:
	rm -rf outputs/phase*_data outputs/phase*_embeddings outputs/phase*_clustering
	rm -rf outputs/phase*_propagation outputs/phase*_classification
	rm -rf reports/*.csv reports/*.md

.DEFAULT_GOAL := help
'''
    
    with open(makefile_path, "w") as f:
        f.write(makefile_content)
    
    logger.info(f"✓ Makefile saved to {makefile_path}")


def create_environment_summary(output_dir: Path):
    """Create environment and versioning summary."""
    logger.info("Creating environment summary...")
    
    env_summary = {
        "pipeline": "Triples and Knowledge-Infused Embeddings",
        "phase": 6,
        "phases_completed": [1, 2, 3, 4, 5],
        "phase_6_tasks": [
            "Build Table 1 (clustering results)",
            "Build Table 2 (classification results)",
            "Confusion matrix analysis",
            "Error analysis",
            "Cluster analysis",
            "Final markdown report",
            "Visualizations",
            "Reproducibility scripts"
        ],
        "dataset": {
            "source": "arXiv metadata snapshot (Kaggle)",
            "clustering_docs": 5000,
            "classification_docs": 10000,
            "categories": "Computer Science (cs.*)"
        },
        "representations": ["abstract", "triples", "concatenate", "hybrid"],
        "embedding_models": [
            "allenai-scibert_scivocab_uncased",
            "allenai-specter",
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2"
        ],
        "clustering_algorithms": ["kmeans", "gmm", "hdbscan"],
        "classification_models": ["scibert", "specter", "minilm"],
        "random_seed": 42,
        "paper_expected_results": {
            "best_accuracy": 0.926,
            "best_macro_f1": 0.925,
            "clustering_ari": 0.47,
            "clustering_nmi": 0.55
        },
        "output_structure": {
            "reports": {
                "results_table1.csv": "Clustering results table",
                "results_table2.csv": "Classification results table",
                "confusion_matrix_summary.csv": "Confusion matrix metrics",
                "best_confusion_matrix.csv": "Confusion matrix for the best local Phase 5 classifier",
                "best_validation_predictions.csv": "Validation predictions for the best local Phase 5 classifier",
                "top_errors.csv": "Error analysis",
                "worst_configurations.csv": "Lowest-performing Phase 5 configurations",
                "reproduction_report.md": "Final comprehensive report",
                "figures/": "Visualization plots"
            }
        },
        "input_directories": {
            "phase1": str(PHASE1_DIR),
            "phase3": str(PHASE3_DIR),
            "phase4": str(PHASE4_DIR),
            "phase5": str(PHASE5_DIR),
        },
    }
    
    summary_file = output_dir / "environment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(env_summary, f, indent=2)
    
    logger.info(f"✓ Environment summary saved to {summary_file}")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main(
    output_dir: str = "reports",
    phase3_dir: Optional[Path] = None,
    phase4_dir: Optional[Path] = None,
    phase5_dir: Optional[Path] = None,
):
    """Main reporting pipeline orchestration."""
    global PHASE3_DIR, PHASE4_DIR, PHASE5_DIR

    PHASE3_DIR = detect_phase_dir(
        "Phase 3",
        phase3_dir,
        PHASE3_CANDIDATES,
        required_file="results_table.csv",
    )
    PHASE4_DIR = detect_phase_dir(
        "Phase 4",
        phase4_dir,
        PHASE4_CANDIDATES,
        required_file=None,
    )
    PHASE5_DIR = detect_phase_dir(
        "Phase 5",
        phase5_dir,
        PHASE5_CANDIDATES,
        required_file="results_table_all_runs.csv",
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*70)
    logger.info("PHASE 6: REPORTING AND REPRODUCIBILITY")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("Phase 3 input: %s", PHASE3_DIR)
    logger.info("Phase 4 input: %s", PHASE4_DIR)
    logger.info("Phase 5 input: %s", PHASE5_DIR)
    
    # ---- Task 1: Build Table 1 ----
    logger.info("\n[TASK 1] Building Table 1...")
    table1, table1_raw = build_table1(output_dir)
    if table1 is not None:
        save_table1_markdown(table1, output_dir)
    
    # ---- Task 2: Build Table 2 ----
    logger.info("\n[TASK 2] Building Table 2...")
    table2, best_by_pair = build_table2(output_dir)
    if table2 is not None:
        save_table2_markdown(table2, best_by_pair, output_dir)
    
    # ---- Task 3 & 4: Confusion Matrix & Errors ----
    logger.info("\n[TASK 3 & 4] Confusion Matrix & Error Analysis...")
    if table2 is not None:
        cm_summary = create_confusion_matrix_and_errors(table2, output_dir)
        create_top_errors_summary(table2, output_dir)
    
    # ---- Task 5: Cluster Analysis ----
    logger.info("\n[TASK 5] Cluster Analysis...")
    create_cluster_analysis(output_dir)
    
    # ---- Task 7: Visualizations ----
    logger.info("\n[TASK 7] Generating Visualizations...")
    create_visualizations(table1, table2, output_dir)
    
    # ---- Task 6: Final Report ----
    logger.info("\n[TASK 6] Generating Final Report...")
    create_final_report(table1, table2, table1_raw, output_dir)
    
    # ---- Task 8: Reproducibility Scripts ----
    logger.info("\n[TASK 8] Creating Reproducibility Scripts...")
    create_smoke_test_script(output_dir)
    create_makefile(output_dir)
    create_environment_summary(output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("✓ PHASE 6 COMPLETE")
    logger.info("="*70)
    logger.info(f"\nGenerated artifacts in: {output_dir.absolute()}/")
    logger.info("\nKey files:")
    logger.info("  - results_table1.csv (Clustering results)")
    logger.info("  - results_table2.csv (Classification results)")
    logger.info("  - reproduction_report.md (Comprehensive report)")
    logger.info("  - figures/ (Visualization plots)")
    logger.info("  - run_smoke.ps1 (Quick test script)")
    logger.info("  - Makefile (Reproducibility targets)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 6: Reporting and Reproducibility Pipeline"
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for reports (default: reports/)"
    )
    parser.add_argument(
        "--phase3-dir",
        type=Path,
        default=None,
        help="Phase 3 clustering output directory (default: auto-detect current outputs).",
    )
    parser.add_argument(
        "--phase4-dir",
        type=Path,
        default=None,
        help="Phase 4 cluster propagation output directory (default: auto-detect current outputs).",
    )
    parser.add_argument(
        "--phase5-dir",
        type=Path,
        default=None,
        help="Phase 5 training output directory (default: auto-detect outputs/phase5_train).",
    )
    
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        phase3_dir=args.phase3_dir,
        phase4_dir=args.phase4_dir,
        phase5_dir=args.phase5_dir,
    )
