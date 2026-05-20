"""Reporting utilities for Phase 7."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from .io_utils import read_csv, write_csv


def create_phase7_report(
    *,
    classification_root: Path,
    quality_root: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    quality_summary_path = quality_root / "triple_quality_summary.csv"
    results_path = classification_root / "results_table_phase7.csv"
    per_label_delta_path = classification_root / "per_label_delta_vs_hybrid.csv"
    per_label_metrics_path = classification_root / "per_label_metrics.csv"

    quality_rows = read_csv(quality_summary_path) if quality_summary_path.exists() else []
    result_rows = read_csv(results_path) if results_path.exists() else []
    delta_rows = read_csv(per_label_delta_path) if per_label_delta_path.exists() else []
    per_label_rows = read_csv(per_label_metrics_path) if per_label_metrics_path.exists() else []

    if quality_summary_path.exists():
        shutil.copy2(quality_summary_path, output_dir / "triple_quality_summary.csv")
    if results_path.exists():
        shutil.copy2(results_path, output_dir / "results_table_phase7.csv")
    if per_label_delta_path.exists():
        shutil.copy2(per_label_delta_path, output_dir / "per_label_delta_vs_hybrid.csv")

    report_path = output_dir / "phase7_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# Phase 7 Report - Triple-Quality-Aware Hybrid Classification\n\n")
        fh.write("## Research Question\n\n")
        fh.write(
            "Phase 7 kiểm chứng giả thuyết: nếu triples có nhiễu, việc chấm điểm "
            "và chỉ dùng triples chất lượng cao có cải thiện classification so với "
            "`hybrid = abstract [SEP] all triples` hay không.\n\n"
        )
        fh.write("Phase 7 cố ý không dùng Phase 4 propagation trong experiment chính. ")
        fh.write(
            "Như vậy delta metric phản ánh tác động của triple-quality filtering, "
            "không bị lẫn với cluster signal.\n\n"
        )

        fh.write("## Triple Quality Summary\n\n")
        if quality_rows:
            fh.write(_markdown_table(quality_rows))
            fh.write("\n\n")
        else:
            fh.write(
                "Chưa tìm thấy `triple_quality_summary.csv`. "
                "Hãy chạy `python -m phase7_quality build` trước.\n\n"
            )

        fh.write("## Classification Ablation\n\n")
        if result_rows:
            ranked_rows = sorted(
                result_rows,
                key=lambda row: (
                    -_float(row.get("accuracy")),
                    -_float(row.get("f1_macro")),
                ),
            )
            fh.write("### Results Sorted by Accuracy\n\n")
            fh.write(_markdown_table(_compact_result_rows(ranked_rows)))
            fh.write("\n\n")

            baseline = _best_hybrid_baseline(result_rows)
            fh.write("### Delta vs Hybrid Baseline\n\n")
            if baseline:
                fh.write(
                    f"Hybrid baseline: accuracy={_float(baseline.get('accuracy')):.4f}, "
                    f"macro-F1={_float(baseline.get('f1_macro')):.4f}.\n\n"
                )
                delta_table = _build_metric_delta_rows(result_rows, baseline)
                fh.write(_markdown_table(delta_table))
                fh.write("\n\n")
            else:
                fh.write(
                    "Không có baseline `hybrid` trong kết quả classification, "
                    "nên chưa tính được delta chính.\n\n"
                )

            fh.write("### Interpretation Checklist\n\n")
            fh.write("- Nếu quality variants có macro-F1 cao hơn `hybrid`, filtering giúp giảm triple noise.\n")
            fh.write("- Nếu accuracy tăng nhưng macro-F1 không tăng, lợi ích chủ yếu nằm ở class lớn.\n")
            fh.write("- Nếu quality variants thấp hơn `hybrid`, rule-based scoring hiện tại chưa đủ hoặc đã loại mất triple hữu ích.\n")
            fh.write("- Nếu `triples` vẫn thấp hơn `abstract/hybrid`, triples-only vẫn thiếu ngữ cảnh so với abstract đầy đủ.\n\n")
        else:
            fh.write(
                "Chưa tìm thấy `results_table_phase7.csv`. "
                "Phần build quality đã có thể chạy nhẹ; phần classify cần fine-tune model "
                "và có thể mất thời gian/GPU. Chỉ chạy khi đã sẵn sàng.\n\n"
            )

        fh.write("## Per-Label Effects\n\n")
        if delta_rows:
            fh.write("Các label hưởng lợi/hại nhất so với baseline `hybrid`:\n\n")
            sorted_delta = sorted(delta_rows, key=lambda row: -_float(row.get("delta_f1_vs_hybrid")))
            fh.write("### Largest Gains\n\n")
            fh.write(_markdown_table(sorted_delta[:15]))
            fh.write("\n\n### Largest Drops\n\n")
            fh.write(_markdown_table(list(reversed(sorted_delta[-15:]))))
            fh.write("\n\n")
        elif per_label_rows:
            fh.write(
                "Có `per_label_metrics.csv`, nhưng chưa có baseline `hybrid` hoặc delta table. "
                "Hãy đảm bảo Phase 7 classify có chạy representation `hybrid`.\n\n"
            )
        else:
            fh.write("Chưa có per-label metrics vì chưa chạy Phase 7 classification.\n\n")

        fh.write("## Generated Files\n\n")
        fh.write("- `phase7_report.md`\n")
        if quality_summary_path.exists():
            fh.write("- `triple_quality_summary.csv`\n")
        if results_path.exists():
            fh.write("- `results_table_phase7.csv`\n")
        if per_label_delta_path.exists():
            fh.write("- `per_label_delta_vs_hybrid.csv`\n")

    _maybe_create_figures(
        quality_rows=quality_rows,
        result_rows=result_rows,
        delta_rows=delta_rows,
        figures_dir=figures_dir,
    )

    return {
        "report_path": str(report_path),
        "has_quality_summary": bool(quality_rows),
        "has_classification_results": bool(result_rows),
        "has_per_label_delta": bool(delta_rows),
    }


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _compact_result_rows(rows: list[dict]) -> list[dict]:
    compact = []
    for row in rows:
        compact.append(
            {
                "representation": row.get("classifier_representation")
                or row.get("phase7_representation", ""),
                "model": row.get("model_alias", ""),
                "accuracy": _round4(row.get("accuracy")),
                "macro_f1": _round4(row.get("f1_macro")),
                "weighted_f1": _round4(row.get("f1_weighted")),
                "mcc": _round4(row.get("mcc")),
                "top3_accuracy": _round4(row.get("top3_accuracy")),
            }
        )
    return compact


def _best_hybrid_baseline(rows: list[dict]) -> dict | None:
    hybrid_rows = [
        row for row in rows
        if (row.get("classifier_representation") or row.get("phase7_representation")) == "hybrid"
    ]
    if not hybrid_rows:
        return None
    return max(hybrid_rows, key=lambda row: (_float(row.get("accuracy")), _float(row.get("f1_macro"))))


def _build_metric_delta_rows(rows: list[dict], baseline: dict) -> list[dict]:
    baseline_acc = _float(baseline.get("accuracy"))
    baseline_macro = _float(baseline.get("f1_macro"))
    baseline_weighted = _float(baseline.get("f1_weighted"))
    delta_rows = []
    for row in rows:
        representation = row.get("classifier_representation") or row.get("phase7_representation", "")
        if representation == "hybrid":
            continue
        delta_rows.append(
            {
                "representation": representation,
                "model": row.get("model_alias", ""),
                "delta_accuracy": _round4(_float(row.get("accuracy")) - baseline_acc),
                "delta_macro_f1": _round4(_float(row.get("f1_macro")) - baseline_macro),
                "delta_weighted_f1": _round4(_float(row.get("f1_weighted")) - baseline_weighted),
            }
        )
    delta_rows.sort(key=lambda row: -_float(row["delta_macro_f1"]))
    return delta_rows


def _maybe_create_figures(
    *,
    quality_rows: list[dict],
    result_rows: list[dict],
    delta_rows: list[dict],
    figures_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if quality_rows:
        summary = [row for row in quality_rows if row.get("split") != "all"]
        if summary:
            labels = [row["split"] for row in summary]
            values = [_float(row.get("avg_quality_score")) for row in summary]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(labels, values, color="#4C78A8")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Average quality score")
            ax.set_title("Triple Quality by Split")
            fig.tight_layout()
            fig.savefig(figures_dir / "quality_score_distribution.png", dpi=200)
            plt.close(fig)

    if result_rows:
        rows = _compact_result_rows(result_rows)
        labels = [row["representation"] for row in rows]
        values = [_float(row["macro_f1"]) for row in rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, values, color="#F58518")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Macro-F1")
        ax.set_title("Phase 7 Macro-F1 Comparison")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(figures_dir / "macro_f1_comparison.png", dpi=200)
        plt.close(fig)

    if delta_rows:
        top = sorted(delta_rows, key=lambda row: abs(_float(row.get("delta_f1_vs_hybrid"))), reverse=True)[:20]
        labels = [f"{row['representation']}:{row['label']}" for row in top]
        values = [_float(row.get("delta_f1_vs_hybrid")) for row in top]
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#54A24B" if value >= 0 else "#E45756" for value in values]
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Delta F1 vs hybrid")
        ax.set_title("Largest Per-Label F1 Changes")
        fig.tight_layout()
        fig.savefig(figures_dir / "per_label_f1_delta.png", dpi=200)
        plt.close(fig)


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _float(value) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _round4(value) -> float:
    return round(_float(value), 4)
