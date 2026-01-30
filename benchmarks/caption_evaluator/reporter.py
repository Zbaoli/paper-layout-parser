"""
Benchmark Reporter

Generates reports from caption matching benchmark evaluation results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .batch import BenchmarkSummary


class BenchmarkReporter:
    """Generates reports from benchmark evaluation results."""

    def __init__(self):
        """Initialize the reporter."""
        pass

    def generate_json_report(
        self,
        summary: BenchmarkSummary,
        output_path: str,
    ) -> str:
        """
        Generate detailed JSON report.

        Args:
            summary: Benchmark summary to report
            output_path: Path to save the report

        Returns:
            Path to the saved report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

        return str(output_file)

    def generate_markdown_report(
        self,
        summary: BenchmarkSummary,
        output_path: str,
    ) -> str:
        """
        Generate human-readable Markdown report.

        Args:
            summary: Benchmark summary to report
            output_path: Path to save the report

        Returns:
            Path to the saved report
        """
        lines = [
            "# Caption Matching Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Information",
            "",
            f"- **Name:** {summary.dataset_name}",
            f"- **Version:** {summary.dataset_version}",
            f"- **Total Documents:** {summary.total_documents}",
            f"- **Successful Evaluations:** {summary.successful_evaluations}",
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Precision | {summary.precision:.4f} |",
            f"| Recall | {summary.recall:.4f} |",
            f"| F1 Score | {summary.f1:.4f} |",
            "",
            "## Detailed Counts",
            "",
            "| Count | Value |",
            "|-------|-------|",
            f"| True Positives | {summary.total_true_positives} |",
            f"| False Positives | {summary.total_false_positives} |",
            f"| False Negatives | {summary.total_false_negatives} |",
            "",
            "## Per-Type Metrics",
            "",
            "### Figure Matching",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        for key, value in summary.figure_metrics.items():
            if isinstance(value, float):
                lines.append(f"| {key.capitalize()} | {value:.4f} |")
            else:
                lines.append(f"| {key.capitalize()} | {value} |")

        lines.extend(
            [
                "",
                "### Table Matching",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]
        )

        for key, value in summary.table_metrics.items():
            if isinstance(value, float):
                lines.append(f"| {key.capitalize()} | {value:.4f} |")
            else:
                lines.append(f"| {key.capitalize()} | {value} |")

        # Per-document results
        lines.extend(
            [
                "",
                "## Per-Document Results",
                "",
                "| Document | Precision | Recall | F1 | Status |",
                "|----------|-----------|--------|-----|--------|",
            ]
        )

        for doc_result in summary.document_results:
            if doc_result.evaluation:
                eval_result = doc_result.evaluation
                lines.append(
                    f"| {doc_result.name} | {eval_result.precision:.4f} | "
                    f"{eval_result.recall:.4f} | {eval_result.f1:.4f} | OK |"
                )
            else:
                error = doc_result.error or "Unknown error"
                # Truncate long error messages
                if len(error) > 30:
                    error = error[:27] + "..."
                lines.append(f"| {doc_result.name} | - | - | - | Error: {error} |")

        # Error summary
        errors = [r for r in summary.document_results if r.error]
        if errors:
            lines.extend(
                [
                    "",
                    "## Errors",
                    "",
                ]
            )
            for doc_result in errors:
                lines.append(f"- **{doc_result.name}:** {doc_result.error}")

        # Configuration
        lines.extend(
            [
                "",
                "## Evaluator Configuration",
                "",
            ]
        )
        for key, value in summary.evaluator_config.items():
            lines.append(f"- **{key}:** {value}")

        # Footer
        lines.extend(
            [
                "",
                "---",
                "",
                f"*Report generated at {summary.created_at}*",
            ]
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return str(output_file)

    def generate_comparison_report(
        self,
        summaries: List[BenchmarkSummary],
        labels: Optional[List[str]] = None,
        output_path: str = "comparison.md",
    ) -> str:
        """
        Generate comparison report from multiple benchmark runs.

        Args:
            summaries: List of benchmark summaries to compare
            labels: Optional labels for each summary
            output_path: Path to save the comparison report

        Returns:
            Path to the saved report
        """
        if not labels:
            labels = [f"Run {i+1}" for i in range(len(summaries))]

        lines = [
            "# Caption Matching Benchmark Comparison",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overall Comparison",
            "",
            "| Run | Dataset | Documents | Precision | Recall | F1 |",
            "|-----|---------|-----------|-----------|--------|-----|",
        ]

        for label, summary in zip(labels, summaries):
            lines.append(
                f"| {label} | {summary.dataset_name} | {summary.total_documents} | "
                f"{summary.precision:.4f} | {summary.recall:.4f} | {summary.f1:.4f} |"
            )

        # Figure comparison
        lines.extend(
            [
                "",
                "## Figure Matching Comparison",
                "",
                "| Run | Precision | Recall | F1 |",
                "|-----|-----------|--------|-----|",
            ]
        )

        for label, summary in zip(labels, summaries):
            fig_metrics = summary.figure_metrics
            lines.append(
                f"| {label} | {fig_metrics.get('precision', 0):.4f} | "
                f"{fig_metrics.get('recall', 0):.4f} | {fig_metrics.get('f1', 0):.4f} |"
            )

        # Table comparison
        lines.extend(
            [
                "",
                "## Table Matching Comparison",
                "",
                "| Run | Precision | Recall | F1 |",
                "|-----|-----------|--------|-----|",
            ]
        )

        for label, summary in zip(labels, summaries):
            tbl_metrics = summary.table_metrics
            lines.append(
                f"| {label} | {tbl_metrics.get('precision', 0):.4f} | "
                f"{tbl_metrics.get('recall', 0):.4f} | {tbl_metrics.get('f1', 0):.4f} |"
            )

        # Configuration comparison
        lines.extend(
            [
                "",
                "## Configuration Comparison",
                "",
            ]
        )

        # Collect all config keys
        all_keys = set()
        for summary in summaries:
            all_keys.update(summary.evaluator_config.keys())

        if all_keys:
            header = "| Config | " + " | ".join(labels) + " |"
            sep = "|--------|" + "|".join(["-------"] * len(labels)) + "|"
            lines.extend([header, sep])

            for key in sorted(all_keys):
                values = [str(s.evaluator_config.get(key, "-")) for s in summaries]
                lines.append(f"| {key} | " + " | ".join(values) + " |")

        # Footer
        lines.extend(
            [
                "",
                "---",
                "",
                f"*Comparison report generated at {datetime.now().isoformat()}*",
            ]
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return str(output_file)


def load_summary_from_json(json_path: str) -> BenchmarkSummary:
    """
    Load a BenchmarkSummary from a JSON report file.

    Args:
        json_path: Path to the JSON report

    Returns:
        BenchmarkSummary instance
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = BenchmarkSummary(
        dataset_name=data.get("dataset", {}).get("name", "unknown"),
        dataset_version=data.get("dataset", {}).get("version", "1.0.0"),
        total_documents=data.get("summary", {}).get("total_documents", 0),
        successful_evaluations=data.get("summary", {}).get("successful_evaluations", 0),
        precision=data.get("summary", {}).get("precision", 0.0),
        recall=data.get("summary", {}).get("recall", 0.0),
        f1=data.get("summary", {}).get("f1", 0.0),
        figure_metrics=data.get("per_type_metrics", {}).get("figure", {}),
        table_metrics=data.get("per_type_metrics", {}).get("table", {}),
        total_true_positives=data.get("detailed_counts", {}).get("true_positives", 0),
        total_false_positives=data.get("detailed_counts", {}).get("false_positives", 0),
        total_false_negatives=data.get("detailed_counts", {}).get("false_negatives", 0),
        evaluator_config=data.get("evaluator_config", {}),
        created_at=data.get("created_at", ""),
    )

    return summary
