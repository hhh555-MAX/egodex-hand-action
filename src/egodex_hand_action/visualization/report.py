"""Markdown experiment report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from egodex_hand_action.contracts.metrics import AggregateMetrics, MetricName
from egodex_hand_action.interfaces.visualization import ReportWriter


class MarkdownReportWriter(ReportWriter):
    """Write a compact Markdown report for aggregate experiment metrics."""

    def write_experiment_report(
        self,
        metrics: Sequence[AggregateMetrics],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self._render(metrics), encoding="utf-8")
        return output_path

    def _render(self, metrics: Sequence[AggregateMetrics]) -> str:
        lines = [
            "# EgoDex Hand Action Experiment Report",
            "",
            "## Summary",
            "",
        ]
        if not metrics:
            lines.extend(["No aggregate metrics were provided.", ""])
            return "\n".join(lines)

        metric_names = self._metric_names(metrics)
        header = ["Method", "Samples", "Videos", *[name.value for name in metric_names]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for item in metrics:
            row = [
                item.method,
                str(item.sample_count),
                str(item.video_count),
                *[
                    _format_value(item.values.get(name))
                    for name in metric_names
                ],
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.extend(self._analysis_notes(metrics))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _metric_names(metrics: Sequence[AggregateMetrics]) -> Sequence[MetricName]:
        ordered = (
            MetricName.MSE,
            MetricName.L1,
            MetricName.FRAME_JITTER,
            MetricName.TEMPORAL_SMOOTHNESS,
        )
        present = {name for item in metrics for name in item.values}
        return tuple(name for name in ordered if name in present)

    @staticmethod
    def _analysis_notes(metrics: Sequence[AggregateMetrics]) -> Sequence[str]:
        lines = ["## Notes", ""]
        for metric_name in (MetricName.MSE, MetricName.L1, MetricName.FRAME_JITTER, MetricName.TEMPORAL_SMOOTHNESS):
            available = [
                item for item in metrics
                if metric_name in item.values
            ]
            if not available:
                continue
            best = min(available, key=lambda item: item.values[metric_name])
            lines.append(f"- Best {metric_name.value}: `{best.method}` with `{best.values[metric_name]:.6f}`.")
        lines.append("")
        return lines


def _format_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"
