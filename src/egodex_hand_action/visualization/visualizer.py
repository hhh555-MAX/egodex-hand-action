"""Pillow and SVG based visualizations for keypoint experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw

from egodex_hand_action.contracts.data import CoordinateSpace, HandActionSample, PredictionRecord
from egodex_hand_action.contracts.metrics import FrameMetrics, MetricName, VideoMetrics
from egodex_hand_action.interfaces.visualization import Visualizer


class VisualizationError(RuntimeError):
    """Raised when a visualization artifact cannot be created."""


class PillowKeypointVisualizer(Visualizer):
    """Render frame overlays and metric curves without heavyweight plotting dependencies."""

    GT_COLOR = (43, 145, 88)
    PREDICTION_COLORS = (
        (220, 68, 59),
        (61, 117, 214),
        (230, 154, 40),
        (130, 81, 186),
    )

    def render_frame_overlay(
        self,
        sample: HandActionSample,
        predictions: Sequence[PredictionRecord],
        output_path: Path,
    ) -> Path:
        image_path = self._sample_image_path(sample)
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        if sample.keypoints_25 is not None:
            self._draw_keypoints(
                draw,
                sample.keypoints_25.keypoints,
                image.size,
                color=self.GT_COLOR,
                radius=3,
            )

        matching_predictions = [
            prediction for prediction in predictions if prediction.sample_id == sample.sample_id
        ]
        for index, prediction in enumerate(matching_predictions):
            color = self.PREDICTION_COLORS[index % len(self.PREDICTION_COLORS)]
            self._draw_keypoints(
                draw,
                prediction.prediction.keypoints,
                image.size,
                color=color,
                radius=2,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path

    def render_video_overlay(
        self,
        samples: Sequence[HandActionSample],
        predictions: Sequence[PredictionRecord],
        output_path: Path,
    ) -> Path:
        output_path.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, str]] = []
        ordered = sorted(samples, key=self._sample_sort_key)
        for sample in ordered:
            frame_path = output_path / f"{sample.sample_id}.png"
            self.render_frame_overlay(sample, predictions, frame_path)
            manifest.append({"sample_id": sample.sample_id, "frame_path": str(frame_path)})

        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(
            json.dumps({"frames": manifest}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path

    def plot_metric_curves(
        self,
        frame_metrics: Sequence[FrameMetrics],
        video_metrics: Sequence[VideoMetrics],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metric_name = self._preferred_metric(frame_metrics, video_metrics)
        points_by_method = self._frame_points_by_method(frame_metrics, metric_name)
        if not points_by_method:
            points_by_method = self._video_points_by_method(video_metrics, metric_name)
        if not points_by_method:
            raise VisualizationError("No metric values available for plotting.")
        output_path.write_text(
            _line_chart_svg(
                points_by_method=points_by_method,
                metric_name=metric_name.value,
                width=960,
                height=420,
            ),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def _sample_image_path(sample: HandActionSample) -> Path:
        if sample.frame is not None and sample.frame.image_path is not None:
            return sample.frame.image_path
        if sample.clip is not None and sample.clip.frame_paths:
            return sample.clip.frame_paths[0]
        raise VisualizationError(f"Sample '{sample.sample_id}' has no image path to render.")

    @classmethod
    def _draw_keypoints(
        cls,
        draw: ImageDraw.ImageDraw,
        keypoints: Sequence[Sequence[float]],
        image_size: tuple[int, int],
        *,
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        width, height = image_size
        points = [
            cls._to_pixel(point, width=width, height=height)
            for point in keypoints
        ]
        for x, y in points:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=color,
            )

    @staticmethod
    def _to_pixel(
        point: Sequence[float],
        *,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        if len(point) < 2:
            raise VisualizationError("Each keypoint must contain at least x and y.")
        x = float(point[0])
        y = float(point[1])
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x * width, y * height
        return x, y

    @staticmethod
    def _sample_sort_key(sample: HandActionSample) -> tuple[str, int, str]:
        if sample.frame is not None:
            return sample.frame.video_id, sample.frame.frame_index, sample.sample_id
        if sample.clip is not None:
            return sample.clip.video_id, sample.clip.start_frame, sample.sample_id
        return "", 0, sample.sample_id

    @staticmethod
    def _preferred_metric(
        frame_metrics: Sequence[FrameMetrics],
        video_metrics: Sequence[VideoMetrics],
    ) -> MetricName:
        for metric in (MetricName.MSE, MetricName.L1, MetricName.FRAME_JITTER):
            if any(metric in frame.values for frame in frame_metrics):
                return metric
            if any(metric in video.values for video in video_metrics):
                return metric
        return MetricName.MSE

    @staticmethod
    def _frame_points_by_method(
        frame_metrics: Sequence[FrameMetrics],
        metric_name: MetricName,
    ) -> Mapping[str, Sequence[tuple[float, float]]]:
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for frame in sorted(frame_metrics, key=lambda item: (item.method, item.video_id, item.frame_index)):
            if metric_name in frame.values:
                grouped[frame.method].append((float(frame.frame_index), float(frame.values[metric_name])))
        return {method: tuple(points) for method, points in grouped.items() if points}

    @staticmethod
    def _video_points_by_method(
        video_metrics: Sequence[VideoMetrics],
        metric_name: MetricName,
    ) -> Mapping[str, Sequence[tuple[float, float]]]:
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for index, video in enumerate(sorted(video_metrics, key=lambda item: (item.method, item.video_id))):
            if metric_name in video.values:
                grouped[video.method].append((float(index), float(video.values[metric_name])))
        return {method: tuple(points) for method, points in grouped.items() if points}


def _line_chart_svg(
    *,
    points_by_method: Mapping[str, Sequence[tuple[float, float]]],
    metric_name: str,
    width: int,
    height: int,
) -> str:
    margin = 48
    colors = ("#dc443b", "#3d75d6", "#e69a28", "#8251ba")
    all_points = [point for points in points_by_method.values() for point in points]
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    def scale(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        sx = margin + ((x - min_x) / (max_x - min_x)) * (width - 2 * margin)
        sy = height - margin - ((y - min_y) / (max_y - min_y)) * (height - 2 * margin)
        return sx, sy

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="28" font-family="Arial" font-size="18" fill="#222">Metric: {metric_name}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#555"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#555"/>',
    ]
    for index, (method, points) in enumerate(points_by_method.items()):
        color = colors[index % len(colors)]
        scaled = [scale(point) for point in points]
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in scaled)
        lines.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in scaled:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
        legend_y = 52 + index * 20
        lines.append(f'<rect x="{width - 180}" y="{legend_y - 10}" width="10" height="10" fill="{color}"/>')
        lines.append(f'<text x="{width - 164}" y="{legend_y}" font-family="Arial" font-size="13" fill="#222">{method}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"

