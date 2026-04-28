"""Metric computation for keypoint error and temporal stability."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from egodex_hand_action.contracts.data import HandActionSample, PredictionRecord
from egodex_hand_action.contracts.metrics import (
    AggregateMetrics,
    FrameMetrics,
    MetricName,
    StabilityMetrics,
    VideoMetrics,
)
from egodex_hand_action.interfaces.evaluation import Evaluator


class EvaluationError(ValueError):
    """Raised when predictions and ground truth cannot be evaluated."""


class KeypointEvaluator(Evaluator):
    """Evaluate 25-keypoint predictions against EgoDex ground truth."""

    def evaluate_frames(
        self,
        predictions: Sequence[PredictionRecord],
        ground_truth: Sequence[HandActionSample],
    ) -> Sequence[FrameMetrics]:
        ground_truth_by_id = self._ground_truth_by_sample_id(ground_truth)
        temporal_values = self._temporal_values(predictions, ground_truth)
        frame_metrics: list[FrameMetrics] = []
        for prediction in predictions:
            sample = ground_truth_by_id.get(prediction.sample_id)
            if sample is None:
                raise EvaluationError(f"No ground truth found for sample '{prediction.sample_id}'.")
            if sample.keypoints_25 is None:
                raise EvaluationError(f"Sample '{sample.sample_id}' is missing keypoints_25.")
            predicted = _matrix(prediction.prediction.keypoints)
            expected = _matrix(sample.keypoints_25.keypoints)
            _ensure_same_shape(predicted, expected, prediction.sample_id)
            frame_metrics.append(
                FrameMetrics(
                    sample_id=prediction.sample_id,
                    video_id=prediction.video_id,
                    frame_index=self._frame_index(prediction, sample),
                    method=prediction.method,
                    values={
                        MetricName.MSE: _mse(predicted, expected),
                        MetricName.L1: _l1(predicted, expected),
                        **temporal_values.get((prediction.method, prediction.sample_id), {}),
                    },
                )
            )
        return tuple(frame_metrics)

    def evaluate_videos(
        self,
        frame_metrics: Sequence[FrameMetrics],
    ) -> Sequence[VideoMetrics]:
        grouped = _group_frame_metrics(frame_metrics)
        video_metrics: list[VideoMetrics] = []
        for (method, video_id), frames in grouped.items():
            ordered = sorted(frames, key=lambda item: item.frame_index)
            values = _mean_metric_values([frame.values for frame in ordered])
            stability = StabilityMetrics(
                video_id=video_id,
                method=method,
                frame_jitter=values.get(MetricName.FRAME_JITTER),
                temporal_smoothness=values.get(MetricName.TEMPORAL_SMOOTHNESS),
            )
            video_metrics.append(
                VideoMetrics(
                    video_id=video_id,
                    method=method,
                    frame_count=len(ordered),
                    values=values,
                    stability=stability,
                )
            )
        return tuple(sorted(video_metrics, key=lambda item: (item.method, item.video_id)))

    def evaluate_sequences(
        self,
        predictions: Sequence[PredictionRecord],
        ground_truth: Sequence[HandActionSample],
    ) -> Sequence[VideoMetrics]:
        return self.evaluate_videos(self.evaluate_frames(predictions, ground_truth))

    def aggregate(
        self,
        video_metrics: Sequence[VideoMetrics],
    ) -> AggregateMetrics:
        if not video_metrics:
            raise EvaluationError("Cannot aggregate an empty video_metrics sequence.")
        methods = {metric.method for metric in video_metrics}
        if len(methods) != 1:
            raise EvaluationError("AggregateMetrics requires video metrics from exactly one method.")
        values = _weighted_mean_video_values(video_metrics)
        return AggregateMetrics(
            method=next(iter(methods)),
            sample_count=sum(metric.frame_count for metric in video_metrics),
            video_count=len(video_metrics),
            values=values,
        )

    def export_report(
        self,
        metrics: AggregateMetrics,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{metrics.method}_metrics.json"
        payload = {
            "method": metrics.method,
            "sample_count": metrics.sample_count,
            "video_count": metrics.video_count,
            "values": {name.value: value for name, value in metrics.values.items()},
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path

    def _temporal_values(
        self,
        predictions: Sequence[PredictionRecord],
        ground_truth: Sequence[HandActionSample],
    ) -> Mapping[tuple[str, str], Mapping[MetricName, float]]:
        gt_by_sample = self._ground_truth_by_sample_id(ground_truth)
        grouped: dict[tuple[str, str], list[PredictionRecord]] = defaultdict(list)
        for prediction in predictions:
            grouped[(prediction.method, prediction.video_id)].append(prediction)

        values_by_sample: dict[tuple[str, str], dict[MetricName, float]] = {}
        for (method, _video_id), records in grouped.items():
            ordered = sorted(records, key=lambda record: self._prediction_frame_index(record, gt_by_sample))
            jitter_by_sample = _frame_jitter_by_sample(ordered, gt_by_sample)
            smoothness_by_sample = _smoothness_by_sample(ordered)
            for sample_id, value in jitter_by_sample.items():
                values_by_sample.setdefault((method, sample_id), {})[MetricName.FRAME_JITTER] = value
            for sample_id, value in smoothness_by_sample.items():
                values_by_sample.setdefault((method, sample_id), {})[MetricName.TEMPORAL_SMOOTHNESS] = value
        return values_by_sample

    @staticmethod
    def _ground_truth_by_sample_id(samples: Sequence[HandActionSample]) -> Mapping[str, HandActionSample]:
        result: dict[str, HandActionSample] = {}
        for sample in samples:
            if sample.sample_id in result:
                raise EvaluationError(f"Duplicate ground truth sample_id: {sample.sample_id}")
            result[sample.sample_id] = sample
        return result

    @staticmethod
    def _frame_index(prediction: PredictionRecord, sample: HandActionSample) -> int:
        if prediction.frame_index is not None:
            return prediction.frame_index
        if sample.frame is not None:
            return sample.frame.frame_index
        if sample.clip is not None:
            return sample.clip.start_frame
        raise EvaluationError(f"Cannot resolve frame index for sample '{sample.sample_id}'.")

    def _prediction_frame_index(
        self,
        prediction: PredictionRecord,
        ground_truth_by_id: Mapping[str, HandActionSample],
    ) -> int:
        sample = ground_truth_by_id.get(prediction.sample_id)
        if sample is None:
            raise EvaluationError(f"No ground truth found for sample '{prediction.sample_id}'.")
        return self._frame_index(prediction, sample)


def _matrix(values: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(float(value) for value in row) for row in values)
    if not matrix:
        raise EvaluationError("Keypoint matrix must not be empty.")
    width = len(matrix[0])
    if width == 0:
        raise EvaluationError("Keypoint rows must not be empty.")
    if any(len(row) != width for row in matrix):
        raise EvaluationError("All keypoint rows must have the same dimension.")
    return matrix


def _ensure_same_shape(
    predicted: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
    sample_id: str,
) -> None:
    if len(predicted) != len(expected):
        raise EvaluationError(
            f"Prediction and ground truth keypoint counts differ for sample '{sample_id}'."
        )
    for predicted_point, expected_point in zip(predicted, expected):
        if len(predicted_point) != len(expected_point):
            raise EvaluationError(
                f"Prediction and ground truth keypoint dimensions differ for sample '{sample_id}'."
            )


def _mse(
    predicted: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
) -> float:
    values = [
        (predicted_value - expected_value) ** 2
        for predicted_point, expected_point in zip(predicted, expected)
        for predicted_value, expected_value in zip(predicted_point, expected_point)
    ]
    return sum(values) / len(values)


def _l1(
    predicted: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
) -> float:
    values = [
        abs(predicted_value - expected_value)
        for predicted_point, expected_point in zip(predicted, expected)
        for predicted_value, expected_value in zip(predicted_point, expected_point)
    ]
    return sum(values) / len(values)


def _group_frame_metrics(
    frame_metrics: Sequence[FrameMetrics],
) -> Mapping[tuple[str, str], Sequence[FrameMetrics]]:
    grouped: dict[tuple[str, str], list[FrameMetrics]] = defaultdict(list)
    for frame in frame_metrics:
        grouped[(frame.method, frame.video_id)].append(frame)
    return {key: tuple(value) for key, value in grouped.items()}


def _mean_metric_values(
    values: Sequence[Mapping[MetricName, float]],
) -> Mapping[MetricName, float]:
    grouped: dict[MetricName, list[float]] = defaultdict(list)
    for item in values:
        for name, value in item.items():
            grouped[name].append(float(value))
    return {
        name: sum(items) / len(items)
        for name, items in grouped.items()
        if items
    }


def _weighted_mean_video_values(video_metrics: Sequence[VideoMetrics]) -> Mapping[MetricName, float]:
    totals: dict[MetricName, float] = defaultdict(float)
    weights: dict[MetricName, int] = defaultdict(int)
    for metric in video_metrics:
        for name, value in metric.values.items():
            totals[name] += float(value) * metric.frame_count
            weights[name] += metric.frame_count
    return {
        name: totals[name] / weights[name]
        for name in totals
        if weights[name] > 0
    }


def _frame_jitter_by_sample(
    records: Sequence[PredictionRecord],
    ground_truth_by_sample: Mapping[str, HandActionSample],
) -> Mapping[str, float]:
    values: dict[str, float] = {}
    if len(records) < 2:
        return values
    for previous, current in zip(records, records[1:]):
        previous_gt = ground_truth_by_sample[previous.sample_id]
        current_gt = ground_truth_by_sample[current.sample_id]
        if previous_gt.keypoints_25 is None or current_gt.keypoints_25 is None:
            raise EvaluationError("Ground truth keypoints are required for frame jitter.")
        predicted_delta = _subtract(
            _matrix(current.prediction.keypoints),
            _matrix(previous.prediction.keypoints),
        )
        expected_delta = _subtract(
            _matrix(current_gt.keypoints_25.keypoints),
            _matrix(previous_gt.keypoints_25.keypoints),
        )
        _ensure_same_shape(predicted_delta, expected_delta, current.sample_id)
        values[current.sample_id] = _l1(predicted_delta, expected_delta)
    return values


def _smoothness_by_sample(records: Sequence[PredictionRecord]) -> Mapping[str, float]:
    values: dict[str, float] = {}
    if len(records) < 3:
        return values
    for first, second, third in zip(records, records[1:], records[2:]):
        acceleration = _subtract(
            _subtract(_matrix(third.prediction.keypoints), _matrix(second.prediction.keypoints)),
            _subtract(_matrix(second.prediction.keypoints), _matrix(first.prediction.keypoints)),
        )
        zero = tuple(tuple(0.0 for _ in point) for point in acceleration)
        values[third.sample_id] = _l1(acceleration, zero)
    return values


def _subtract(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    _ensure_same_shape(left, right, "sequence")
    return tuple(
        tuple(left_value - right_value for left_value, right_value in zip(left_point, right_point))
        for left_point, right_point in zip(left, right)
    )
