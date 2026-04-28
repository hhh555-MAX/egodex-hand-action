"""Metric result contracts for error and stability analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class MetricName(str, Enum):
    MSE = "mse"
    L1 = "l1"
    FRAME_JITTER = "frame_jitter"
    TEMPORAL_SMOOTHNESS = "temporal_smoothness"


@dataclass(frozen=True)
class FrameMetrics:
    sample_id: str
    video_id: str
    frame_index: int
    method: str
    values: Mapping[MetricName, float]


@dataclass(frozen=True)
class StabilityMetrics:
    video_id: str
    method: str
    frame_jitter: float | None = None
    temporal_smoothness: float | None = None
    per_keypoint_values: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoMetrics:
    video_id: str
    method: str
    frame_count: int
    values: Mapping[MetricName, float]
    stability: StabilityMetrics | None = None


@dataclass(frozen=True)
class AggregateMetrics:
    method: str
    sample_count: int
    video_count: int
    values: Mapping[MetricName, float]

