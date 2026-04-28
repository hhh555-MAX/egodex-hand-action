"""Evaluation interfaces for error and stability analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from egodex_hand_action.contracts.data import HandActionSample, PredictionRecord
from egodex_hand_action.contracts.metrics import AggregateMetrics, FrameMetrics, VideoMetrics


class Evaluator(ABC):
    @abstractmethod
    def evaluate_frames(
        self,
        predictions: Sequence[PredictionRecord],
        ground_truth: Sequence[HandActionSample],
    ) -> Sequence[FrameMetrics]: ...

    @abstractmethod
    def evaluate_videos(
        self,
        frame_metrics: Sequence[FrameMetrics],
    ) -> Sequence[VideoMetrics]: ...

    @abstractmethod
    def aggregate(
        self,
        video_metrics: Sequence[VideoMetrics],
    ) -> AggregateMetrics: ...

    @abstractmethod
    def export_report(
        self,
        metrics: AggregateMetrics,
        output_dir: Path,
    ) -> Path: ...

