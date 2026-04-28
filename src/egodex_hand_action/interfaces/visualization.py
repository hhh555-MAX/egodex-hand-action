"""Visualization and reporting interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from egodex_hand_action.contracts.data import HandActionSample, PredictionRecord
from egodex_hand_action.contracts.metrics import AggregateMetrics, FrameMetrics, VideoMetrics


class Visualizer(ABC):
    @abstractmethod
    def render_frame_overlay(
        self,
        sample: HandActionSample,
        predictions: Sequence[PredictionRecord],
        output_path: Path,
    ) -> Path: ...

    @abstractmethod
    def render_video_overlay(
        self,
        samples: Sequence[HandActionSample],
        predictions: Sequence[PredictionRecord],
        output_path: Path,
    ) -> Path: ...

    @abstractmethod
    def plot_metric_curves(
        self,
        frame_metrics: Sequence[FrameMetrics],
        video_metrics: Sequence[VideoMetrics],
        output_path: Path,
    ) -> Path: ...


class ReportWriter(ABC):
    @abstractmethod
    def write_experiment_report(
        self,
        metrics: Sequence[AggregateMetrics],
        output_path: Path,
    ) -> Path: ...
