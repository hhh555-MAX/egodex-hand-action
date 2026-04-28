"""Inference interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from egodex_hand_action.contracts.data import HandActionSample, PredictionRecord
from egodex_hand_action.interfaces.model import HandActionModel


class Predictor(ABC):
    @abstractmethod
    def predict_samples(
        self,
        model: HandActionModel,
        samples: Sequence[HandActionSample],
    ) -> Sequence[PredictionRecord]: ...

    @abstractmethod
    def predict_video(
        self,
        model: HandActionModel,
        video_path: Path,
    ) -> Sequence[PredictionRecord]: ...

    @abstractmethod
    def export_predictions(
        self,
        records: Sequence[PredictionRecord],
        output_path: Path,
    ) -> Path: ...

