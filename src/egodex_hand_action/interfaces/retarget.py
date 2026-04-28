"""Interfaces for 21-to-25 keypoint retargeting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from egodex_hand_action.contracts.data import KeypointPrediction, PredictionRecord


class KeypointRetargeter(ABC):
    @abstractmethod
    def source_topology_name(self) -> str: ...

    @abstractmethod
    def target_topology_name(self) -> str: ...

    @abstractmethod
    def retarget_prediction(self, prediction: KeypointPrediction) -> KeypointPrediction: ...

    @abstractmethod
    def retarget_records(self, records: Sequence[PredictionRecord]) -> Sequence[PredictionRecord]: ...

