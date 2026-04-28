"""Model interfaces for baseline and external model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

from egodex_hand_action.contracts.data import HandActionSample, PredictionRecord


class HandActionModel(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: Path) -> None: ...

    @abstractmethod
    def save_checkpoint(self, checkpoint_path: Path) -> None: ...

    @abstractmethod
    def predict(self, samples: Sequence[HandActionSample]) -> Sequence[PredictionRecord]: ...


class BaselineModelFactory(ABC):
    @abstractmethod
    def create(self, config: Mapping[str, Any]) -> HandActionModel: ...


class PhantomAdapter(ABC):
    @abstractmethod
    def prepare_input(self, sample: HandActionSample) -> Mapping[str, Any]: ...

    @abstractmethod
    def predict_21_keypoints(self, sample: HandActionSample) -> PredictionRecord: ...

