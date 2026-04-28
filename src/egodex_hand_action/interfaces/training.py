"""Training and experiment runner interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from egodex_hand_action.contracts.data import PredictionRecord
from egodex_hand_action.contracts.experiment import ExperimentConfig
from egodex_hand_action.contracts.metrics import AggregateMetrics
from egodex_hand_action.interfaces.dataset import EgoDexDataset
from egodex_hand_action.interfaces.model import HandActionModel


class Trainer(ABC):
    @abstractmethod
    def fit(
        self,
        model: HandActionModel,
        train_dataset: EgoDexDataset,
        validation_dataset: EgoDexDataset,
        config: ExperimentConfig,
    ) -> Path: ...


class ExperimentRunner(ABC):
    @abstractmethod
    def run_training(self, config: ExperimentConfig) -> Path: ...

    @abstractmethod
    def run_prediction(self, config: ExperimentConfig) -> Sequence[PredictionRecord]: ...

    @abstractmethod
    def run_evaluation(self, config: ExperimentConfig) -> AggregateMetrics: ...

