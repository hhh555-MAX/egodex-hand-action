"""Dataset and preprocessing interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from egodex_hand_action.contracts.data import DatasetSplit, HandActionSample


class EgoDexDataset(ABC):
    @abstractmethod
    def split(self) -> DatasetSplit: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_sample(self, index: int) -> HandActionSample: ...


class DatasetIndexBuilder(ABC):
    @abstractmethod
    def build(self, dataset_root: Path, output_dir: Path) -> Sequence[Path]: ...


class Preprocessor(ABC):
    @abstractmethod
    def transform_sample(self, sample: HandActionSample) -> HandActionSample: ...

    @abstractmethod
    def transform_batch(self, samples: Iterable[HandActionSample]) -> Sequence[HandActionSample]: ...

