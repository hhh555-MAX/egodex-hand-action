"""Experiment and module configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BaselineModelConfig:
    backbone_name: str
    output_keypoint_count: int = 25
    output_dimension: int = 2
    pretrained: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhantomConfig:
    repository_path: Path
    checkpoint_path: Path | None = None
    output_keypoint_count: int = 21
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetargetConfig:
    source_topology_name: str
    target_topology_name: str
    source_keypoint_count: int = 21
    target_keypoint_count: int = 25
    mapping_path: Path | None = None
    strategy: str = "rule_based"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    max_epochs: int
    learning_rate: float
    seed: int
    output_dir: Path
    device: str = "auto"
    metric_names: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvaluationConfig:
    prediction_dirs: Sequence[Path]
    ground_truth_path: Path
    output_dir: Path
    metric_names: Sequence[str]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    dataset_root: Path
    split_dir: Path
    training: TrainingConfig
    baseline: BaselineModelConfig | None = None
    phantom: PhantomConfig | None = None
    retarget: RetargetConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

