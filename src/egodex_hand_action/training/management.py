"""Experiment configuration, artifact, and prediction management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from egodex_hand_action.contracts.data import (
    CoordinateSpace,
    KeypointPrediction,
    PredictionRecord,
)
from egodex_hand_action.contracts.experiment import (
    BaselineModelConfig,
    ExperimentConfig,
    PhantomConfig,
    RetargetConfig,
    TrainingConfig,
)


class TrainingManagementError(RuntimeError):
    """Raised when experiment artifacts or configuration cannot be managed."""


@dataclass(frozen=True)
class ExperimentArtifactPaths:
    root_dir: Path
    config_path: Path
    checkpoints_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    logs_dir: Path
    manifest_path: Path


class ExperimentManager:
    """Create and maintain a reproducible experiment artifact layout."""

    CONFIG_FILENAME = "config.json"
    MANIFEST_FILENAME = "manifest.json"

    def prepare(self, config: ExperimentConfig) -> ExperimentArtifactPaths:
        paths = self.paths_for(config)
        for directory in (
            paths.root_dir,
            paths.checkpoints_dir,
            paths.predictions_dir,
            paths.metrics_dir,
            paths.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.save_config(config, paths.config_path)
        self.write_manifest(
            paths,
            {
                "experiment_id": config.experiment_id,
                "status": "prepared",
                "artifacts": {},
            },
        )
        return paths

    def paths_for(self, config: ExperimentConfig) -> ExperimentArtifactPaths:
        root_dir = config.training.output_dir / config.experiment_id
        return ExperimentArtifactPaths(
            root_dir=root_dir,
            config_path=root_dir / self.CONFIG_FILENAME,
            checkpoints_dir=root_dir / "checkpoints",
            predictions_dir=root_dir / "predictions",
            metrics_dir=root_dir / "metrics",
            logs_dir=root_dir / "logs",
            manifest_path=root_dir / self.MANIFEST_FILENAME,
        )

    def save_config(self, config: ExperimentConfig, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_to_jsonable(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def load_config(self, path: Path) -> ExperimentConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TrainingManagementError("Experiment config JSON must be an object.")
        return _experiment_config_from_dict(payload)

    def write_manifest(
        self,
        paths: ExperimentArtifactPaths,
        manifest: Mapping[str, Any],
    ) -> Path:
        paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        paths.manifest_path.write_text(
            json.dumps(_to_jsonable(dict(manifest)), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return paths.manifest_path

    def update_manifest(
        self,
        paths: ExperimentArtifactPaths,
        updates: Mapping[str, Any],
    ) -> Path:
        manifest: dict[str, Any] = {}
        if paths.manifest_path.exists():
            loaded = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                manifest.update(loaded)
        manifest.update(updates)
        return self.write_manifest(paths, manifest)


class JsonPredictionStore:
    """Read and write normalized model prediction records."""

    def save(self, records: Sequence[PredictionRecord], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "predictions": [_prediction_to_dict(record) for record in records],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def load(self, path: Path) -> Sequence[PredictionRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("predictions")
        if not isinstance(payload, Sequence) or isinstance(payload, str):
            raise TrainingManagementError("Prediction JSON must contain a 'predictions' array.")
        return tuple(_prediction_from_dict(item) for item in payload)


def _prediction_to_dict(record: PredictionRecord) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "video_id": record.video_id,
        "frame_index": record.frame_index,
        "method": record.method,
        "runtime_ms": record.runtime_ms,
        "metadata": dict(record.metadata),
        "prediction": {
            "keypoints": [list(point) for point in record.prediction.keypoints],
            "coordinate_space": record.prediction.coordinate_space.value,
            "topology_name": record.prediction.topology_name,
            "confidence": None
            if record.prediction.confidence is None
            else list(record.prediction.confidence),
        },
    }


def _prediction_from_dict(item: Any) -> PredictionRecord:
    if not isinstance(item, Mapping):
        raise TrainingManagementError("Each prediction entry must be an object.")
    prediction = item["prediction"]
    if not isinstance(prediction, Mapping):
        raise TrainingManagementError("'prediction' must be an object.")
    confidence = prediction.get("confidence")
    return PredictionRecord(
        sample_id=str(item["sample_id"]),
        video_id=str(item["video_id"]),
        frame_index=None if item.get("frame_index") is None else int(item["frame_index"]),
        method=str(item["method"]),
        runtime_ms=None if item.get("runtime_ms") is None else float(item["runtime_ms"]),
        metadata=dict(item.get("metadata", {})),
        prediction=KeypointPrediction(
            keypoints=tuple(
                tuple(float(value) for value in point)
                for point in prediction["keypoints"]
            ),
            coordinate_space=CoordinateSpace(str(prediction["coordinate_space"])),
            topology_name=str(prediction["topology_name"]),
            confidence=None if confidence is None else tuple(float(value) for value in confidence),
        ),
    )


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_to_jsonable(item) for item in value]
    return value


def _experiment_config_from_dict(payload: Mapping[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=str(payload["experiment_id"]),
        dataset_root=Path(str(payload["dataset_root"])),
        split_dir=Path(str(payload["split_dir"])),
        training=_training_config_from_dict(payload["training"]),
        baseline=None
        if payload.get("baseline") is None
        else _baseline_config_from_dict(payload["baseline"]),
        phantom=None
        if payload.get("phantom") is None
        else _phantom_config_from_dict(payload["phantom"]),
        retarget=None
        if payload.get("retarget") is None
        else _retarget_config_from_dict(payload["retarget"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _training_config_from_dict(payload: Mapping[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        batch_size=int(payload["batch_size"]),
        max_epochs=int(payload["max_epochs"]),
        learning_rate=float(payload["learning_rate"]),
        seed=int(payload["seed"]),
        output_dir=Path(str(payload["output_dir"])),
        device=str(payload.get("device", "auto")),
        metric_names=tuple(str(value) for value in payload.get("metric_names", ())),
    )


def _baseline_config_from_dict(payload: Mapping[str, Any]) -> BaselineModelConfig:
    return BaselineModelConfig(
        backbone_name=str(payload["backbone_name"]),
        output_keypoint_count=int(payload.get("output_keypoint_count", 25)),
        output_dimension=int(payload.get("output_dimension", 2)),
        pretrained=bool(payload.get("pretrained", True)),
        extra=dict(payload.get("extra", {})),
    )


def _phantom_config_from_dict(payload: Mapping[str, Any]) -> PhantomConfig:
    checkpoint = payload.get("checkpoint_path")
    return PhantomConfig(
        repository_path=Path(str(payload["repository_path"])),
        checkpoint_path=None if checkpoint is None else Path(str(checkpoint)),
        output_keypoint_count=int(payload.get("output_keypoint_count", 21)),
        extra=dict(payload.get("extra", {})),
    )


def _retarget_config_from_dict(payload: Mapping[str, Any]) -> RetargetConfig:
    mapping_path = payload.get("mapping_path")
    return RetargetConfig(
        source_topology_name=str(payload["source_topology_name"]),
        target_topology_name=str(payload["target_topology_name"]),
        source_keypoint_count=int(payload.get("source_keypoint_count", 21)),
        target_keypoint_count=int(payload.get("target_keypoint_count", 25)),
        mapping_path=None if mapping_path is None else Path(str(mapping_path)),
        strategy=str(payload.get("strategy", "rule_based")),
    )
