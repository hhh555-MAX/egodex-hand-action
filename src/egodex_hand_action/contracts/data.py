"""Core data structures for EgoDex hand action prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    UNKNOWN = "unknown"


class CoordinateSpace(str, Enum):
    IMAGE_PIXEL = "image_pixel"
    IMAGE_NORMALIZED = "image_normalized"
    CAMERA_3D = "camera_3d"
    CANONICAL_HAND = "canonical_hand"


@dataclass(frozen=True)
class FrameReference:
    video_id: str
    frame_index: int
    image_path: Path | None = None
    timestamp_ms: float | None = None


@dataclass(frozen=True)
class VideoClipReference:
    video_id: str
    start_frame: int
    end_frame: int
    frame_paths: Sequence[Path] = field(default_factory=tuple)
    fps: float | None = None


@dataclass(frozen=True)
class KeypointAnnotation:
    keypoints: Sequence[Sequence[float]]
    coordinate_space: CoordinateSpace
    handedness: Handedness = Handedness.UNKNOWN
    visibility: Sequence[bool] | None = None
    topology_name: str | None = None


@dataclass(frozen=True)
class HandActionSample:
    sample_id: str
    split: DatasetSplit
    frame: FrameReference | None = None
    clip: VideoClipReference | None = None
    keypoints_25: KeypointAnnotation | None = None
    action_label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KeypointPrediction:
    keypoints: Sequence[Sequence[float]]
    coordinate_space: CoordinateSpace
    topology_name: str
    confidence: Sequence[float] | None = None


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    video_id: str
    method: str
    prediction: KeypointPrediction
    frame_index: int | None = None
    runtime_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

