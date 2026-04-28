"""JSON-backed data management for EgoDex hand action samples."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from egodex_hand_action.contracts.data import (
    CoordinateSpace,
    DatasetSplit,
    FrameReference,
    HandActionSample,
    Handedness,
    KeypointAnnotation,
    VideoClipReference,
)
from egodex_hand_action.interfaces.dataset import DatasetIndexBuilder, EgoDexDataset


class DatasetIndexError(ValueError):
    """Raised when a dataset index is missing required fields or is malformed."""


class JsonSampleCodec:
    """Convert between JSON dictionaries and core sample dataclasses."""

    @classmethod
    def sample_from_dict(
        cls,
        item: Mapping[str, Any],
        *,
        default_split: DatasetSplit | None = None,
        base_dir: Path | None = None,
    ) -> HandActionSample:
        cls._require_keys(item, ("sample_id",))
        split = cls._parse_split(item.get("split"), default_split)

        frame = cls._frame_from_dict(item.get("frame"), base_dir=base_dir)
        clip = cls._clip_from_dict(item.get("clip"), base_dir=base_dir)
        if frame is None and clip is None:
            raise DatasetIndexError("Sample must define either 'frame' or 'clip'.")

        return HandActionSample(
            sample_id=str(item["sample_id"]),
            split=split,
            frame=frame,
            clip=clip,
            keypoints_25=cls._keypoints_from_dict(item.get("keypoints_25")),
            action_label=cls._optional_str(item.get("action_label")),
            metadata=cls._mapping(item.get("metadata")),
        )

    @classmethod
    def sample_to_dict(
        cls,
        sample: HandActionSample,
        *,
        base_dir: Path | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "split": sample.split.value,
            "metadata": dict(sample.metadata),
        }
        if sample.frame is not None:
            payload["frame"] = cls._frame_to_dict(sample.frame, base_dir=base_dir)
        if sample.clip is not None:
            payload["clip"] = cls._clip_to_dict(sample.clip, base_dir=base_dir)
        if sample.keypoints_25 is not None:
            payload["keypoints_25"] = cls._keypoints_to_dict(sample.keypoints_25)
        if sample.action_label is not None:
            payload["action_label"] = sample.action_label
        return payload

    @staticmethod
    def _parse_split(value: Any, default_split: DatasetSplit | None) -> DatasetSplit:
        if value is None:
            if default_split is None:
                raise DatasetIndexError("Sample is missing required field 'split'.")
            return default_split
        try:
            return DatasetSplit(str(value))
        except ValueError as exc:
            allowed = ", ".join(split.value for split in DatasetSplit)
            raise DatasetIndexError(f"Unsupported split '{value}'. Allowed: {allowed}.") from exc

    @classmethod
    def _frame_from_dict(
        cls,
        value: Any,
        *,
        base_dir: Path | None,
    ) -> FrameReference | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise DatasetIndexError("'frame' must be a JSON object.")
        cls._require_keys(value, ("video_id", "frame_index"))
        return FrameReference(
            video_id=str(value["video_id"]),
            frame_index=int(value["frame_index"]),
            image_path=cls._optional_path(value.get("image_path"), base_dir=base_dir),
            timestamp_ms=cls._optional_float(value.get("timestamp_ms")),
        )

    @classmethod
    def _clip_from_dict(
        cls,
        value: Any,
        *,
        base_dir: Path | None,
    ) -> VideoClipReference | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise DatasetIndexError("'clip' must be a JSON object.")
        cls._require_keys(value, ("video_id", "start_frame", "end_frame"))
        return VideoClipReference(
            video_id=str(value["video_id"]),
            start_frame=int(value["start_frame"]),
            end_frame=int(value["end_frame"]),
            frame_paths=tuple(
                cls._path(path, base_dir=base_dir)
                for path in cls._sequence(value.get("frame_paths"))
            ),
            fps=cls._optional_float(value.get("fps")),
        )

    @classmethod
    def _keypoints_from_dict(cls, value: Any) -> KeypointAnnotation | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise DatasetIndexError("'keypoints_25' must be a JSON object.")
        cls._require_keys(value, ("keypoints", "coordinate_space"))
        keypoints = cls._keypoint_matrix(value["keypoints"])
        visibility_value = value.get("visibility")
        visibility = None
        if visibility_value is not None:
            visibility = tuple(bool(item) for item in cls._sequence(visibility_value))
        return KeypointAnnotation(
            keypoints=keypoints,
            coordinate_space=CoordinateSpace(str(value["coordinate_space"])),
            handedness=Handedness(str(value.get("handedness", Handedness.UNKNOWN.value))),
            visibility=visibility,
            topology_name=cls._optional_str(value.get("topology_name")),
        )

    @staticmethod
    def _frame_to_dict(frame: FrameReference, *, base_dir: Path | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "video_id": frame.video_id,
            "frame_index": frame.frame_index,
        }
        if frame.image_path is not None:
            payload["image_path"] = JsonSampleCodec._serialize_path(frame.image_path, base_dir)
        if frame.timestamp_ms is not None:
            payload["timestamp_ms"] = frame.timestamp_ms
        return payload

    @staticmethod
    def _clip_to_dict(clip: VideoClipReference, *, base_dir: Path | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "video_id": clip.video_id,
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "frame_paths": [
                JsonSampleCodec._serialize_path(path, base_dir)
                for path in clip.frame_paths
            ],
        }
        if clip.fps is not None:
            payload["fps"] = clip.fps
        return payload

    @staticmethod
    def _keypoints_to_dict(annotation: KeypointAnnotation) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "keypoints": [list(point) for point in annotation.keypoints],
            "coordinate_space": annotation.coordinate_space.value,
            "handedness": annotation.handedness.value,
        }
        if annotation.visibility is not None:
            payload["visibility"] = list(annotation.visibility)
        if annotation.topology_name is not None:
            payload["topology_name"] = annotation.topology_name
        return payload

    @staticmethod
    def _require_keys(item: Mapping[str, Any], keys: Iterable[str]) -> None:
        missing = [key for key in keys if key not in item]
        if missing:
            raise DatasetIndexError(f"Missing required field(s): {', '.join(missing)}.")

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise DatasetIndexError("'metadata' must be a JSON object.")
        return dict(value)

    @staticmethod
    def _sequence(value: Any) -> Sequence[Any]:
        if value is None:
            return ()
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise DatasetIndexError("Expected a JSON array.")
        return value

    @staticmethod
    def _keypoint_matrix(value: Any) -> Sequence[Sequence[float]]:
        rows = JsonSampleCodec._sequence(value)
        matrix: list[tuple[float, ...]] = []
        for row in rows:
            values = JsonSampleCodec._sequence(row)
            matrix.append(tuple(float(number) for number in values))
        return tuple(matrix)

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        return None if value is None else str(value)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _optional_path(value: Any, *, base_dir: Path | None) -> Path | None:
        return None if value is None else JsonSampleCodec._path(value, base_dir=base_dir)

    @staticmethod
    def _path(value: Any, *, base_dir: Path | None) -> Path:
        path = Path(str(value))
        if path.is_absolute() or base_dir is None:
            return path
        return base_dir / path

    @staticmethod
    def _serialize_path(path: Path, base_dir: Path | None) -> str:
        if base_dir is None:
            return str(path)
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            return str(path)


class JsonIndexStore:
    """Read and write normalized EgoDex JSON index files."""

    def __init__(self, codec: type[JsonSampleCodec] = JsonSampleCodec) -> None:
        self._codec = codec

    def load_samples(
        self,
        index_path: Path,
        *,
        default_split: DatasetSplit | None = None,
        base_dir: Path | None = None,
    ) -> Sequence[HandActionSample]:
        payload = self._read_json(index_path)
        items = self._extract_items(payload)
        sample_base_dir = base_dir or index_path.parent
        return tuple(
            self._codec.sample_from_dict(
                item,
                default_split=default_split,
                base_dir=sample_base_dir,
            )
            for item in items
        )

    def save_samples(
        self,
        samples: Sequence[HandActionSample],
        index_path: Path,
        *,
        base_dir: Path | None = None,
    ) -> Path:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        sample_base_dir = base_dir or index_path.parent
        payload = {
            "version": 1,
            "samples": [
                self._codec.sample_to_dict(sample, base_dir=sample_base_dir)
                for sample in samples
            ],
        }
        index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return index_path

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            raise DatasetIndexError(f"Dataset index does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _extract_items(payload: Any) -> Sequence[Mapping[str, Any]]:
        if isinstance(payload, Mapping):
            payload = payload.get("samples")
        if not isinstance(payload, Sequence) or isinstance(payload, str):
            raise DatasetIndexError("Dataset index must contain a 'samples' array.")
        items: list[Mapping[str, Any]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                raise DatasetIndexError("Each sample entry must be a JSON object.")
            items.append(item)
        return tuple(items)


class EgoDexJsonDataset(EgoDexDataset):
    """In-memory EgoDex dataset loaded from a normalized JSON index."""

    def __init__(
        self,
        index_path: Path,
        *,
        split: DatasetSplit | None = None,
        store: JsonIndexStore | None = None,
    ) -> None:
        self._index_path = index_path
        self._store = store or JsonIndexStore()
        self._samples = self._store.load_samples(index_path, default_split=split)
        self._split = self._resolve_split(split, self._samples)
        self._validate_samples(self._samples, self._split)

    @property
    def index_path(self) -> Path:
        return self._index_path

    def split(self) -> DatasetSplit:
        return self._split

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def get_sample(self, index: int) -> HandActionSample:
        return self._samples[index]

    @staticmethod
    def _resolve_split(
        requested_split: DatasetSplit | None,
        samples: Sequence[HandActionSample],
    ) -> DatasetSplit:
        if requested_split is not None:
            return requested_split
        if not samples:
            raise DatasetIndexError("Cannot infer split from an empty dataset.")
        splits = {sample.split for sample in samples}
        if len(splits) != 1:
            raise DatasetIndexError("A dataset instance must contain exactly one split.")
        return next(iter(splits))

    @staticmethod
    def _validate_samples(
        samples: Sequence[HandActionSample],
        expected_split: DatasetSplit,
    ) -> None:
        seen_sample_ids: set[str] = set()
        for sample in samples:
            if sample.split != expected_split:
                raise DatasetIndexError(
                    f"Sample '{sample.sample_id}' belongs to split '{sample.split.value}', "
                    f"expected '{expected_split.value}'."
                )
            if sample.sample_id in seen_sample_ids:
                raise DatasetIndexError(f"Duplicate sample_id found: {sample.sample_id}")
            seen_sample_ids.add(sample.sample_id)


class JsonDatasetIndexBuilder(DatasetIndexBuilder):
    """Build split index files from a manifest-style JSON annotation file."""

    DEFAULT_MANIFEST_NAMES = (
        "egodex_index.json",
        "annotations.json",
        "manifest.json",
    )

    def __init__(
        self,
        *,
        manifest_path: Path | None = None,
        store: JsonIndexStore | None = None,
    ) -> None:
        self._manifest_path = manifest_path
        self._store = store or JsonIndexStore()

    def build(self, dataset_root: Path, output_dir: Path) -> Sequence[Path]:
        manifest_path = self._manifest_path or self._find_manifest(dataset_root)
        samples = self._store.load_samples(manifest_path, base_dir=dataset_root)

        grouped: dict[DatasetSplit, list[HandActionSample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.split].append(sample)

        output_paths: list[Path] = []
        for split in DatasetSplit:
            split_samples = tuple(grouped.get(split, ()))
            if not split_samples:
                continue
            output_path = output_dir / f"{split.value}.json"
            self._store.save_samples(split_samples, output_path, base_dir=dataset_root)
            output_paths.append(output_path)
        return tuple(output_paths)

    @classmethod
    def _find_manifest(cls, dataset_root: Path) -> Path:
        for name in cls.DEFAULT_MANIFEST_NAMES:
            candidate = dataset_root / name
            if candidate.exists():
                return candidate
        expected = ", ".join(cls.DEFAULT_MANIFEST_NAMES)
        raise DatasetIndexError(
            f"No dataset manifest found under {dataset_root}. Expected one of: {expected}."
        )
