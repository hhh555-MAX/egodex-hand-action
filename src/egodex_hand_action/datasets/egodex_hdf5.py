"""EgoDex official HDF5/MP4 manifest builder."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from egodex_hand_action.contracts.data import (
    CoordinateSpace,
    DatasetSplit,
    FrameReference,
    HandActionSample,
    Handedness,
    KeypointAnnotation,
)
from egodex_hand_action.datasets.json_index import DatasetIndexError, JsonIndexStore
from egodex_hand_action.interfaces.dataset import DatasetIndexBuilder


EGODEX_LEFT_HAND_25_JOINTS: tuple[str, ...] = (
    "leftHand",
    "leftIndexFingerTip",
    "leftIndexFingerKnuckle",
    "leftMiddleFingerTip",
    "leftMiddleFingerKnuckle",
    "leftRingFingerTip",
    "leftRingFingerKnuckle",
    "leftPinkyFingerTip",
    "leftPinkyFingerKnuckle",
    "leftThumbTip",
    "leftThumbBase",
    "leftThumbProximal",
    "leftThumbIntermediate",
    "leftThumbMetacarpal",
    "leftThumbMetacarpalBase",
    "leftThumbMetacarpalProximal",
    "leftThumbMetacarpalIntermediate",
    "leftThumbMetacarpalMetacarpal",
    "leftThumbMetacarpalMetacarpalBase",
    "leftThumbMetacarpalMetacarpalProximal",
    "leftThumbMetacarpalMetacarpalIntermediate",
    "leftThumbMetacarpalMetacarpalMetacarpal",
    "leftThumbMetacarpalMetacarpalMetacarpalBase",
    "leftThumbMetacarpalMetacarpalMetacarpalProximal",
    "leftThumbMetacarpalMetacarpalMetacarpalIntermediate",
)


@dataclass(frozen=True)
class EgoDexHdf5ManifestConfig:
    hand: Handedness = Handedness.LEFT
    frame_root: Path | None = None
    frame_extension: str = ".jpg"
    topology_name: str = "egodex_25"
    coordinate_space: CoordinateSpace = CoordinateSpace.CAMERA_3D
    joint_names: Sequence[str] | None = None
    manifest_filename: str = "manifest.json"
    write_split_files: bool = True


class EgoDexHdf5ManifestBuilder(DatasetIndexBuilder):
    """Build normalized manifest/split files from official EgoDex HDF5 and MP4 pairs."""

    def __init__(
        self,
        config: EgoDexHdf5ManifestConfig | None = None,
        *,
        store: JsonIndexStore | None = None,
    ) -> None:
        self._config = config or EgoDexHdf5ManifestConfig()
        self._store = store or JsonIndexStore()

    def build(self, dataset_root: Path, output_dir: Path) -> Sequence[Path]:
        h5py = self._import_h5py()
        pairs = tuple(_iter_hdf5_mp4_pairs(dataset_root))
        if not pairs:
            raise DatasetIndexError(f"No paired .hdf5/.mp4 files found under {dataset_root}.")

        samples: list[HandActionSample] = []
        for hdf5_path, mp4_path in pairs:
            samples.extend(
                self._samples_from_pair(
                    h5py=h5py,
                    dataset_root=dataset_root,
                    hdf5_path=hdf5_path,
                    mp4_path=mp4_path,
                )
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._store.save_samples(
            tuple(samples),
            output_dir / self._config.manifest_filename,
            base_dir=dataset_root,
        )
        output_paths = [manifest_path]

        if self._config.write_split_files:
            for split in DatasetSplit:
                split_samples = tuple(sample for sample in samples if sample.split == split)
                if not split_samples:
                    continue
                output_paths.append(
                    self._store.save_samples(
                        split_samples,
                        output_dir / f"{split.value}.json",
                        base_dir=dataset_root,
                    )
                )

        return tuple(output_paths)

    def _samples_from_pair(
        self,
        *,
        h5py: Any,
        dataset_root: Path,
        hdf5_path: Path,
        mp4_path: Path,
    ) -> Sequence[HandActionSample]:
        video_id = _video_id(dataset_root, hdf5_path)
        split = _split_from_path(dataset_root, hdf5_path)
        task = hdf5_path.parent.name
        part = hdf5_path.parent.parent.name
        joint_names = self._joint_names()

        with h5py.File(hdf5_path, "r") as file:
            transforms = file.get("transforms")
            if transforms is None:
                raise DatasetIndexError(f"HDF5 file has no 'transforms' group: {hdf5_path}")
            resolved_joint_names = _resolve_joint_names(joint_names, transforms.keys(), hdf5_path)
            frame_count = _frame_count(transforms, resolved_joint_names, hdf5_path)
            descriptions = _description_attrs(file.attrs)
            confidence_values = _confidence_values(file, resolved_joint_names, frame_count)

            samples = []
            for frame_index in range(frame_count):
                keypoints = tuple(
                    _translation_from_transform(transforms[joint_name][frame_index])
                    for joint_name in resolved_joint_names
                )
                confidence = None
                if confidence_values is not None:
                    confidence = tuple(values[frame_index] for values in confidence_values)

                samples.append(
                    HandActionSample(
                        sample_id=f"{video_id}_{self._config.hand.value}_frame{frame_index:06d}",
                        split=split,
                        frame=FrameReference(
                            video_id=video_id,
                            frame_index=frame_index,
                            image_path=self._image_path(video_id, frame_index),
                        ),
                        keypoints_25=KeypointAnnotation(
                            keypoints=keypoints,
                            coordinate_space=self._config.coordinate_space,
                            handedness=self._config.hand,
                            visibility=None if confidence is None else tuple(value > 0.0 for value in confidence),
                            topology_name=self._config.topology_name,
                        ),
                        metadata={
                            "source_hdf5": str(hdf5_path.relative_to(dataset_root)),
                            "source_mp4": str(mp4_path.relative_to(dataset_root)),
                            "task": task,
                            "part": part,
                            "joint_names": tuple(joint_names),
                            "resolved_joint_names": tuple(resolved_joint_names),
                            "confidences": confidence,
                            **descriptions,
                        },
                    )
                )
        return tuple(samples)

    def _joint_names(self) -> Sequence[str]:
        if self._config.joint_names is not None:
            return tuple(self._config.joint_names)
        if self._config.hand == Handedness.LEFT:
            return EGODEX_LEFT_HAND_25_JOINTS
        if self._config.hand == Handedness.RIGHT:
            return tuple(_left_to_right(name) for name in EGODEX_LEFT_HAND_25_JOINTS)
        raise DatasetIndexError("EgoDex HDF5 manifest builder currently supports left or right hand.")

    def _image_path(self, video_id: str, frame_index: int) -> Path | None:
        if self._config.frame_root is None:
            return None
        extension = self._config.frame_extension
        if not extension.startswith("."):
            extension = f".{extension}"
        return self._config.frame_root / video_id / f"{frame_index:06d}{extension}"

    @staticmethod
    def _import_h5py():
        try:
            return importlib.import_module("h5py")
        except ImportError as exc:
            raise DatasetIndexError(
                "EgoDexHdf5ManifestBuilder requires h5py. Install it with 'pip install h5py'."
            ) from exc


def _iter_hdf5_mp4_pairs(dataset_root: Path) -> Sequence[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for hdf5_path in sorted(dataset_root.rglob("*.hdf5")):
        mp4_path = hdf5_path.with_suffix(".mp4")
        if mp4_path.exists():
            pairs.append((hdf5_path, mp4_path))
    return tuple(pairs)


def _split_from_path(dataset_root: Path, path: Path) -> DatasetSplit:
    relative_parts = path.relative_to(dataset_root).parts
    if not relative_parts:
        return DatasetSplit.TRAIN
    split_name = relative_parts[0].lower()
    if split_name == "test":
        return DatasetSplit.TEST
    if split_name in {"val", "valid", "validation"}:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TRAIN


def _video_id(dataset_root: Path, hdf5_path: Path) -> str:
    relative = hdf5_path.relative_to(dataset_root).with_suffix("")
    return "_".join(relative.parts)


def _left_to_right(name: str) -> str:
    if name.startswith("left"):
        return "right" + name[len("left") :]
    return name


def _resolve_joint_names(
    expected_names: Sequence[str],
    available_names: Any,
    hdf5_path: Path,
) -> Sequence[str]:
    available = set(str(name) for name in available_names)
    resolved = []
    missing = []
    for name in expected_names:
        if name in available:
            resolved.append(name)
            continue
        alias = _thumb_typo_alias(name)
        if alias in available:
            resolved.append(alias)
            continue
        missing.append(name)
    if missing:
        raise DatasetIndexError(
            f"HDF5 file is missing expected EgoDex joint(s): {', '.join(missing)}. "
            f"Available transform names in {hdf5_path}: {', '.join(sorted(available))}."
        )
    return tuple(resolved)


def _thumb_typo_alias(name: str) -> str:
    if "Thumb" in name:
        return name.replace("Thumb", "Thunmb")
    if "Thunmb" in name:
        return name.replace("Thunmb", "Thumb")
    return name


def _frame_count(transforms: Any, joint_names: Sequence[str], hdf5_path: Path) -> int:
    counts = []
    for joint_name in joint_names:
        shape = transforms[joint_name].shape
        if len(shape) != 3 or shape[1:] != (4, 4):
            raise DatasetIndexError(
                f"Transform '{joint_name}' in {hdf5_path} must have shape N x 4 x 4, got {shape}."
            )
        counts.append(int(shape[0]))
    if len(set(counts)) != 1:
        raise DatasetIndexError(f"Joint transform frame counts do not match in {hdf5_path}: {counts}.")
    return counts[0]


def _translation_from_transform(transform: Any) -> tuple[float, float, float]:
    return (
        float(transform[0, 3]),
        float(transform[1, 3]),
        float(transform[2, 3]),
    )


def _confidence_values(
    file: Any,
    joint_names: Sequence[str],
    frame_count: int,
) -> Sequence[Sequence[float]] | None:
    confidences = file.get("confidences")
    if confidences is None:
        return None
    values = []
    for joint_name in joint_names:
        if joint_name not in confidences:
            alias = _thumb_typo_alias(joint_name)
            if alias not in confidences:
                return None
            joint_name = alias
        dataset = confidences[joint_name]
        if len(dataset.shape) != 1 or int(dataset.shape[0]) != frame_count:
            return None
        values.append(tuple(float(value) for value in dataset[:]))
    return tuple(values)


def _description_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("llm_description", "llm_description2", "which_llm_description")
    result = {}
    for key in keys:
        if key in attrs:
            result[key] = _decode_attr(attrs[key])
    return result


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value.item() if hasattr(value, "item") else value
