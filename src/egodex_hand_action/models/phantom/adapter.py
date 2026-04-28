"""JSON command adapter for integrating the external Phantom project."""

from __future__ import annotations

import json
import os
import importlib
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, Mapping, Sequence

from egodex_hand_action.contracts.data import (
    CoordinateSpace,
    HandActionSample,
    KeypointPrediction,
    PredictionRecord,
)
from egodex_hand_action.contracts.experiment import PhantomConfig
from egodex_hand_action.interfaces.model import PhantomAdapter


class PhantomAdapterError(RuntimeError):
    """Raised when Phantom input preparation or external execution fails."""


@dataclass(frozen=True)
class PhantomCommandConfig:
    repository_path: Path
    command_template: Sequence[str]
    checkpoint_path: Path | None = None
    scratch_dir: Path | None = None
    timeout_seconds: float | None = None
    output_keypoint_count: int = 21
    output_dimension: int = 2
    input_coordinate_space: CoordinateSpace = CoordinateSpace.IMAGE_NORMALIZED
    output_coordinate_space: CoordinateSpace = CoordinateSpace.IMAGE_NORMALIZED
    topology_name: str = "phantom_21"
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PhantomProcessDataConfig:
    repository_path: Path
    demo_name: str
    data_root_dir: Path
    processed_data_root_dir: Path
    mode: str = "hand3d"
    target_hand: str = "left"
    bimanual_setup: str = "single_arm"
    demo_num: str | None = None
    config_name: str = "default"
    n_processes: int = 1
    timeout_seconds: float | None = None
    output_keypoint_count: int = 21
    output_dimension: int = 3
    output_coordinate_space: CoordinateSpace = CoordinateSpace.CAMERA_3D
    topology_name: str = "phantom_21"
    hand_data_path_template: str = (
        "{processed_data_root_dir}/{demo_name}/{data_sub_folder}/hand_processor/hand_data_{target_hand}.npz"
    )
    environment: Mapping[str, str] = field(default_factory=dict)


class PhantomJsonAdapter(PhantomAdapter):
    """Call Phantom through a normalized JSON input/output contract."""

    INPUT_FILENAME = "{sample_id}_phantom_input.json"
    OUTPUT_FILENAME = "{sample_id}_phantom_output.json"

    def __init__(self, config: PhantomCommandConfig) -> None:
        self._config = config
        self._validate_config(config)

    @classmethod
    def from_phantom_config(cls, config: PhantomConfig) -> "PhantomJsonAdapter":
        command_template = config.extra.get("command_template")
        if command_template is None:
            command_template = ()
        return cls(
            PhantomCommandConfig(
                repository_path=config.repository_path,
                checkpoint_path=config.checkpoint_path,
                command_template=tuple(str(part) for part in command_template),
                scratch_dir=cls._optional_path(config.extra.get("scratch_dir")),
                timeout_seconds=cls._optional_float(config.extra.get("timeout_seconds")),
                output_keypoint_count=config.output_keypoint_count,
                output_dimension=int(config.extra.get("output_dimension", 2)),
                input_coordinate_space=CoordinateSpace(
                    str(config.extra.get("input_coordinate_space", CoordinateSpace.IMAGE_NORMALIZED.value))
                ),
                output_coordinate_space=CoordinateSpace(
                    str(config.extra.get("output_coordinate_space", CoordinateSpace.IMAGE_NORMALIZED.value))
                ),
                topology_name=str(config.extra.get("topology_name", "phantom_21")),
                environment={
                    str(key): str(value)
                    for key, value in dict(config.extra.get("environment", {})).items()
                },
            )
        )

    def prepare_input(self, sample: HandActionSample) -> Mapping[str, Any]:
        image_paths = self._image_paths(sample)
        if not image_paths:
            raise PhantomAdapterError(f"Sample '{sample.sample_id}' has no image path for Phantom.")
        payload: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "video_id": self._video_id(sample),
            "frame_index": sample.frame.frame_index if sample.frame is not None else None,
            "image_paths": [str(path) for path in image_paths],
            "coordinate_space": self._config.input_coordinate_space.value,
            "metadata": dict(sample.metadata),
        }
        if self._config.checkpoint_path is not None:
            payload["checkpoint_path"] = str(self._config.checkpoint_path)
        return payload

    def predict_21_keypoints(self, sample: HandActionSample) -> PredictionRecord:
        started_at = time.perf_counter()
        scratch_dir = self._scratch_dir()
        scratch_dir.mkdir(parents=True, exist_ok=True)

        input_path = scratch_dir / self.INPUT_FILENAME.format(sample_id=sample.sample_id)
        output_path = scratch_dir / self.OUTPUT_FILENAME.format(sample_id=sample.sample_id)

        self._write_json(input_path, self.prepare_input(sample))
        self._run_phantom(input_path=input_path, output_path=output_path, sample=sample)
        payload = self._read_json(output_path)
        prediction = self._prediction_from_payload(payload)

        return PredictionRecord(
            sample_id=sample.sample_id,
            video_id=self._video_id(sample),
            frame_index=sample.frame.frame_index if sample.frame is not None else None,
            method="phantom",
            prediction=prediction,
            runtime_ms=(time.perf_counter() - started_at) * 1000.0,
            metadata={
                "phantom_input_path": str(input_path),
                "phantom_output_path": str(output_path),
            },
        )

    def _run_phantom(
        self,
        *,
        input_path: Path,
        output_path: Path,
        sample: HandActionSample,
    ) -> None:
        if not self._config.command_template:
            raise PhantomAdapterError(
                "Phantom command_template is required. "
                "Use placeholders such as {input_json}, {output_json}, {checkpoint_path}, "
                "and {repository_path}."
            )
        command = self._render_command(
            input_path=input_path,
            output_path=output_path,
            sample=sample,
        )
        result = subprocess.run(
            command,
            cwd=self._config.repository_path,
            env={**os.environ, **self._config.environment},
            capture_output=True,
            text=True,
            timeout=self._config.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise PhantomAdapterError(
                "Phantom command failed with exit code "
                f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        if not output_path.exists():
            raise PhantomAdapterError(f"Phantom did not create output JSON: {output_path}")

    def _render_command(
        self,
        *,
        input_path: Path,
        output_path: Path,
        sample: HandActionSample,
    ) -> Sequence[str]:
        values = {
            "input_json": str(input_path),
            "output_json": str(output_path),
            "checkpoint_path": str(self._config.checkpoint_path or ""),
            "repository_path": str(self._config.repository_path),
            "sample_id": sample.sample_id,
            "video_id": self._video_id(sample),
        }
        return tuple(self._format_part(part, values) for part in self._config.command_template)

    def _prediction_from_payload(self, payload: Mapping[str, Any]) -> KeypointPrediction:
        keypoints = self._keypoints_from_payload(payload)
        confidence = payload.get("confidence")
        if confidence is not None:
            confidence = tuple(float(value) for value in self._sequence(confidence))
        return KeypointPrediction(
            keypoints=keypoints,
            coordinate_space=CoordinateSpace(
                str(payload.get("coordinate_space", self._config.output_coordinate_space.value))
            ),
            topology_name=str(payload.get("topology_name", self._config.topology_name)),
            confidence=confidence,
        )

    def _keypoints_from_payload(self, payload: Mapping[str, Any]) -> Sequence[Sequence[float]]:
        if "keypoints_21" in payload:
            raw_keypoints = payload["keypoints_21"]
        elif "keypoints" in payload:
            raw_keypoints = payload["keypoints"]
        else:
            raise PhantomAdapterError("Phantom output must contain 'keypoints_21' or 'keypoints'.")

        keypoints = tuple(
            tuple(float(value) for value in self._sequence(point))
            for point in self._sequence(raw_keypoints)
        )
        if len(keypoints) != self._config.output_keypoint_count:
            raise PhantomAdapterError(
                f"Expected {self._config.output_keypoint_count} Phantom keypoints, "
                f"got {len(keypoints)}."
            )
        for point in keypoints:
            if len(point) < self._config.output_dimension:
                raise PhantomAdapterError(
                    f"Each Phantom keypoint must contain at least {self._config.output_dimension} values."
                )
        return keypoints

    def _scratch_dir(self) -> Path:
        if self._config.scratch_dir is not None:
            return self._config.scratch_dir
        return self._config.repository_path / ".egodex_phantom"

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> Mapping[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise PhantomAdapterError("Phantom output JSON must be an object.")
        return payload

    @staticmethod
    def _image_paths(sample: HandActionSample) -> Sequence[Path]:
        if sample.clip is not None and sample.clip.frame_paths:
            return tuple(sample.clip.frame_paths)
        if sample.frame is not None and sample.frame.image_path is not None:
            return (sample.frame.image_path,)
        return ()

    @staticmethod
    def _video_id(sample: HandActionSample) -> str:
        if sample.frame is not None:
            return sample.frame.video_id
        if sample.clip is not None:
            return sample.clip.video_id
        raise PhantomAdapterError(f"Sample '{sample.sample_id}' has neither frame nor clip reference.")

    @staticmethod
    def _sequence(value: Any) -> Sequence[Any]:
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise PhantomAdapterError("Expected a JSON array.")
        return value

    @staticmethod
    def _format_part(template: str, values: Mapping[str, str]) -> str:
        fields = {name for _, name, _, _ in Formatter().parse(template) if name}
        unknown = fields - values.keys()
        if unknown:
            raise PhantomAdapterError(f"Unknown Phantom command placeholder(s): {', '.join(sorted(unknown))}")
        return template.format(**values)

    @staticmethod
    def _validate_config(config: PhantomCommandConfig) -> None:
        if not config.repository_path.exists():
            raise PhantomAdapterError(f"Phantom repository path does not exist: {config.repository_path}")
        if config.checkpoint_path is not None and not config.checkpoint_path.exists():
            raise PhantomAdapterError(f"Phantom checkpoint path does not exist: {config.checkpoint_path}")
        if config.output_keypoint_count <= 0:
            raise PhantomAdapterError("output_keypoint_count must be positive.")
        if config.output_dimension <= 0:
            raise PhantomAdapterError("output_dimension must be positive.")

    @staticmethod
    def _optional_path(value: Any) -> Path | None:
        return None if value is None else Path(str(value))

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)


class PhantomProcessDataAdapter(PhantomAdapter):
    """Integrate the official Phantom process_data.py pipeline and HandSequence npz output."""

    def __init__(self, config: PhantomProcessDataConfig) -> None:
        self._config = config
        self._validate_config(config)

    @classmethod
    def from_phantom_config(cls, config: PhantomConfig) -> "PhantomProcessDataAdapter":
        extra = dict(config.extra)
        return cls(
            PhantomProcessDataConfig(
                repository_path=config.repository_path,
                demo_name=str(extra["demo_name"]),
                data_root_dir=Path(str(extra["data_root_dir"])),
                processed_data_root_dir=Path(str(extra["processed_data_root_dir"])),
                mode=str(extra.get("mode", "hand3d")),
                target_hand=str(extra.get("target_hand", "left")),
                bimanual_setup=str(extra.get("bimanual_setup", "single_arm")),
                demo_num=None if extra.get("demo_num") is None else str(extra["demo_num"]),
                config_name=str(extra.get("config_name", "default")),
                n_processes=int(extra.get("n_processes", 1)),
                timeout_seconds=PhantomJsonAdapter._optional_float(extra.get("timeout_seconds")),
                output_keypoint_count=config.output_keypoint_count,
                output_dimension=int(extra.get("output_dimension", 3)),
                output_coordinate_space=CoordinateSpace(
                    str(extra.get("output_coordinate_space", CoordinateSpace.CAMERA_3D.value))
                ),
                topology_name=str(extra.get("topology_name", "phantom_21")),
                hand_data_path_template=str(
                    extra.get(
                        "hand_data_path_template",
                        PhantomProcessDataConfig.hand_data_path_template,
                    )
                ),
                environment={
                    str(key): str(value)
                    for key, value in dict(extra.get("environment", {})).items()
                },
            )
        )

    def prepare_input(self, sample: HandActionSample) -> Mapping[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "video_id": PhantomJsonAdapter._video_id(sample),
            "frame_index": _sample_frame_index(sample),
            "demo_name": self._demo_name(sample),
            "demo_num": self._demo_num(sample),
            "hand_data_path": str(self._hand_data_path(sample)),
        }

    def run_processing(self) -> None:
        command = self.command()
        result = subprocess.run(
            command,
            cwd=self._config.repository_path,
            env={**os.environ, **self._config.environment},
            capture_output=True,
            text=True,
            timeout=self._config.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise PhantomAdapterError(
                "Phantom process_data.py failed with exit code "
                f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    def command(self) -> Sequence[str]:
        command = [
            "python",
            "process_data.py",
        ]
        if self._config.config_name != "default":
            command.append(f"--config-name={self._config.config_name}")
        command.extend(
            [
            f"demo_name={self._config.demo_name}",
            f"data_root_dir={self._config.data_root_dir}",
            f"processed_data_root_dir={self._config.processed_data_root_dir}",
            f"mode={self._config.mode}",
            f"target_hand={self._config.target_hand}",
            f"bimanual_setup={self._config.bimanual_setup}",
            f"n_processes={self._config.n_processes}",
            ]
        )
        if self._config.demo_num is not None:
            command.append(f"demo_num={self._config.demo_num}")
        return tuple(command)

    def predict_21_keypoints(self, sample: HandActionSample) -> PredictionRecord:
        hand_data_path = self._hand_data_path(sample)
        return self._record_from_npz(sample=sample, hand_data_path=hand_data_path)

    def load_predictions(
        self,
        samples: Sequence[HandActionSample],
    ) -> Sequence[PredictionRecord]:
        return tuple(self.predict_21_keypoints(sample) for sample in samples)

    def _record_from_npz(
        self,
        *,
        sample: HandActionSample,
        hand_data_path: Path,
    ) -> PredictionRecord:
        np = self._import_numpy()
        if not hand_data_path.exists():
            raise PhantomAdapterError(f"Phantom hand data npz does not exist: {hand_data_path}")
        data = np.load(hand_data_path, allow_pickle=True)
        field_name = "kpts_3d" if self._config.output_dimension == 3 else "kpts_2d"
        if field_name not in data:
            raise PhantomAdapterError(
                f"Phantom hand data file {hand_data_path} does not contain '{field_name}'. "
                f"Available fields: {', '.join(data.files)}."
            )
        keypoints_by_frame = data[field_name]
        frame_indices = data["frame_indices"] if "frame_indices" in data else np.arange(len(keypoints_by_frame))
        hand_detected = data["hand_detected"] if "hand_detected" in data else None
        frame_index = _sample_frame_index(sample)
        row_index = self._row_index_for_frame(frame_indices, frame_index)
        keypoints = tuple(
            tuple(float(value) for value in point)
            for point in keypoints_by_frame[row_index]
        )
        if len(keypoints) != self._config.output_keypoint_count:
            raise PhantomAdapterError(
                f"Expected {self._config.output_keypoint_count} Phantom keypoints, got {len(keypoints)}."
            )
        confidence = None
        if hand_detected is not None:
            detected = bool(hand_detected[row_index])
            confidence = tuple(1.0 if detected else 0.0 for _ in keypoints)

        return PredictionRecord(
            sample_id=sample.sample_id,
            video_id=PhantomJsonAdapter._video_id(sample),
            frame_index=frame_index,
            method=f"phantom_{self._config.mode}",
            prediction=KeypointPrediction(
                keypoints=keypoints,
                coordinate_space=self._config.output_coordinate_space,
                topology_name=self._config.topology_name,
                confidence=confidence,
            ),
            metadata={
                "phantom_hand_data_path": str(hand_data_path),
                "phantom_field": field_name,
                "phantom_row_index": row_index,
                "target_hand": self._config.target_hand,
            },
        )

    def _hand_data_path(self, sample: HandActionSample) -> Path:
        metadata_path = sample.metadata.get("phantom_hand_data_path")
        if metadata_path is not None:
            return Path(str(metadata_path))
        data_sub_folder = self._data_sub_folder(sample)
        values = {
            "processed_data_root_dir": str(self._config.processed_data_root_dir),
            "demo_name": self._demo_name(sample),
            "demo_num": self._demo_num(sample),
            "data_sub_folder": data_sub_folder,
            "target_hand": self._config.target_hand,
            "mode": self._config.mode,
            "video_id": PhantomJsonAdapter._video_id(sample),
            "sample_id": sample.sample_id,
        }
        candidate = Path(self._config.hand_data_path_template.format(**values))
        if candidate.exists():
            return candidate
        discovered = self._discover_hand_data_path(sample, data_sub_folder)
        return discovered or candidate

    def _demo_name(self, sample: HandActionSample) -> str:
        return str(sample.metadata.get("phantom_demo_name", self._config.demo_name))

    def _demo_num(self, sample: HandActionSample) -> str:
        if sample.metadata.get("phantom_demo_num") is not None:
            return str(sample.metadata["phantom_demo_num"])
        if self._config.demo_num is not None:
            return self._config.demo_num
        return PhantomJsonAdapter._video_id(sample)

    def _data_sub_folder(self, sample: HandActionSample) -> str:
        if sample.metadata.get("phantom_data_sub_folder") is not None:
            return str(sample.metadata["phantom_data_sub_folder"])
        return self._demo_num(sample)

    def _discover_hand_data_path(
        self,
        sample: HandActionSample,
        data_sub_folder: str,
    ) -> Path | None:
        demo_root = self._config.processed_data_root_dir / self._demo_name(sample)
        filename = f"hand_data_{self._config.target_hand}.npz"
        preferred_suffixes = (
            Path(data_sub_folder) / "hand_processor" / filename,
            Path(PhantomJsonAdapter._video_id(sample)) / "hand_processor" / filename,
        )
        for suffix in preferred_suffixes:
            candidate = demo_root / suffix
            if candidate.exists():
                return candidate

        matches = tuple(sorted(demo_root.glob(f"**/hand_processor/{filename}")))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise PhantomAdapterError(
                "Multiple Phantom hand data files were found. "
                "Set sample.metadata['phantom_hand_data_path'] or "
                "sample.metadata['phantom_data_sub_folder'] to disambiguate. "
                f"Matches: {', '.join(str(path) for path in matches[:10])}"
            )
        return None

    @staticmethod
    def _row_index_for_frame(frame_indices: Any, frame_index: int) -> int:
        for index, value in enumerate(frame_indices):
            if int(value) == frame_index:
                return index
        raise PhantomAdapterError(f"Frame index {frame_index} not found in Phantom hand data.")

    @staticmethod
    def _import_numpy():
        try:
            return importlib.import_module("numpy")
        except ImportError as exc:
            raise PhantomAdapterError("Phantom npz parsing requires numpy.") from exc

    @staticmethod
    def _validate_config(config: PhantomProcessDataConfig) -> None:
        if not config.repository_path.exists():
            raise PhantomAdapterError(f"Phantom repository path does not exist: {config.repository_path}")
        if config.output_keypoint_count <= 0:
            raise PhantomAdapterError("output_keypoint_count must be positive.")
        if config.output_dimension not in {2, 3}:
            raise PhantomAdapterError("output_dimension must be 2 or 3.")


def _sample_frame_index(sample: HandActionSample) -> int:
    if sample.frame is not None:
        return sample.frame.frame_index
    if sample.clip is not None:
        return sample.clip.start_frame
    raise PhantomAdapterError(f"Sample '{sample.sample_id}' has neither frame nor clip reference.")
