"""Prediction orchestration for hand action models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from egodex_hand_action.contracts.data import (
    DatasetSplit,
    FrameReference,
    HandActionSample,
    PredictionRecord,
    VideoClipReference,
)
from egodex_hand_action.interfaces.model import HandActionModel
from egodex_hand_action.interfaces.inference import Predictor
from egodex_hand_action.training.management import JsonPredictionStore


class InferenceError(RuntimeError):
    """Raised when inference input or model output is invalid."""


@dataclass(frozen=True)
class VideoFrameSampleBuilder:
    """Build inference samples from a frame directory or single image file."""

    image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    clip_size: int | None = None
    stride: int = 1
    split: DatasetSplit = DatasetSplit.TEST

    def build(self, video_path: Path) -> Sequence[HandActionSample]:
        if video_path.is_file():
            return (self._single_image_sample(video_path),)
        if video_path.is_dir():
            frame_paths = self._frame_paths(video_path)
            if self.clip_size is None:
                return self._frame_samples(video_path.name, frame_paths)
            return self._clip_samples(video_path.name, frame_paths)
        raise InferenceError(f"Inference path does not exist: {video_path}")

    def _single_image_sample(self, image_path: Path) -> HandActionSample:
        return HandActionSample(
            sample_id=image_path.stem,
            split=self.split,
            frame=FrameReference(
                video_id=image_path.stem,
                frame_index=0,
                image_path=image_path,
            ),
        )

    def _frame_samples(
        self,
        video_id: str,
        frame_paths: Sequence[Path],
    ) -> Sequence[HandActionSample]:
        return tuple(
            HandActionSample(
                sample_id=f"{video_id}_{index:06d}",
                split=self.split,
                frame=FrameReference(
                    video_id=video_id,
                    frame_index=index,
                    image_path=path,
                ),
            )
            for index, path in enumerate(frame_paths)
        )

    def _clip_samples(
        self,
        video_id: str,
        frame_paths: Sequence[Path],
    ) -> Sequence[HandActionSample]:
        if self.clip_size is None:
            raise InferenceError("clip_size must be set to build clip samples.")
        if self.clip_size <= 0:
            raise InferenceError("clip_size must be positive.")
        if self.stride <= 0:
            raise InferenceError("stride must be positive.")

        samples: list[HandActionSample] = []
        for start in range(0, len(frame_paths), self.stride):
            end = min(start + self.clip_size, len(frame_paths))
            if start >= end:
                continue
            clip_paths = tuple(frame_paths[start:end])
            samples.append(
                HandActionSample(
                    sample_id=f"{video_id}_clip_{start:06d}_{end - 1:06d}",
                    split=self.split,
                    clip=VideoClipReference(
                        video_id=video_id,
                        start_frame=start,
                        end_frame=end - 1,
                        frame_paths=clip_paths,
                    ),
                )
            )
            if end == len(frame_paths):
                break
        return tuple(samples)

    def _frame_paths(self, directory: Path) -> Sequence[Path]:
        extensions = {extension.lower() for extension in self.image_extensions}
        frame_paths = tuple(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            )
        )
        if not frame_paths:
            raise InferenceError(f"No image frames found under directory: {directory}")
        return frame_paths


class SimplePredictor(Predictor):
    """Run model inference over samples and persist normalized predictions."""

    def __init__(
        self,
        *,
        prediction_store: JsonPredictionStore | None = None,
        video_sample_builder: VideoFrameSampleBuilder | None = None,
    ) -> None:
        self._prediction_store = prediction_store or JsonPredictionStore()
        self._video_sample_builder = video_sample_builder or VideoFrameSampleBuilder()

    def predict_samples(
        self,
        model: HandActionModel,
        samples: Sequence[HandActionSample],
    ) -> Sequence[PredictionRecord]:
        if not samples:
            return ()
        records = model.predict(samples)
        self._validate_records(samples, records)
        return tuple(records)

    def predict_video(
        self,
        model: HandActionModel,
        video_path: Path,
    ) -> Sequence[PredictionRecord]:
        samples = self._video_sample_builder.build(video_path)
        return self.predict_samples(model, samples)

    def export_predictions(
        self,
        records: Sequence[PredictionRecord],
        output_path: Path,
    ) -> Path:
        return self._prediction_store.save(records, output_path)

    @staticmethod
    def _validate_records(
        samples: Sequence[HandActionSample],
        records: Sequence[PredictionRecord],
    ) -> None:
        if len(samples) != len(records):
            raise InferenceError(
                f"Model returned {len(records)} prediction records for {len(samples)} samples."
            )
        sample_ids = {sample.sample_id for sample in samples}
        record_ids = {record.sample_id for record in records}
        missing = sample_ids - record_ids
        extra = record_ids - sample_ids
        if missing or extra:
            raise InferenceError(
                "Prediction sample IDs do not match input samples. "
                f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
            )
