"""Sample-level preprocessing for EgoDex hand action data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from egodex_hand_action.contracts.data import (
    CoordinateSpace,
    FrameReference,
    HandActionSample,
    KeypointAnnotation,
    VideoClipReference,
)
from egodex_hand_action.interfaces.dataset import Preprocessor


class PreprocessingError(ValueError):
    """Raised when a sample cannot be preprocessed with the requested transform."""


class SampleTransform(Protocol):
    def transform_sample(self, sample: HandActionSample) -> HandActionSample: ...


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise PreprocessingError("Image width and height must be positive.")


@dataclass(frozen=True)
class ClipSamplingConfig:
    num_frames: int
    stride: int = 1
    include_end_frame: bool = True

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise PreprocessingError("num_frames must be positive.")
        if self.stride <= 0:
            raise PreprocessingError("stride must be positive.")


class KeypointNormalizer:
    """Normalize or denormalize 2D image keypoints using image dimensions."""

    IMAGE_SIZE_METADATA_KEY = "image_size"

    def __init__(
        self,
        *,
        target_space: CoordinateSpace = CoordinateSpace.IMAGE_NORMALIZED,
        image_size: ImageSize | None = None,
        metadata_key: str = IMAGE_SIZE_METADATA_KEY,
    ) -> None:
        self._target_space = target_space
        self._image_size = image_size
        self._metadata_key = metadata_key

    def transform_sample(self, sample: HandActionSample) -> HandActionSample:
        if sample.keypoints_25 is None:
            return sample
        transformed = self.transform_annotation(
            sample.keypoints_25,
            image_size=self._resolve_image_size(sample),
        )
        return replace(sample, keypoints_25=transformed)

    def transform_annotation(
        self,
        annotation: KeypointAnnotation,
        *,
        image_size: ImageSize,
    ) -> KeypointAnnotation:
        if annotation.coordinate_space == self._target_space:
            return annotation
        if annotation.coordinate_space == CoordinateSpace.IMAGE_PIXEL:
            if self._target_space == CoordinateSpace.IMAGE_NORMALIZED:
                return self._pixel_to_normalized(annotation, image_size)
        if annotation.coordinate_space == CoordinateSpace.IMAGE_NORMALIZED:
            if self._target_space == CoordinateSpace.IMAGE_PIXEL:
                return self._normalized_to_pixel(annotation, image_size)
        raise PreprocessingError(
            "Unsupported keypoint coordinate conversion: "
            f"{annotation.coordinate_space.value} -> {self._target_space.value}"
        )

    def _resolve_image_size(self, sample: HandActionSample) -> ImageSize:
        if self._image_size is not None:
            return self._image_size
        value = sample.metadata.get(self._metadata_key)
        return self._parse_image_size(value)

    @staticmethod
    def _parse_image_size(value: object) -> ImageSize:
        if isinstance(value, ImageSize):
            return value
        if isinstance(value, Mapping):
            return ImageSize(width=int(value["width"]), height=int(value["height"]))
        if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
            return ImageSize(width=int(value[0]), height=int(value[1]))
        raise PreprocessingError(
            "Image size is required for keypoint normalization. "
            "Provide ImageSize or sample metadata {'image_size': {'width': W, 'height': H}}."
        )

    def _pixel_to_normalized(
        self,
        annotation: KeypointAnnotation,
        image_size: ImageSize,
    ) -> KeypointAnnotation:
        keypoints = tuple(
            self._convert_point(point, scale_x=1.0 / image_size.width, scale_y=1.0 / image_size.height)
            for point in annotation.keypoints
        )
        return replace(annotation, keypoints=keypoints, coordinate_space=CoordinateSpace.IMAGE_NORMALIZED)

    def _normalized_to_pixel(
        self,
        annotation: KeypointAnnotation,
        image_size: ImageSize,
    ) -> KeypointAnnotation:
        keypoints = tuple(
            self._convert_point(point, scale_x=float(image_size.width), scale_y=float(image_size.height))
            for point in annotation.keypoints
        )
        return replace(annotation, keypoints=keypoints, coordinate_space=CoordinateSpace.IMAGE_PIXEL)

    @staticmethod
    def _convert_point(
        point: Sequence[float],
        *,
        scale_x: float,
        scale_y: float,
    ) -> tuple[float, ...]:
        if len(point) < 2:
            raise PreprocessingError("Each 2D keypoint must contain at least x and y.")
        converted = [float(point[0]) * scale_x, float(point[1]) * scale_y]
        converted.extend(float(value) for value in point[2:])
        return tuple(converted)


class ClipSampler:
    """Create a deterministic fixed-length clip view from frame-level metadata."""

    def __init__(self, config: ClipSamplingConfig) -> None:
        self._config = config

    def transform_sample(self, sample: HandActionSample) -> HandActionSample:
        if sample.clip is None:
            return sample
        sampled_clip = self.sample_clip(sample.clip)
        return replace(sample, clip=sampled_clip)

    def sample_clip(self, clip: VideoClipReference) -> VideoClipReference:
        indices = self._sample_indices(clip.start_frame, clip.end_frame)
        frame_paths = self._sample_paths(clip.frame_paths, indices, clip.start_frame)
        return replace(
            clip,
            start_frame=indices[0],
            end_frame=indices[-1],
            frame_paths=frame_paths,
        )

    def _sample_indices(self, start_frame: int, end_frame: int) -> tuple[int, ...]:
        if end_frame < start_frame:
            raise PreprocessingError("clip.end_frame must be greater than or equal to clip.start_frame.")

        available = list(range(start_frame, end_frame + 1, self._config.stride))
        if not self._config.include_end_frame and available and available[-1] == end_frame:
            available = available[:-1]
        if not available:
            raise PreprocessingError("Clip sampling produced no frames.")
        if len(available) >= self._config.num_frames:
            return tuple(available[: self._config.num_frames])

        padded = list(available)
        while len(padded) < self._config.num_frames:
            padded.append(available[-1])
        return tuple(padded)

    @staticmethod
    def _sample_paths(
        frame_paths: Sequence[Path],
        indices: Sequence[int],
        start_frame: int,
    ) -> tuple[Path, ...]:
        if not frame_paths:
            return ()
        sampled: list[Path] = []
        for frame_index in indices:
            offset = frame_index - start_frame
            if offset < 0 or offset >= len(frame_paths):
                raise PreprocessingError("Frame path count does not cover sampled clip indices.")
            sampled.append(frame_paths[offset])
        return tuple(sampled)


class SamplePreprocessor(Preprocessor):
    """Apply sample transforms in order."""

    def __init__(self, transforms: Sequence[SampleTransform] = ()) -> None:
        self._transforms = tuple(transforms)

    def transform_sample(self, sample: HandActionSample) -> HandActionSample:
        transformed = sample
        for transform in self._transforms:
            transformed = transform.transform_sample(transformed)
        return transformed

    def transform_batch(self, samples: Iterable[HandActionSample]) -> Sequence[HandActionSample]:
        return tuple(self.transform_sample(sample) for sample in samples)


class MetadataImageSizePreprocessor(SamplePreprocessor):
    """Convenience preprocessor for common image-pixel to normalized-keypoint flow."""

    def __init__(self) -> None:
        super().__init__(
            transforms=(
                KeypointNormalizer(target_space=CoordinateSpace.IMAGE_NORMALIZED),
            )
        )
