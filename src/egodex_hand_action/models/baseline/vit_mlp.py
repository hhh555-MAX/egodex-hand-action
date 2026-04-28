"""ViT + MLP baseline model backed by PyTorch and torchvision."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from egodex_hand_action.contracts.data import (
    CoordinateSpace,
    HandActionSample,
    KeypointPrediction,
    PredictionRecord,
)
from egodex_hand_action.contracts.experiment import BaselineModelConfig
from egodex_hand_action.interfaces.model import BaselineModelFactory, HandActionModel


class BaselineModelError(RuntimeError):
    """Raised when the baseline model cannot be created or used."""


@dataclass(frozen=True)
class VitMlpRuntimeConfig:
    backbone_name: str
    output_keypoint_count: int = 25
    output_dimension: int = 2
    pretrained: bool = True
    image_size: int = 224
    mlp_hidden_dims: Sequence[int] = field(default_factory=lambda: (512,))
    device: str = "cpu"
    topology_name: str = "egodex_25"
    output_coordinate_space: CoordinateSpace = CoordinateSpace.IMAGE_NORMALIZED

    @property
    def output_size(self) -> int:
        return self.output_keypoint_count * self.output_dimension


class VitMlpBaselineModel(HandActionModel):
    """Baseline model: image or clip -> ViT visual encoder -> MLP -> keypoints."""

    def __init__(self, config: VitMlpRuntimeConfig) -> None:
        self._config = config
        self._torch = self._import_required("torch")
        self._nn = self._torch.nn
        self._transforms = self._import_required("torchvision.transforms")
        self._models = self._import_required("torchvision.models")
        self._device = self._resolve_device(config.device)
        self._model = self._build_model().to(self._device)
        self._image_transform = self._build_image_transform()

    def name(self) -> str:
        return f"baseline_{self._config.backbone_name}_mlp"

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        state = self._torch.load(checkpoint_path, map_location=self._device)
        model_state = state.get("model_state_dict", state) if isinstance(state, Mapping) else state
        self._model.load_state_dict(model_state)

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "model_name": self.name(),
                "config": self._config,
                "model_state_dict": self._model.state_dict(),
            },
            checkpoint_path,
        )

    def predict(self, samples: Sequence[HandActionSample]) -> Sequence[PredictionRecord]:
        self._model.eval()
        records: list[PredictionRecord] = []
        with self._torch.no_grad():
            for sample in samples:
                output = self._predict_sample_tensor(sample)
                records.append(self._prediction_record(sample, output))
        return tuple(records)

    def torch_module(self):
        return self._model

    def _build_model(self):
        backbone = self._build_backbone()
        feature_dim = self._feature_dim(backbone)
        backbone.heads = self._nn.Identity()
        head = self._build_mlp(feature_dim)
        return self._nn.Sequential(backbone, head)

    def _build_backbone(self):
        builders = {
            "vit_b_16": ("vit_b_16", "ViT_B_16_Weights"),
            "vit_b_32": ("vit_b_32", "ViT_B_32_Weights"),
            "vit_l_16": ("vit_l_16", "ViT_L_16_Weights"),
            "vit_l_32": ("vit_l_32", "ViT_L_32_Weights"),
        }
        if self._config.backbone_name not in builders:
            supported = ", ".join(sorted(builders))
            raise BaselineModelError(
                f"Unsupported ViT backbone '{self._config.backbone_name}'. "
                f"Supported backbones: {supported}."
            )
        builder_name, weights_name = builders[self._config.backbone_name]
        builder = getattr(self._models, builder_name)
        weights = None
        if self._config.pretrained:
            weights = getattr(self._models, weights_name).DEFAULT
        return builder(weights=weights)

    def _feature_dim(self, backbone) -> int:
        hidden_dim = getattr(backbone, "hidden_dim", None)
        if hidden_dim is None:
            raise BaselineModelError("Could not infer ViT feature dimension from backbone.")
        return int(hidden_dim)

    def _build_mlp(self, input_dim: int):
        layers: list[Any] = []
        previous_dim = input_dim
        for hidden_dim in self._config.mlp_hidden_dims:
            layers.append(self._nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(self._nn.GELU())
            previous_dim = int(hidden_dim)
        layers.append(self._nn.Linear(previous_dim, self._config.output_size))
        return self._nn.Sequential(*layers)

    def _build_image_transform(self):
        return self._transforms.Compose(
            (
                self._transforms.Resize((self._config.image_size, self._config.image_size)),
                self._transforms.ToTensor(),
                self._transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )
        )

    def _predict_sample_tensor(self, sample: HandActionSample):
        image_paths = self._image_paths(sample)
        if not image_paths:
            raise BaselineModelError(f"Sample '{sample.sample_id}' has no image path for prediction.")
        batch = self._torch.stack([self._load_image_tensor(path) for path in image_paths]).to(self._device)
        outputs = self._model(batch)
        if outputs.ndim == 1:
            return outputs
        return outputs.mean(dim=0)

    def _load_image_tensor(self, image_path: Path):
        if not image_path.exists():
            raise BaselineModelError(f"Image file does not exist: {image_path}")
        with Image.open(image_path) as image:
            return self._image_transform(image.convert("RGB"))

    @staticmethod
    def _image_paths(sample: HandActionSample) -> Sequence[Path]:
        if sample.clip is not None and sample.clip.frame_paths:
            return tuple(sample.clip.frame_paths)
        if sample.frame is not None and sample.frame.image_path is not None:
            return (sample.frame.image_path,)
        return ()

    def _prediction_record(self, sample: HandActionSample, output) -> PredictionRecord:
        keypoints = self._tensor_to_keypoints(output)
        video_id = self._video_id(sample)
        frame_index = sample.frame.frame_index if sample.frame is not None else None
        return PredictionRecord(
            sample_id=sample.sample_id,
            video_id=video_id,
            frame_index=frame_index,
            method=self.name(),
            prediction=KeypointPrediction(
                keypoints=keypoints,
                coordinate_space=self._config.output_coordinate_space,
                topology_name=self._config.topology_name,
            ),
        )

    def _tensor_to_keypoints(self, output) -> Sequence[Sequence[float]]:
        values = output.detach().cpu().reshape(
            self._config.output_keypoint_count,
            self._config.output_dimension,
        )
        return tuple(tuple(float(value) for value in row.tolist()) for row in values)

    @staticmethod
    def _video_id(sample: HandActionSample) -> str:
        if sample.frame is not None:
            return sample.frame.video_id
        if sample.clip is not None:
            return sample.clip.video_id
        raise BaselineModelError(f"Sample '{sample.sample_id}' has neither frame nor clip reference.")

    def _resolve_device(self, requested_device: str):
        if requested_device == "auto":
            requested_device = "cuda" if self._torch.cuda.is_available() else "cpu"
        return self._torch.device(requested_device)

    @staticmethod
    def _import_required(module_name: str):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise BaselineModelError(
                "Baseline ViT+MLP requires PyTorch and torchvision. "
                "Install compatible 'torch' and 'torchvision' packages before creating the model."
            ) from exc


class TorchVisionVitMlpFactory(BaselineModelFactory):
    """Create torchvision-backed ViT + MLP baseline models."""

    def create(self, config: Mapping[str, Any]) -> HandActionModel:
        runtime_config = self._runtime_config(config)
        return VitMlpBaselineModel(runtime_config)

    @classmethod
    def from_baseline_config(
        cls,
        config: BaselineModelConfig,
        *,
        device: str = "cpu",
    ) -> VitMlpBaselineModel:
        runtime_config = VitMlpRuntimeConfig(
            backbone_name=config.backbone_name,
            output_keypoint_count=config.output_keypoint_count,
            output_dimension=config.output_dimension,
            pretrained=config.pretrained,
            device=device,
            image_size=int(config.extra.get("image_size", 224)),
            mlp_hidden_dims=tuple(config.extra.get("mlp_hidden_dims", (512,))),
            topology_name=str(config.extra.get("topology_name", "egodex_25")),
            output_coordinate_space=CoordinateSpace(
                str(config.extra.get("output_coordinate_space", CoordinateSpace.IMAGE_NORMALIZED.value))
            ),
        )
        return VitMlpBaselineModel(runtime_config)

    @staticmethod
    def _runtime_config(config: Mapping[str, Any]) -> VitMlpRuntimeConfig:
        return VitMlpRuntimeConfig(
            backbone_name=str(config["backbone_name"]),
            output_keypoint_count=int(config.get("output_keypoint_count", 25)),
            output_dimension=int(config.get("output_dimension", 2)),
            pretrained=bool(config.get("pretrained", True)),
            image_size=int(config.get("image_size", 224)),
            mlp_hidden_dims=tuple(int(value) for value in config.get("mlp_hidden_dims", (512,))),
            device=str(config.get("device", "cpu")),
            topology_name=str(config.get("topology_name", "egodex_25")),
            output_coordinate_space=CoordinateSpace(
                str(config.get("output_coordinate_space", CoordinateSpace.IMAGE_NORMALIZED.value))
            ),
        )
