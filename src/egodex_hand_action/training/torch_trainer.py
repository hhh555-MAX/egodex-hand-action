"""PyTorch trainer for keypoint regression baselines."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Sequence

from egodex_hand_action.contracts.data import HandActionSample
from egodex_hand_action.contracts.experiment import ExperimentConfig
from egodex_hand_action.interfaces.dataset import EgoDexDataset
from egodex_hand_action.interfaces.model import HandActionModel
from egodex_hand_action.interfaces.training import Trainer
from egodex_hand_action.training.management import ExperimentManager, TrainingManagementError


class TorchKeypointRegressionTrainer(Trainer):
    """Train a baseline model against EgoDex 25-keypoint regression targets."""

    def __init__(self, experiment_manager: ExperimentManager | None = None) -> None:
        self._experiment_manager = experiment_manager or ExperimentManager()

    def fit(
        self,
        model: HandActionModel,
        train_dataset: EgoDexDataset,
        validation_dataset: EgoDexDataset,
        config: ExperimentConfig,
    ) -> Path:
        torch = self._import_torch()
        self._set_seed(torch, config.training.seed)
        paths = self._experiment_manager.prepare(config)
        module = self._torch_module(model)
        device = self._resolve_device(torch, config.training.device)
        module.to(device)

        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=config.training.learning_rate,
        )
        loss_fn = torch.nn.MSELoss()

        best_validation_loss: float | None = None
        best_checkpoint_path = paths.checkpoints_dir / "best.pt"
        epoch_log_path = paths.logs_dir / "epochs.jsonl"

        for epoch in range(1, config.training.max_epochs + 1):
            train_loss = self._run_epoch(
                torch=torch,
                model=model,
                module=module,
                dataset=train_dataset,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch_size=config.training.batch_size,
                device=device,
                train=True,
            )
            validation_loss = self._run_epoch(
                torch=torch,
                model=model,
                module=module,
                dataset=validation_dataset,
                optimizer=None,
                loss_fn=loss_fn,
                batch_size=config.training.batch_size,
                device=device,
                train=False,
            )
            self._append_epoch_log(
                epoch_log_path,
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                },
            )
            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                model.save_checkpoint(best_checkpoint_path)

        self._experiment_manager.update_manifest(
            paths,
            {
                "status": "trained",
                "best_checkpoint_path": str(best_checkpoint_path),
                "best_validation_loss": best_validation_loss,
                "epoch_log_path": str(epoch_log_path),
            },
        )
        return best_checkpoint_path

    def _run_epoch(
        self,
        *,
        torch: Any,
        model: HandActionModel,
        module: Any,
        dataset: EgoDexDataset,
        optimizer: Any | None,
        loss_fn: Any,
        batch_size: int,
        device: Any,
        train: bool,
    ) -> float:
        if train:
            module.train()
        else:
            module.eval()

        samples = tuple(dataset.get_sample(index) for index in range(len(dataset)))
        if not samples:
            raise TrainingManagementError("Cannot train or validate on an empty dataset.")

        losses: list[float] = []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in self._batches(samples, batch_size):
                if train:
                    optimizer.zero_grad()
                loss = self._batch_loss(
                    torch=torch,
                    model=model,
                    module=module,
                    samples=batch,
                    loss_fn=loss_fn,
                    device=device,
                )
                if train:
                    loss.backward()
                    optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
        return sum(losses) / len(losses)

    def _batch_loss(
        self,
        *,
        torch: Any,
        model: HandActionModel,
        module: Any,
        samples: Sequence[HandActionSample],
        loss_fn: Any,
        device: Any,
    ):
        predictions = []
        targets = []
        for sample in samples:
            predictions.append(self._forward_sample(torch, model, module, sample, device))
            targets.append(self._target_tensor(torch, sample, device))
        prediction_batch = torch.stack(predictions)
        target_batch = torch.stack(targets)
        return loss_fn(prediction_batch, target_batch)

    def _forward_sample(
        self,
        torch: Any,
        model: HandActionModel,
        module: Any,
        sample: HandActionSample,
        device: Any,
    ):
        image_paths = self._image_paths(model, sample)
        if not image_paths:
            raise TrainingManagementError(f"Sample '{sample.sample_id}' has no image path.")
        image_tensors = torch.stack(
            [self._load_image_tensor(model, path) for path in image_paths]
        ).to(device)
        outputs = module(image_tensors)
        if outputs.ndim == 1:
            return outputs
        return outputs.mean(dim=0)

    @staticmethod
    def _target_tensor(torch: Any, sample: HandActionSample, device: Any):
        if sample.keypoints_25 is None:
            raise TrainingManagementError(f"Sample '{sample.sample_id}' is missing keypoints_25.")
        flattened = [
            float(value)
            for point in sample.keypoints_25.keypoints
            for value in point
        ]
        return torch.tensor(flattened, dtype=torch.float32, device=device)

    @staticmethod
    def _batches(
        samples: Sequence[HandActionSample],
        batch_size: int,
    ) -> Sequence[Sequence[HandActionSample]]:
        if batch_size <= 0:
            raise TrainingManagementError("batch_size must be positive.")
        return tuple(samples[index : index + batch_size] for index in range(0, len(samples), batch_size))

    @staticmethod
    def _torch_module(model: HandActionModel):
        torch_module = getattr(model, "torch_module", None)
        if torch_module is None:
            raise TrainingManagementError(
                "TorchKeypointRegressionTrainer requires a model with a torch_module() method."
            )
        return torch_module()

    @staticmethod
    def _image_paths(model: HandActionModel, sample: HandActionSample) -> Sequence[Path]:
        image_paths = getattr(model, "_image_paths", None)
        if image_paths is None:
            raise TrainingManagementError(
                "TorchKeypointRegressionTrainer requires a baseline model with _image_paths()."
            )
        return image_paths(sample)

    @staticmethod
    def _load_image_tensor(model: HandActionModel, path: Path):
        load_image_tensor = getattr(model, "_load_image_tensor", None)
        if load_image_tensor is None:
            raise TrainingManagementError(
                "TorchKeypointRegressionTrainer requires a baseline model with _load_image_tensor()."
            )
        return load_image_tensor(path)

    @staticmethod
    def _set_seed(torch: Any, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_device(torch: Any, requested_device: str):
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(requested_device)

    @staticmethod
    def _append_epoch_log(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _import_torch():
        try:
            return importlib.import_module("torch")
        except ImportError as exc:
            raise TrainingManagementError(
                "TorchKeypointRegressionTrainer requires PyTorch. "
                "Install a compatible 'torch' package before training."
            ) from exc
