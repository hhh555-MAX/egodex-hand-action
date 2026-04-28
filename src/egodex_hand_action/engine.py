"""Core orchestration engine for EgoDex hand action experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from egodex_hand_action.contracts.data import HandActionSample, PredictionRecord
from egodex_hand_action.contracts.experiment import ExperimentConfig
from egodex_hand_action.contracts.metrics import AggregateMetrics, FrameMetrics, VideoMetrics
from egodex_hand_action.interfaces.dataset import EgoDexDataset
from egodex_hand_action.interfaces.evaluation import Evaluator
from egodex_hand_action.interfaces.inference import Predictor
from egodex_hand_action.interfaces.model import HandActionModel
from egodex_hand_action.interfaces.training import ExperimentRunner, Trainer
from egodex_hand_action.interfaces.visualization import ReportWriter, Visualizer


class PipelineStage(str, Enum):
    TRAIN = "train"
    PREDICT = "predict"
    EVALUATE = "evaluate"
    VISUALIZE = "visualize"
    REPORT = "report"


@dataclass(frozen=True)
class EngineDatasets:
    train: EgoDexDataset | None = None
    validation: EgoDexDataset | None = None
    prediction: EgoDexDataset | None = None
    evaluation: EgoDexDataset | None = None


@dataclass(frozen=True)
class EngineComponents:
    model: HandActionModel
    datasets: EngineDatasets
    trainer: Trainer | None = None
    predictor: Predictor | None = None
    evaluator: Evaluator | None = None
    visualizer: Visualizer | None = None
    report_writer: ReportWriter | None = None


@dataclass(frozen=True)
class PipelineResult:
    checkpoint_path: Path | None = None
    predictions: Sequence[PredictionRecord] = field(default_factory=tuple)
    frame_metrics: Sequence[FrameMetrics] = field(default_factory=tuple)
    video_metrics: Sequence[VideoMetrics] = field(default_factory=tuple)
    aggregate_metrics: AggregateMetrics | None = None
    artifacts: Sequence[Path] = field(default_factory=tuple)


class EgoDexEngine(ExperimentRunner):
    def __init__(self, components: EngineComponents) -> None:
        self._components = components
        self._last_predictions: Sequence[PredictionRecord] = ()
        self._last_frame_metrics: Sequence[FrameMetrics] = ()
        self._last_video_metrics: Sequence[VideoMetrics] = ()
        self._last_aggregate_metrics: AggregateMetrics | None = None

    def run_training(self, config: ExperimentConfig) -> Path:
        trainer = self._require_component(self._components.trainer, "trainer")
        train_dataset = self._require_component(self._components.datasets.train, "train dataset")
        validation_dataset = self._require_component(
            self._components.datasets.validation,
            "validation dataset",
        )

        return trainer.fit(
            model=self._components.model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            config=config,
        )

    def run_prediction(self, config: ExperimentConfig) -> Sequence[PredictionRecord]:
        predictor = self._require_component(self._components.predictor, "predictor")
        prediction_dataset = self._require_component(
            self._components.datasets.prediction,
            "prediction dataset",
        )
        samples = self._collect_samples(prediction_dataset)
        predictions = predictor.predict_samples(self._components.model, samples)

        output_path = self._prediction_output_path(config)
        predictor.export_predictions(predictions, output_path)

        self._last_predictions = predictions
        return predictions

    def run_evaluation(self, config: ExperimentConfig) -> AggregateMetrics:
        evaluator = self._require_component(self._components.evaluator, "evaluator")
        evaluation_dataset = self._require_component(
            self._components.datasets.evaluation,
            "evaluation dataset",
        )
        predictions = self._last_predictions or self.run_prediction(config)
        ground_truth = self._collect_samples(evaluation_dataset)

        frame_metrics = evaluator.evaluate_frames(predictions, ground_truth)
        video_metrics = evaluator.evaluate_videos(frame_metrics)
        aggregate_metrics = evaluator.aggregate(video_metrics)
        evaluator.export_report(aggregate_metrics, config.training.output_dir)

        self._last_frame_metrics = frame_metrics
        self._last_video_metrics = video_metrics
        self._last_aggregate_metrics = aggregate_metrics
        return aggregate_metrics

    def run_pipeline(
        self,
        config: ExperimentConfig,
        stages: Sequence[PipelineStage] = (
            PipelineStage.TRAIN,
            PipelineStage.PREDICT,
            PipelineStage.EVALUATE,
        ),
    ) -> PipelineResult:
        checkpoint_path: Path | None = None
        artifacts: list[Path] = []

        if PipelineStage.TRAIN in stages:
            checkpoint_path = self.run_training(config)
            artifacts.append(checkpoint_path)

        if PipelineStage.PREDICT in stages:
            self.run_prediction(config)
            artifacts.append(self._prediction_output_path(config))

        if PipelineStage.EVALUATE in stages:
            self.run_evaluation(config)

        if PipelineStage.VISUALIZE in stages:
            artifacts.extend(self.run_visualization(config))

        if PipelineStage.REPORT in stages:
            report_path = self.run_report(config)
            artifacts.append(report_path)

        return PipelineResult(
            checkpoint_path=checkpoint_path,
            predictions=self._last_predictions,
            frame_metrics=self._last_frame_metrics,
            video_metrics=self._last_video_metrics,
            aggregate_metrics=self._last_aggregate_metrics,
            artifacts=tuple(artifacts),
        )

    def run_visualization(self, config: ExperimentConfig) -> Sequence[Path]:
        visualizer = self._require_component(self._components.visualizer, "visualizer")
        evaluation_dataset = self._require_component(
            self._components.datasets.evaluation,
            "evaluation dataset",
        )
        samples = self._collect_samples(evaluation_dataset)
        output_path = config.training.output_dir / f"{config.experiment_id}_metrics.png"

        return (
            visualizer.plot_metric_curves(
                frame_metrics=self._last_frame_metrics,
                video_metrics=self._last_video_metrics,
                output_path=output_path,
            ),
        )

    def run_report(self, config: ExperimentConfig) -> Path:
        report_writer = self._require_component(self._components.report_writer, "report writer")
        aggregate_metrics = self._require_component(
            self._last_aggregate_metrics,
            "aggregate metrics",
        )
        output_path = config.training.output_dir / f"{config.experiment_id}_report.md"
        return report_writer.write_experiment_report((aggregate_metrics,), output_path)

    @staticmethod
    def _collect_samples(dataset: EgoDexDataset) -> Sequence[HandActionSample]:
        return tuple(dataset.get_sample(index) for index in range(len(dataset)))

    @staticmethod
    def _prediction_output_path(config: ExperimentConfig) -> Path:
        return config.training.output_dir / f"{config.experiment_id}_predictions.json"

    @staticmethod
    def _require_component(component: object | None, name: str):
        if component is None:
            raise RuntimeError(f"Missing required engine component: {name}")
        return component


def main(
    engine: EgoDexEngine,
    config: ExperimentConfig,
    stages: Sequence[PipelineStage] = (
        PipelineStage.TRAIN,
        PipelineStage.PREDICT,
        PipelineStage.EVALUATE,
    ),
) -> PipelineResult:
    return engine.run_pipeline(config=config, stages=stages)
