"""Data structures and interfaces shared across project modules."""

from egodex_hand_action.contracts.data import (
    CoordinateSpace,
    DatasetSplit,
    FrameReference,
    HandActionSample,
    Handedness,
    KeypointAnnotation,
    KeypointPrediction,
    PredictionRecord,
    VideoClipReference,
)
from egodex_hand_action.contracts.experiment import (
    BaselineModelConfig,
    EvaluationConfig,
    ExperimentConfig,
    PhantomConfig,
    RetargetConfig,
    TrainingConfig,
)
from egodex_hand_action.contracts.metrics import (
    AggregateMetrics,
    FrameMetrics,
    MetricName,
    StabilityMetrics,
    VideoMetrics,
)

__all__ = [
    "AggregateMetrics",
    "BaselineModelConfig",
    "CoordinateSpace",
    "DatasetSplit",
    "EvaluationConfig",
    "ExperimentConfig",
    "FrameMetrics",
    "FrameReference",
    "HandActionSample",
    "Handedness",
    "KeypointAnnotation",
    "KeypointPrediction",
    "MetricName",
    "PhantomConfig",
    "PredictionRecord",
    "RetargetConfig",
    "StabilityMetrics",
    "TrainingConfig",
    "VideoClipReference",
    "VideoMetrics",
]

