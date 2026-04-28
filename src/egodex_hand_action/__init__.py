"""Core contracts and orchestration for the EgoDex hand action project."""

from egodex_hand_action.engine import (
    EgoDexEngine,
    EngineComponents,
    EngineDatasets,
    PipelineResult,
    PipelineStage,
    main,
)

__all__ = [
    "EgoDexEngine",
    "EngineComponents",
    "EngineDatasets",
    "PipelineResult",
    "PipelineStage",
    "main",
]
