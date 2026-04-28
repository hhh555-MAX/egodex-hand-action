"""Rule-based keypoint retargeting."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from egodex_hand_action.contracts.data import KeypointPrediction, PredictionRecord
from egodex_hand_action.contracts.experiment import RetargetConfig
from egodex_hand_action.interfaces.retarget import KeypointRetargeter


class RetargetError(ValueError):
    """Raised when keypoint retargeting cannot be performed."""


@dataclass(frozen=True)
class KeypointMappingRule:
    target_index: int
    source_indices: Sequence[int]
    weights: Sequence[float] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.target_index < 0:
            raise RetargetError("target_index must be non-negative.")
        if not self.source_indices:
            raise RetargetError("source_indices must not be empty.")
        if self.weights is not None and len(self.weights) != len(self.source_indices):
            raise RetargetError("weights length must match source_indices length.")

    def normalized_weights(self) -> tuple[float, ...]:
        if self.weights is None:
            weight = 1.0 / len(self.source_indices)
            return tuple(weight for _ in self.source_indices)
        total = sum(float(weight) for weight in self.weights)
        if total == 0:
            raise RetargetError(f"Mapping rule for target {self.target_index} has zero total weight.")
        return tuple(float(weight) / total for weight in self.weights)


class RuleBasedRetargeter(KeypointRetargeter):
    """Retarget keypoint predictions with deterministic mapping rules."""

    def __init__(
        self,
        *,
        source_topology_name: str,
        target_topology_name: str,
        source_keypoint_count: int,
        target_keypoint_count: int,
        rules: Sequence[KeypointMappingRule],
    ) -> None:
        self._source_topology_name = source_topology_name
        self._target_topology_name = target_topology_name
        self._source_keypoint_count = source_keypoint_count
        self._target_keypoint_count = target_keypoint_count
        self._rules = tuple(sorted(rules, key=lambda rule: rule.target_index))
        self._validate_rules()

    @classmethod
    def from_config(cls, config: RetargetConfig) -> "RuleBasedRetargeter":
        if config.mapping_path is not None:
            rules = load_mapping_rules(config.mapping_path)
        else:
            rules = default_phantom21_to_egodex25_rules()
        return cls(
            source_topology_name=config.source_topology_name,
            target_topology_name=config.target_topology_name,
            source_keypoint_count=config.source_keypoint_count,
            target_keypoint_count=config.target_keypoint_count,
            rules=rules,
        )

    def source_topology_name(self) -> str:
        return self._source_topology_name

    def target_topology_name(self) -> str:
        return self._target_topology_name

    def retarget_prediction(self, prediction: KeypointPrediction) -> KeypointPrediction:
        self._validate_prediction(prediction)
        keypoints = tuple(
            self._apply_rule(rule, prediction.keypoints)
            for rule in self._rules
        )
        confidence = self._retarget_confidence(prediction.confidence)
        return KeypointPrediction(
            keypoints=keypoints,
            coordinate_space=prediction.coordinate_space,
            topology_name=self._target_topology_name,
            confidence=confidence,
        )

    def retarget_records(self, records: Sequence[PredictionRecord]) -> Sequence[PredictionRecord]:
        return tuple(self._retarget_record(record) for record in records)

    def _retarget_record(self, record: PredictionRecord) -> PredictionRecord:
        metadata = {
            **dict(record.metadata),
            "source_topology_name": record.prediction.topology_name,
            "target_topology_name": self._target_topology_name,
        }
        return replace(
            record,
            prediction=self.retarget_prediction(record.prediction),
            metadata=metadata,
        )

    def _apply_rule(
        self,
        rule: KeypointMappingRule,
        source_keypoints: Sequence[Sequence[float]],
    ) -> tuple[float, ...]:
        points = [source_keypoints[index] for index in rule.source_indices]
        dimension = len(points[0])
        if any(len(point) != dimension for point in points):
            raise RetargetError("All source keypoints must have the same dimension.")
        weights = rule.normalized_weights()
        return tuple(
            sum(float(point[axis]) * weights[index] for index, point in enumerate(points))
            for axis in range(dimension)
        )

    def _retarget_confidence(self, confidence: Sequence[float] | None) -> Sequence[float] | None:
        if confidence is None:
            return None
        if len(confidence) != self._source_keypoint_count:
            raise RetargetError(
                f"Expected {self._source_keypoint_count} confidence values, got {len(confidence)}."
            )
        return tuple(
            sum(float(confidence[source_index]) * weight for source_index, weight in zip(rule.source_indices, rule.normalized_weights()))
            for rule in self._rules
        )

    def _validate_prediction(self, prediction: KeypointPrediction) -> None:
        if len(prediction.keypoints) != self._source_keypoint_count:
            raise RetargetError(
                f"Expected {self._source_keypoint_count} source keypoints, got {len(prediction.keypoints)}."
            )
        if prediction.topology_name != self._source_topology_name:
            raise RetargetError(
                f"Expected source topology '{self._source_topology_name}', "
                f"got '{prediction.topology_name}'."
            )

    def _validate_rules(self) -> None:
        if self._source_keypoint_count <= 0 or self._target_keypoint_count <= 0:
            raise RetargetError("source and target keypoint counts must be positive.")
        if len(self._rules) != self._target_keypoint_count:
            raise RetargetError(
                f"Expected {self._target_keypoint_count} mapping rules, got {len(self._rules)}."
            )
        target_indices = [rule.target_index for rule in self._rules]
        expected_indices = list(range(self._target_keypoint_count))
        if target_indices != expected_indices:
            raise RetargetError(
                "Mapping rules must define every target index exactly once from "
                f"0 to {self._target_keypoint_count - 1}."
            )
        for rule in self._rules:
            for source_index in rule.source_indices:
                if source_index < 0 or source_index >= self._source_keypoint_count:
                    raise RetargetError(
                        f"Source index {source_index} for target {rule.target_index} is out of range."
                    )


def default_phantom21_to_egodex25_rules() -> Sequence[KeypointMappingRule]:
    """Return a conservative default mapping from Phantom 21 points to EgoDex 25 points."""

    direct_rules = [
        KeypointMappingRule(target_index=index, source_indices=(index,))
        for index in range(21)
    ]
    supplemental_rules = [
        KeypointMappingRule(target_index=21, source_indices=(0, 1), weights=(0.5, 0.5), name="wrist_thumb_base_midpoint"),
        KeypointMappingRule(target_index=22, source_indices=(0, 5), weights=(0.5, 0.5), name="wrist_index_base_midpoint"),
        KeypointMappingRule(target_index=23, source_indices=(0, 9), weights=(0.5, 0.5), name="wrist_middle_base_midpoint"),
        KeypointMappingRule(target_index=24, source_indices=(0, 17), weights=(0.5, 0.5), name="wrist_pinky_base_midpoint"),
    ]
    return tuple(direct_rules + supplemental_rules)


def load_mapping_rules(path: Path) -> Sequence[KeypointMappingRule]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_rules = payload.get("rules")
    else:
        raw_rules = payload
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, str):
        raise RetargetError("Mapping file must contain a 'rules' array.")
    return tuple(_rule_from_mapping(item) for item in raw_rules)


def save_mapping_rules(rules: Sequence[KeypointMappingRule], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "rules": [_rule_to_mapping(rule) for rule in rules],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _rule_from_mapping(item: Any) -> KeypointMappingRule:
    if not isinstance(item, Mapping):
        raise RetargetError("Each mapping rule must be a JSON object.")
    return KeypointMappingRule(
        target_index=int(item["target_index"]),
        source_indices=tuple(int(index) for index in _sequence(item["source_indices"])),
        weights=None if item.get("weights") is None else tuple(float(weight) for weight in _sequence(item["weights"])),
        name=None if item.get("name") is None else str(item["name"]),
    )


def _rule_to_mapping(rule: KeypointMappingRule) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "target_index": rule.target_index,
        "source_indices": list(rule.source_indices),
    }
    if rule.weights is not None:
        payload["weights"] = list(rule.weights)
    if rule.name is not None:
        payload["name"] = rule.name
    return payload


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RetargetError("Expected a JSON array.")
    return value
