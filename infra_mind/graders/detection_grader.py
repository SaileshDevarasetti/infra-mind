from __future__ import annotations

from typing import Any, Dict, Sequence


class DetectionGrader:
    def grade(self, explored_nodes: Sequence[str], scenario: Dict[str, Any]) -> Dict[str, Any]:
        anomalous_nodes = set(scenario.get("ground_truth_blast_radius", []))
        explored = set(explored_nodes)
        true_positive = len(anomalous_nodes & explored)
        precision = true_positive / max(1, len(explored))
        recall = true_positive / max(1, len(anomalous_nodes))
        score = 0.5 * precision + 0.5 * recall
        return {
            "score": round(score, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "true_positive_nodes": sorted(anomalous_nodes & explored),
        }
