from __future__ import annotations

from typing import Any, Dict


class DiagnosisGrader:
    def grade(self, diagnosis: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        true_roots = set(scenario["failure_injection"]["root_nodes"])
        guessed_roots = set(diagnosis.get("root_causes", []))
        true_blast = set(scenario.get("ground_truth_blast_radius", []))
        guessed_blast = set(diagnosis.get("blast_radius", []))

        root_score = len(true_roots & guessed_roots) / max(1, len(true_roots))
        blast_iou = len(true_blast & guessed_blast) / max(1, len(true_blast | guessed_blast))
        red_herring_penalty = 0.15 if guessed_roots & set(scenario.get("red_herrings", [])) else 0.0
        score = max(0.0, 0.6 * root_score + 0.4 * blast_iou - red_herring_penalty)
        return {
            "score": round(score, 4),
            "root_score": round(root_score, 4),
            "blast_radius_iou": round(blast_iou, 4),
            "red_herring_penalty": red_herring_penalty,
        }
