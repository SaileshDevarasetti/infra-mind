from __future__ import annotations

from typing import Any, Dict, Sequence

from env.counterfactual import CounterfactualAnalyzer
from tasks.base_task import BaseTask


class HealingGrader:
    def __init__(self, task: BaseTask) -> None:
        self.task = task

    def grade(
        self,
        actions: Sequence[Dict[str, Any]],
        healed_nodes: Sequence[str],
        diagnosis: Dict[str, Any],
        secondary_failure_caused: bool = False,
    ) -> Dict[str, Any]:
        simulator = self.task.create_simulator()
        counterfactual = CounterfactualAnalyzer(
            simulator,
            self.task.scenario["ground_truth_blast_radius"],
        ).score(actions)
        true_blast = set(self.task.scenario["ground_truth_blast_radius"])
        healed = set(healed_nodes)
        restored_fraction = len(healed & true_blast) / max(1, len(true_blast))
        prevented_spread = counterfactual["counterfactual_score"]
        score = max(0.0, 0.5 * restored_fraction + 0.5 * prevented_spread - (0.2 if secondary_failure_caused else 0.0))
        return {
            "score": round(score, 4),
            "restored_fraction": round(restored_fraction, 4),
            "counterfactual_score": round(counterfactual["counterfactual_score"], 4),
            "prevented_spread": round(prevented_spread, 4),
            "secondary_failure_penalty": 0.2 if secondary_failure_caused else 0.0,
            "diagnosis_considered": diagnosis,
        }
