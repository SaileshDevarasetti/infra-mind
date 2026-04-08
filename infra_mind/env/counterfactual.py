from __future__ import annotations

from typing import Any, Dict, Sequence

from env.cascade_simulator import CascadeSimulator


class CounterfactualAnalyzer:
    def __init__(self, simulator: CascadeSimulator, true_blast_radius: Sequence[str]) -> None:
        self.simulator = simulator
        self.true_blast_radius = list(true_blast_radius)

    def score(self, actions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        replay = self.simulator.replay_with_actions(actions, start_time=0)
        true_size = max(1, len(self.true_blast_radius))
        counterfactual_size = len(replay["blast_radius"])
        prevented_spread = (true_size - counterfactual_size) / true_size
        score = max(0.0, round(prevented_spread, 4))
        return {
            "counterfactual_score": score,
            "prevented_spread": score,
            "counterfactual_blast_radius": replay["blast_radius"],
        }
