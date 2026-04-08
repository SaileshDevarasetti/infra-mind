from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from env.cascade_simulator import CascadeSimulator
from env.graph_engine import TemporalKnowledgeGraph


SCENARIO_DIR = Path(__file__).resolve().parents[1] / "data" / "scenarios"


@dataclass
class TaskMetadata:
    task_id: str
    task_name: str
    task_description: str
    action_budget: int
    max_steps: int


class BaseTask:
    metadata: TaskMetadata

    def __init__(self, scenario_id: int = 0) -> None:
        self.scenario_id = scenario_id
        self.scenario = self.load_scenario(scenario_id)

    def load_scenario(self, scenario_id: int) -> Dict[str, Any]:
        path = SCENARIO_DIR / f"{self.metadata.task_id}_{scenario_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def create_graph(self) -> TemporalKnowledgeGraph:
        return TemporalKnowledgeGraph(
            nodes=self.scenario["graph"]["nodes"],
            edges=self.scenario["graph"]["edges"],
            seed=self.scenario["seed"],
        )

    def create_simulator(self) -> CascadeSimulator:
        graph = self.create_graph()
        simulator = CascadeSimulator(
            graph=graph,
            seed=self.scenario["seed"],
            red_herring_nodes=self.scenario.get("red_herrings", []),
            acceleration_after_step=15 if self.metadata.task_id == "hard" else None,
            acceleration_factor=1.35 if self.metadata.task_id == "hard" else 1.0,
        )
        injection = self.scenario["failure_injection"]
        simulator.inject_failures(injection["root_nodes"], injection["failure_types"])
        if self.scenario.get("red_herrings"):
            simulator.seed_red_herrings(self.scenario["red_herrings"])
        return simulator

    def evaluate_success(self, diagnosis: Dict[str, Any], final_blast_radius: List[str], health_ratio: float) -> bool:
        raise NotImplementedError

    @property
    def action_budget(self) -> int:
        return self.metadata.action_budget

    @property
    def max_steps(self) -> int:
        return self.metadata.max_steps
