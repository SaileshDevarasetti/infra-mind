from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Sequence

from env.graph_engine import TemporalKnowledgeGraph
from env.models import Action, GraphSnapshot


BETA_BY_DEPENDENCY = {
    "sync_rpc": 0.7,
    "async_queue": 0.2,
    "db_connection": 0.9,
    "cache_read": 0.4,
}


class CascadeSimulator:
    def __init__(
        self,
        graph: TemporalKnowledgeGraph,
        seed: int = 0,
        step_seconds: int = 30,
        natural_failure_gamma: float = 0.1,
        red_herring_nodes: Optional[Sequence[str]] = None,
        acceleration_after_step: Optional[int] = None,
        acceleration_factor: float = 1.0,
    ) -> None:
        self.graph = graph
        self.seed = seed
        self.rng = random.Random(seed)
        self.step_seconds = step_seconds
        self.natural_failure_gamma = natural_failure_gamma
        self.red_herring_nodes = set(red_herring_nodes or [])
        self.acceleration_after_step = acceleration_after_step
        self.acceleration_factor = acceleration_factor
        self.current_step = 0
        self.trajectory: List[Dict[str, Any]] = []
        self.true_cascade_nodes: set[str] = set()
        self.injection_points: List[str] = []

        for node_id in self.red_herring_nodes:
            if node_id in self.graph.graph:
                self.graph.graph.nodes[node_id]["metadata"]["red_herring"] = True

    def inject_failures(self, root_nodes: Sequence[str], failure_types: Sequence[str]) -> None:
        self.injection_points = list(root_nodes)
        for index, node_id in enumerate(root_nodes):
            failure_type = failure_types[index] if index < len(failure_types) else "crash"
            self.true_cascade_nodes.add(node_id)
            self.graph.inject_failure(node_id, failure_type=failure_type, severity=0.62)
            self.graph.graph.nodes[node_id]["metadata"]["in_cascade"] = True
        self._record_trajectory()

    def seed_red_herrings(self, nodes: Sequence[str]) -> None:
        for node_id in nodes:
            if node_id not in self.graph.graph:
                continue
            node = self.graph.graph.nodes[node_id]
            node["current_health"] = min(node["current_health"], 0.58)
            node["failure_state"] = "degraded"
            node["failure_type"] = node.get("failure_type") or "maintenance"
            node["metadata"]["red_herring"] = True
            self.graph._normalize_node_metrics(node_id)  # noqa: SLF001
        self.graph.update_edge_weights(self.step_seconds)
        self._record_trajectory()

    def step(self, actions: Optional[Sequence[Action]] = None) -> GraphSnapshot:
        action_targets = {action.target_node_id for action in actions or [] if action.target_node_id}
        for action in actions or []:
            self.graph.apply_action(action)

        acceleration = 1.0
        if self.acceleration_after_step is not None and self.current_step >= self.acceleration_after_step:
            acceleration = self.acceleration_factor

        transitions: List[tuple[str, str, str]] = []
        graph = self.graph.graph
        for node_id in list(graph.nodes):
            node = graph.nodes[node_id]
            if node.get("metadata", {}).get("red_herring") and node_id not in self.true_cascade_nodes:
                continue

            state = node["failure_state"]
            if state == "healthy":
                infection_pressure = self._infection_pressure(node_id)
                if infection_pressure <= 0.0:
                    continue
                roll = self.rng.random()
                if roll < infection_pressure * acceleration:
                    transitions.append((node_id, "healthy", "degraded"))
            elif state == "degraded":
                if self.rng.random() < self.natural_failure_gamma * acceleration:
                    transitions.append((node_id, "degraded", "failed"))
            elif state == "failed" and node_id in action_targets:
                transitions.append((node_id, "failed", "recovering"))

        for node_id, old_state, new_state in transitions:
            node = graph.nodes[node_id]
            node["failure_state"] = new_state
            node["metadata"]["in_cascade"] = True
            self.true_cascade_nodes.add(node_id)
            if new_state == "degraded":
                node["current_health"] = min(node["current_health"], 0.56)
                node["failure_type"] = node.get("failure_type") or "timeout"
                node["degraded_since"] = self.current_step * self.step_seconds
            elif new_state == "failed":
                node["current_health"] = min(node["current_health"], 0.28)
                node["failure_type"] = node.get("failure_type") or "crash"
            elif new_state == "recovering":
                node["current_health"] = max(node["current_health"], 0.65)
                node["failure_type"] = None
            self.graph._normalize_node_metrics(node_id)  # noqa: SLF001

        self.current_step += 1
        self.graph.advance_time(self.step_seconds)
        self._record_trajectory()
        return self.graph.full_snapshot()

    def run(self, max_steps: int, actions_by_step: Optional[Dict[int, Sequence[Action]]] = None) -> List[Dict[str, Any]]:
        for step_index in range(max_steps):
            self.step(actions_by_step.get(step_index) if actions_by_step else None)
        return copy.deepcopy(self.trajectory)

    def replay_with_actions(
        self,
        actions: Sequence[Dict[str, Any]] | Sequence[Action],
        start_time: int = 0,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        cloned = CascadeSimulator(
            graph=self.graph.clone(),
            seed=self.seed,
            step_seconds=self.step_seconds,
            natural_failure_gamma=self.natural_failure_gamma,
            red_herring_nodes=list(self.red_herring_nodes),
            acceleration_after_step=self.acceleration_after_step,
            acceleration_factor=self.acceleration_factor,
        )
        if self.injection_points:
            failure_types = [
                cloned.graph.graph.nodes[node].get("failure_type") or "crash"
                for node in self.injection_points
            ]
            cloned.inject_failures(self.injection_points, failure_types)

        normalized_actions = self._normalize_action_schedule(actions, start_time)
        max_horizon = max_steps if max_steps is not None else max(self.current_step + 1, 1)
        cloned.run(max_horizon, actions_by_step=normalized_actions)
        final_snapshot = cloned.graph.full_snapshot()
        return {
            "trajectory": cloned.trajectory,
            "blast_radius": final_snapshot.blast_radius,
            "cascade_wavefront": final_snapshot.cascade_wavefront,
        }

    def _infection_pressure(self, node_id: str) -> float:
        graph = self.graph.graph
        neighbors = list(graph.predecessors(node_id))
        if not neighbors:
            return 0.0

        failed_contributors = []
        for neighbor in neighbors:
            neighbor_state = graph.nodes[neighbor]["failure_state"]
            if neighbor_state not in {"degraded", "failed"}:
                continue
            edge = graph.edges[(neighbor, node_id)]
            beta = BETA_BY_DEPENDENCY.get(edge["dependency_type"], 0.2)
            failed_contributors.append(beta * edge["current_weight"])

        if not failed_contributors:
            return 0.0

        failed_neighbor_count = len(failed_contributors)
        total_neighbor_count = len(neighbors)
        avg_pressure = sum(failed_contributors) / failed_neighbor_count
        return min(0.95, avg_pressure * (failed_neighbor_count / total_neighbor_count))

    def _normalize_action_schedule(
        self,
        actions: Sequence[Dict[str, Any]] | Sequence[Action],
        start_time: int,
    ) -> Dict[int, List[Action]]:
        schedule: Dict[int, List[Action]] = {}
        for item in actions:
            if isinstance(item, Action):
                step = start_time
                action = item
            else:
                action = Action.model_validate(item["action"])
                step = max(start_time, int(item.get("step", start_time)))
            schedule.setdefault(step, []).append(action)
        return schedule

    def _record_trajectory(self) -> None:
        snapshot = self.graph.full_snapshot()
        self.trajectory.append(
            {
                "step": self.current_step,
                "timestamp": snapshot.timestamp.isoformat(),
                "wavefront": snapshot.cascade_wavefront,
                "blast_radius": snapshot.blast_radius,
                "cascade_nodes": sorted(self.true_cascade_nodes),
            }
        )
