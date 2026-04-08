from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from env.cascade_simulator import CascadeSimulator
from env.counterfactual import CounterfactualAnalyzer
from env.models import Action, Observation, Reward
from env.reward_engine import RewardEngine
from tasks import TASK_REGISTRY, BaseTask


AVAILABLE_ACTIONS = [
    "explore_node",
    "query_dependency",
    "hypothesize_root",
    "execute_heal",
    "circuit_break",
    "rollback_deployment",
    "scale_out",
    "restart_service",
    "submit_diagnosis",
]


@dataclass
class StepResult:
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class InfraMindEnvironment:
    def __init__(self, task_id: str = "easy", scenario_id: int = 0) -> None:
        self.task_id = task_id
        self.scenario_id = scenario_id
        self.task: BaseTask
        self.simulator: CascadeSimulator
        self.counterfactual_simulator: CascadeSimulator
        self.reward_engine: RewardEngine
        self.current_focus_node: str = ""
        self.previous_action_result: Optional[str] = None
        self.action_budget_remaining = 0
        self.step_number = 0
        self.time_elapsed_seconds = 0.0
        self.done = False
        self.success = False
        self.secondary_failure_caused = False
        self.triggered_traps: List[str] = []
        self.action_trace: List[Dict[str, Any]] = []
        self.reset(task_id, scenario_id)

    def reset(self, task_id: Optional[str] = None, scenario_id: Optional[int] = None) -> Observation:
        if task_id is not None:
            self.task_id = task_id
        if scenario_id is not None:
            self.scenario_id = scenario_id
        task_cls = TASK_REGISTRY[self.task_id]
        self.task = task_cls(self.scenario_id)
        self.simulator = self.task.create_simulator()
        self.counterfactual_simulator = self.task.create_simulator()
        self.reward_engine = RewardEngine(max_steps=self.task.max_steps)
        self.current_focus_node = self.simulator.graph.get_most_anomalous_node()
        self.simulator.graph.set_visibility_center(self.current_focus_node, radius=2)
        self.previous_action_result = None
        self.action_budget_remaining = self.task.action_budget
        self.step_number = 0
        self.time_elapsed_seconds = 0.0
        self.done = False
        self.success = False
        self.secondary_failure_caused = False
        self.triggered_traps = []
        self.action_trace = []
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        if self.done:
            reward = Reward(
                step_reward=0.0,
                cumulative_reward=self.reward_engine.cumulative_reward,
                episode_reward=self.reward_engine.cumulative_reward,
                counterfactual_score=0.0,
                feedback="Episode already completed.",
                penalty_reasons=[],
                bonus_reasons=[],
                done=True,
                success=self.success,
            )
            return StepResult(observation=self._build_observation(), reward=reward, done=True, info={"status": "completed"})

        previous_snapshot = self.simulator.graph.full_snapshot()
        self.step_number += 1
        self.action_budget_remaining -= 1

        if action.target_node_id and action.target_node_id in self.simulator.graph.graph:
            self.current_focus_node = action.target_node_id
            current_radius = self.simulator.graph.node_visibility.get(action.target_node_id, self.simulator.graph.visibility_radius)
            self.simulator.graph.set_visibility_center(action.target_node_id, radius=current_radius)

        if action.action_type == "submit_diagnosis":
            counterfactual = CounterfactualAnalyzer(
                self.counterfactual_simulator,
                self.task.scenario["ground_truth_blast_radius"],
            ).score(self.action_trace)
            full_snapshot = self.simulator.graph.full_snapshot()
            diagnosis = action.diagnosis or {}
            health_ratio = self._health_ratio(full_snapshot)
            self.success = self.task.evaluate_success(diagnosis, full_snapshot.blast_radius, health_ratio)
            reward_payload = self.reward_engine.finalize_episode(
                diagnosis=diagnosis,
                success=self.success,
                context={
                    "true_root_nodes": self.task.scenario["failure_injection"]["root_nodes"],
                    "true_blast_radius": self.task.scenario["ground_truth_blast_radius"],
                    "red_herrings": self.task.scenario.get("red_herrings", []),
                    "counterfactual_score": counterfactual["counterfactual_score"],
                    "steps_taken": self.step_number,
                    "secondary_failure_caused": self.secondary_failure_caused,
                },
            )
            self.done = True
            self.previous_action_result = "Final diagnosis submitted."
            reward = Reward(**reward_payload)
            observation = self._build_observation()
            return StepResult(
                observation=observation,
                reward=reward,
                done=True,
                info={"counterfactual": counterfactual, "health_ratio": health_ratio},
            )

        if action.action_type == "restart_service":
            self._apply_healing_trap(action.target_node_id)

        current_snapshot = self.simulator.step([action])
        self.time_elapsed_seconds += self.simulator.step_seconds
        self.previous_action_result = self.simulator.graph.action_log[-1]["side_effects"]

        self.action_trace.append({"step": max(0, self.step_number - 1), "action": action.model_dump()})
        reward_payload = self.reward_engine.evaluate_step(
            action=action,
            previous_snapshot=previous_snapshot,
            current_snapshot=current_snapshot,
            context={
                "step_number": self.step_number,
                "side_effects": self.previous_action_result,
                "true_cascade_nodes": self._true_cascade_nodes(),
                "red_herrings": self.task.scenario.get("red_herrings", []),
            },
        )

        auto_done = self.action_budget_remaining <= 0 or self.step_number >= self.task.max_steps
        if auto_done:
            counterfactual = CounterfactualAnalyzer(
                self.counterfactual_simulator,
                self.task.scenario["ground_truth_blast_radius"],
            ).score(self.action_trace)
            diagnosis = {
                "root_causes": [],
                "blast_radius": current_snapshot.blast_radius,
                "isolated_edges": [],
                "trap_triggered_nodes": self.triggered_traps,
            }
            health_ratio = self._health_ratio(current_snapshot)
            self.success = self.task.evaluate_success(diagnosis, current_snapshot.blast_radius, health_ratio)
            final_payload = self.reward_engine.finalize_episode(
                diagnosis=diagnosis,
                success=self.success,
                context={
                    "true_root_nodes": self.task.scenario["failure_injection"]["root_nodes"],
                    "true_blast_radius": self.task.scenario["ground_truth_blast_radius"],
                    "red_herrings": self.task.scenario.get("red_herrings", []),
                    "counterfactual_score": counterfactual["counterfactual_score"],
                    "steps_taken": self.step_number,
                    "secondary_failure_caused": self.secondary_failure_caused,
                },
            )
            reward_payload = final_payload
            auto_done = True
            self.done = True
        reward = Reward(
            step_reward=reward_payload["step_reward"],
            cumulative_reward=reward_payload["cumulative_reward"],
            episode_reward=reward_payload.get("episode_reward"),
            counterfactual_score=reward_payload.get("counterfactual_score"),
            feedback=reward_payload["feedback"],
            penalty_reasons=reward_payload["penalty_reasons"],
            bonus_reasons=reward_payload["bonus_reasons"],
            done=auto_done,
            success=self.success if auto_done else False,
        )
        observation = self._build_observation()
        if auto_done:
            self.done = True
        return StepResult(
            observation=observation,
            reward=reward,
            done=reward.done,
            info={
                "focus_node": self.current_focus_node,
                "blast_radius_size": len(current_snapshot.blast_radius),
                "wavefront_size": len(current_snapshot.cascade_wavefront),
                "triggered_traps": list(self.triggered_traps),
            },
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "full_snapshot": self.simulator.graph.full_snapshot(),
            "visible_snapshot": self.simulator.graph.get_subgraph_around(
                self.current_focus_node,
                self.simulator.graph.node_visibility.get(self.current_focus_node, self.simulator.graph.visibility_radius),
            ),
            "task": self.task.metadata.__dict__,
            "action_budget_remaining": self.action_budget_remaining,
            "step_number": self.step_number,
            "time_elapsed_seconds": self.time_elapsed_seconds,
            "done": self.done,
            "success": self.success,
        }

    def _build_observation(self) -> Observation:
        radius = self.simulator.graph.node_visibility.get(self.current_focus_node, self.simulator.graph.visibility_radius)
        visible_snapshot = self.simulator.graph.get_subgraph_around(self.current_focus_node, radius)
        historical = self.simulator.graph.get_temporal_snapshots(n=5)
        return Observation(
            task_id=self.task.metadata.task_id,
            task_name=self.task.metadata.task_name,
            task_description=self.task.metadata.task_description,
            graph_snapshot=visible_snapshot,
            visible_subgraph_radius=radius,
            historical_snapshots=historical,
            available_actions=AVAILABLE_ACTIONS,
            action_budget_remaining=max(0, self.action_budget_remaining),
            time_elapsed_seconds=self.time_elapsed_seconds,
            step_number=self.step_number,
            previous_action_result=self.previous_action_result,
        )

    def _true_cascade_nodes(self) -> List[str]:
        nodes = set()
        for step in self.task.scenario.get("cascade_trajectory", []):
            nodes.update(step.get("cascade_nodes", []))
        return sorted(nodes)

    def _health_ratio(self, snapshot) -> float:
        total = max(1, len(snapshot.nodes))
        healthy = sum(1 for node in snapshot.nodes if node.health_score >= 0.7)
        return healthy / total

    def _apply_healing_trap(self, node_id: Optional[str]) -> None:
        if not node_id:
            return
        for node in self.task.scenario["graph"]["nodes"]:
            if node["node_id"] != node_id:
                continue
            trap_target = node.get("metadata", {}).get("restart_trap_target")
            if not trap_target:
                return
            if trap_target not in self.simulator.graph.graph:
                return
            target = self.simulator.graph.graph.nodes[trap_target]
            target["failure_state"] = "failed"
            target["failure_type"] = "crash"
            target["current_health"] = min(target["current_health"], 0.22)
            target["metadata"]["in_cascade"] = True
            self.simulator.graph._normalize_node_metrics(trap_target)  # noqa: SLF001
            self.secondary_failure_caused = True
            self.triggered_traps.append(node_id)
            self.previous_action_result = (
                f"Restarting {node_id} triggered an undocumented dependency and failed {trap_target}."
            )
            return
