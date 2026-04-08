from __future__ import annotations

from typing import Any, Dict, List, Sequence

from env.models import Action, GraphSnapshot


class RewardEngine:
    def __init__(self, max_steps: int, gamma: float = 0.95) -> None:
        self.max_steps = max_steps
        self.gamma = gamma
        self.cumulative_reward = 0.0

    def potential(self, snapshot: GraphSnapshot, steps_taken: int) -> float:
        total_nodes = max(1, len(snapshot.nodes))
        healthy_nodes = sum(1 for node in snapshot.nodes if node.health_score >= 0.7)
        blast_radius_fraction = len(snapshot.blast_radius) / total_nodes
        cascade_wavefront_fraction = len(snapshot.cascade_wavefront) / total_nodes
        phi = (
            0.4 * (healthy_nodes / total_nodes)
            + 0.3 * (1 - blast_radius_fraction)
            + 0.2 * (1 - cascade_wavefront_fraction)
            - 0.1 * (steps_taken / max(1, self.max_steps))
        )
        return round(phi, 6)

    def evaluate_step(
        self,
        action: Action,
        previous_snapshot: GraphSnapshot,
        current_snapshot: GraphSnapshot,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        step_number = int(context["step_number"])
        env_reward = 0.0
        penalties: List[str] = []
        bonuses: List[str] = []
        side_effects = context.get("side_effects", "")
        true_cascade_nodes = set(context.get("true_cascade_nodes", []))
        red_herrings = set(context.get("red_herrings", []))
        blast_radius = set(previous_snapshot.blast_radius)

        if action.action_type == "explore_node":
            target = action.target_node_id
            node = next((item for item in current_snapshot.nodes if item.node_id == target), None)
            if node and node.health_score < 0.6:
                env_reward += 0.10
                bonuses.append("Exploration exposed a genuinely anomalous node.")
            else:
                env_reward -= 0.05
                penalties.append("Exploration spent on a healthy node.")

        elif action.action_type == "query_dependency":
            if action.target_edge:
                edge = next(
                    (
                        item
                        for item in current_snapshot.edges
                        if (item.source_id, item.target_id) == tuple(action.target_edge)
                    ),
                    None,
                )
                if edge and edge.health_weight < 0.3:
                    env_reward += 0.08
                    bonuses.append("Dependency inspection revealed a broken edge.")

        elif action.action_type == "hypothesize_root":
            candidate = (action.hypothesis or {}).get("root_node") or action.target_node_id
            if candidate in true_cascade_nodes:
                env_reward += 0.12
                bonuses.append("Hypothesis named a node on the true cascade path.")

        elif action.action_type in {"execute_heal", "rollback_deployment", "scale_out"}:
            target = action.target_node_id
            prev_health = self._node_health(previous_snapshot, target)
            current_health = self._node_health(current_snapshot, target)
            improvement = current_health - prev_health
            if improvement > 0.2:
                env_reward += 0.15
                bonuses.append("Healing action materially improved node health.")
            if target in red_herrings:
                env_reward -= 0.08
                penalties.append("Healing action targeted a red-herring node.")
            if target not in blast_radius:
                env_reward -= 0.15
                penalties.append("Healing action targeted a node outside the blast radius.")

        elif action.action_type == "circuit_break":
            edge_key = tuple(action.target_edge) if action.target_edge else None
            if edge_key and set(edge_key) & true_cascade_nodes:
                env_reward += 0.20
                bonuses.append("Circuit breaker isolated a cascade edge.")

        elif action.action_type == "restart_service":
            if len(current_snapshot.cascade_wavefront) > len(previous_snapshot.cascade_wavefront):
                env_reward -= 0.10
                penalties.append("Restart expanded the cascade wavefront.")

        if step_number > 15:
            env_reward -= 0.05
            penalties.append("Analysis paralysis penalty after 15 steps.")

        prev_potential = self.potential(previous_snapshot, step_number - 1)
        current_potential = self.potential(current_snapshot, step_number)
        shaped_reward = env_reward + self.gamma * current_potential - prev_potential
        self.cumulative_reward += shaped_reward
        feedback = side_effects or "Environment transition applied."
        return {
            "step_reward": round(shaped_reward, 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "feedback": feedback,
            "penalty_reasons": penalties,
            "bonus_reasons": bonuses,
        }

    def finalize_episode(
        self,
        diagnosis: Dict[str, Any],
        success: bool,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        episode_reward = 0.0
        penalties: List[str] = []
        bonuses: List[str] = []
        true_roots = set(context.get("true_root_nodes", []))
        true_blast_radius = set(context.get("true_blast_radius", []))
        guessed_roots = set(diagnosis.get("root_causes", []))
        guessed_blast_radius = set(diagnosis.get("blast_radius", []))
        red_herrings = set(context.get("red_herrings", []))
        counterfactual_score = float(context.get("counterfactual_score", 0.0))
        steps_taken = int(context.get("steps_taken", self.max_steps))

        if true_roots and guessed_roots and true_roots.issubset(guessed_roots):
            episode_reward += 0.35
            bonuses.append("Root cause nodes correctly identified.")
        if true_blast_radius and abs(len(guessed_blast_radius) - len(true_blast_radius)) <= 2:
            episode_reward += 0.25
            bonuses.append("Blast radius was tightly bounded.")
        if counterfactual_score > 0.0:
            episode_reward += min(0.20, counterfactual_score * 0.20)
            bonuses.append("Counterfactual replay showed meaningful prevention.")
        if steps_taken < 12:
            episode_reward += 0.10
            bonuses.append("Solved efficiently within 12 actions.")
        if context.get("secondary_failure_caused"):
            episode_reward -= 0.20
            penalties.append("A healing action caused a permanent secondary failure.")
        if guessed_roots & red_herrings:
            episode_reward -= 0.15
            penalties.append("Diagnosis named a red-herring node as root cause.")

        self.cumulative_reward += episode_reward
        return {
            "step_reward": round(episode_reward, 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "episode_reward": round(self.cumulative_reward, 4),
            "counterfactual_score": round(counterfactual_score, 4),
            "feedback": "Episode complete.",
            "penalty_reasons": penalties,
            "bonus_reasons": bonuses,
            "done": True,
            "success": success,
        }

    def _node_health(self, snapshot: GraphSnapshot, node_id: str | None) -> float:
        if not node_id:
            return 1.0
        node = next((item for item in snapshot.nodes if item.node_id == node_id), None)
        return node.health_score if node else 1.0
