from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from openai import OpenAI

from env.environment import InfraMindEnvironment
from env.models import Action, Observation


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
# Do NOT set a default for HF_TOKEN — it must come from the environment / secrets
HF_TOKEN = os.getenv("HF_TOKEN")
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def _log_start(run_id: str) -> None:
    print(f"START {run_id}", flush=True)


def _log_step(msg: str) -> None:
    print(f"STEP {msg}", flush=True)


def _log_end(run_id: str, status: str = "SUCCESS") -> None:
    print(f"END {run_id} {status}", flush=True)


@dataclass
class ActionStats:
    total_reward: float = 0.0
    count: int = 0

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.count if self.count else 0.0


@dataclass
class MCTSLitePlanner:
    exploration_constant: float = 1.41
    rollout_count: int = 5
    rollout_depth: int = 2
    seed: int = 7
    stats: Dict[str, ActionStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_action(self, env: InfraMindEnvironment) -> Action:
        candidates = self.propose_actions(env._build_observation())  # noqa: SLF001
        if not candidates:
            return Action(action_type="submit_diagnosis", diagnosis={"root_causes": [], "blast_radius": []}, reasoning="No actions available.")

        total_visits = sum(self.stats.get(self._key(action), ActionStats()).count for action in candidates) + 1
        for _ in range(self.rollout_count):
            candidate = self._ucb_select(candidates, total_visits)
            simulated_reward = self._simulate_rollout(env, candidate)
            key = self._key(candidate)
            record = self.stats.setdefault(key, ActionStats())
            record.total_reward += simulated_reward
            record.count += 1
            total_visits += 1

        best_action = max(
            candidates,
            key=lambda action: self._ucb_score(action, total_visits),
        )
        return best_action

    def propose_actions(self, observation: Observation) -> List[Action]:
        nodes = sorted(observation.graph_snapshot.nodes, key=lambda item: item.health_score)
        edges = sorted(observation.graph_snapshot.edges, key=lambda item: item.health_weight)
        candidates: List[Action] = []

        for node in nodes[:3]:
            candidates.append(
                Action(
                    action_type="explore_node",
                    target_node_id=node.node_id,
                    reasoning=f"Expand visibility around anomalous node {node.node_id}.",
                )
            )
        for edge in edges[:2]:
            candidates.append(
                Action(
                    action_type="query_dependency",
                    target_edge=(edge.source_id, edge.target_id),
                    reasoning=f"Inspect weak dependency {edge.source_id}->{edge.target_id}.",
                )
            )
            if edge.health_weight < 0.45:
                candidates.append(
                    Action(
                        action_type="circuit_break",
                        target_edge=(edge.source_id, edge.target_id),
                        reasoning=f"Isolate potentially cascading dependency {edge.source_id}->{edge.target_id}.",
                    )
                )
        for node in nodes[:2]:
            if node.is_failed or node.is_degraded:
                candidates.append(
                    Action(
                        action_type="execute_heal",
                        target_node_id=node.node_id,
                        heal_strategy="smart_heal",
                        reasoning=f"Attempt healing on degraded node {node.node_id}.",
                    )
                )
                candidates.append(
                    Action(
                        action_type="scale_out",
                        target_node_id=node.node_id,
                        reasoning=f"Relieve load on degraded node {node.node_id}.",
                    )
                )
        if nodes:
            candidates.append(
                Action(
                    action_type="hypothesize_root",
                    target_node_id=nodes[0].node_id,
                    hypothesis={"root_node": nodes[0].node_id},
                    reasoning=f"Current best root candidate is {nodes[0].node_id}.",
                )
            )
        deduped: Dict[str, Action] = {}
        for action in candidates:
            deduped.setdefault(self._key(action), action)
        return list(deduped.values())

    def _simulate_rollout(self, env: InfraMindEnvironment, candidate: Action) -> float:
        replay = self._replay_env(env)
        start_reward = replay.reward_engine.cumulative_reward
        result = replay.step(candidate)
        for _ in range(self.rollout_depth):
            if result.done:
                break
            options = self.propose_actions(result.observation)
            if not options:
                break
            result = replay.step(self.rng.choice(options))
        return replay.reward_engine.cumulative_reward - start_reward

    def _replay_env(self, env: InfraMindEnvironment) -> InfraMindEnvironment:
        replay = InfraMindEnvironment(task_id=env.task_id, scenario_id=env.scenario_id)
        for item in env.action_trace:
            replay.step(Action.model_validate(item["action"]))
        return replay

    def _ucb_select(self, candidates: Sequence[Action], total_visits: int) -> Action:
        unexplored = [action for action in candidates if self.stats.get(self._key(action), ActionStats()).count == 0]
        if unexplored:
            return self.rng.choice(unexplored)
        return max(candidates, key=lambda action: self._ucb_score(action, total_visits))

    def _ucb_score(self, action: Action, total_visits: int) -> float:
        record = self.stats.get(self._key(action), ActionStats())
        if record.count == 0:
            return float("inf")
        return record.average_reward + self.exploration_constant * math.sqrt(math.log(total_visits) / record.count)

    def _key(self, action: Action) -> str:
        target = action.target_node_id or ":".join(action.target_edge or ())
        return f"{action.action_type}:{target}:{action.heal_strategy or ''}"


class LLMDiagnoser:
    def __init__(self) -> None:
        # Prefer HF_TOKEN provided via environment/secrets (no default).
        self.api_key = HF_TOKEN
        if not self.api_key:
            self.client = None
        else:
            # Configure the OpenAI client with the provided base URL and token
            self.client = OpenAI(base_url=API_BASE_URL, api_key=self.api_key)

    def diagnosis_action(self, observation: Observation, final: bool = False) -> Action:
        if not self.client:
            return self._fallback_action(observation, final=final)

        system_prompt = (
            "You are diagnosing a cascading cloud infrastructure failure. "
            "Return strict JSON matching the Action schema. Do not include markdown."
        )
        user_prompt = self._prompt_from_observation(observation, final=final)
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                content = response.choices[0].message.content or "{}"
                payload = json.loads(content)
                return Action.model_validate(payload)
            except Exception:
                if attempt == 2:
                    return self._fallback_action(observation, final=final)
                time.sleep(2**attempt)
        return self._fallback_action(observation, final=final)

    def _prompt_from_observation(self, observation: Observation, final: bool) -> str:
        adjacency = []
        for edge in observation.graph_snapshot.edges:
            adjacency.append(
                f"{edge.source_id} -> {edge.target_id} "
                f"[{edge.dependency_type}, weight={edge.health_weight:.2f}, broken={edge.is_circuit_broken}]"
            )
        node_lines = [
            f"{node.node_id}: health={node.health_score:.2f}, failed={node.is_failed}, degraded={node.is_degraded}, error={node.error_rate:.2f}"
            for node in observation.graph_snapshot.nodes
        ]
        instruction = (
            "Produce a submit_diagnosis action with root_causes, blast_radius, isolated_edges, and trap_triggered_nodes."
            if final
            else "Produce a hypothesize_root action with a single best root_node."
        )
        return (
            f"Task: {observation.task_name}\n"
            f"Description: {observation.task_description}\n"
            f"Step: {observation.step_number}, budget remaining: {observation.action_budget_remaining}\n"
            f"Nodes:\n" + "\n".join(node_lines[:18]) + "\n"
            f"Adjacency:\n" + "\n".join(adjacency[:24]) + "\n"
            f"Blast radius: {observation.graph_snapshot.blast_radius}\n"
            f"Wavefront: {observation.graph_snapshot.cascade_wavefront}\n"
            f"{instruction}\n"
            "The JSON must include reasoning."
        )

    def _fallback_action(self, observation: Observation, final: bool) -> Action:
        nodes = sorted(observation.graph_snapshot.nodes, key=lambda item: item.health_score)
        edges = sorted(observation.graph_snapshot.edges, key=lambda item: item.health_weight)
        roots = [node.node_id for node in nodes[:2]]
        if final:
            isolated = [
                [edge.source_id, edge.target_id]
                for edge in edges
                if edge.health_weight < 0.45 or edge.is_circuit_broken
            ][:3]
            return Action(
                action_type="submit_diagnosis",
                diagnosis={
                    "root_causes": roots,
                    "blast_radius": observation.graph_snapshot.blast_radius,
                    "isolated_edges": isolated,
                    "trap_triggered_nodes": [],
                },
                reasoning="Fallback heuristic diagnosis based on lowest-health nodes and weak edges.",
            )
        root = roots[0] if roots else None
        return Action(
            action_type="hypothesize_root",
            target_node_id=root,
            hypothesis={"root_node": root},
            reasoning="Fallback heuristic root-cause hypothesis.",
        )


def run_episode(task_id: str, scenario_id: int, planner: MCTSLitePlanner, diagnoser: LLMDiagnoser) -> Dict[str, Any]:
    env = InfraMindEnvironment(task_id=task_id, scenario_id=scenario_id)
    observation = env.reset(task_id, scenario_id)
    episode_actions: List[Dict[str, Any]] = []
    final_reward = 0.0

    while not env.done:
        if observation.step_number >= 10 or observation.action_budget_remaining <= 4:
            action = diagnoser.diagnosis_action(observation, final=True)
        elif observation.step_number >= 8:
            action = diagnoser.diagnosis_action(observation, final=False)
        else:
            action = planner.select_action(env)
        episode_actions.append(action.model_dump())
        result = env.step(action)
        observation = result.observation
        final_reward = result.reward.cumulative_reward
        if result.done:
            return {
                "task_id": task_id,
                "scenario_id": scenario_id,
                "score": round(result.reward.episode_reward or final_reward, 4),
                "success": result.reward.success,
                "counterfactual_score": result.reward.counterfactual_score,
                "steps": env.step_number,
                "actions": episode_actions,
            }
    return {
        "task_id": task_id,
        "scenario_id": scenario_id,
        "score": round(final_reward, 4),
        "success": env.success,
        "counterfactual_score": None,
        "steps": env.step_number,
        "actions": episode_actions,
    }


def aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Dict[str, Any]] = {}
    for task_id in {"easy", "medium", "hard"}:
        task_results = [item for item in results if item["task_id"] == task_id]
        summary[task_id] = {
            "episodes": len(task_results),
            "average_score": round(sum(item["score"] for item in task_results) / max(1, len(task_results)), 4),
            "success_rate": round(sum(1 for item in task_results if item["success"]) / max(1, len(task_results)), 4),
        }
    summary["overall"] = {
        "episodes": len(results),
        "average_score": round(sum(item["score"] for item in results) / max(1, len(results)), 4),
        "success_rate": round(sum(1 for item in results if item["success"]) / max(1, len(results)), 4),
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "hf_token_present": bool(HF_TOKEN),
    }
    return summary


def main() -> None:
    planner = MCTSLitePlanner()
    diagnoser = LLMDiagnoser()
    results: List[Dict[str, Any]] = []

    run_id = uuid.uuid4().hex
    _log_start(run_id)
    try:
        for task_id in ["easy", "medium", "hard"]:
            for scenario_id in [0, 1, 2]:
                _log_step(f"start_episode {task_id}:{scenario_id}")
                result = run_episode(task_id, scenario_id, planner, diagnoser)
                results.append(result)
                _log_step(f"end_episode {task_id}:{scenario_id} score={result['score']:.4f} success={result['success']}")
                print(
                    f"{task_id}:{scenario_id} "
                    f"score={result['score']:.4f} success={result['success']} "
                    f"counterfactual={result['counterfactual_score']}"
                )

        payload = {"episodes": results, "aggregate": aggregate(results)}
        RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload["aggregate"], indent=2))
        _log_end(run_id, status="SUCCESS")
    except Exception as exc:  # ensure END is always emitted for automated checks
        _log_step(f"error {str(exc)}")
        _log_end(run_id, status="FAILURE")
        raise


if __name__ == "__main__":
    main()
