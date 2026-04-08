from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from env.cascade_simulator import CascadeSimulator
from env.graph_engine import TemporalKnowledgeGraph


SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


@dataclass
class TaskConfig:
    task_id: str
    scenario_count: int
    max_steps: int


class ScenarioGenerator:
    def __init__(self) -> None:
        self.configs = {
            "easy": TaskConfig(task_id="easy", scenario_count=10, max_steps=7),
            "medium": TaskConfig(task_id="medium", scenario_count=10, max_steps=10),
            "hard": TaskConfig(task_id="hard", scenario_count=10, max_steps=12),
        }

    def generate_all(self) -> Dict[str, List[Dict[str, Any]]]:
        SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
        generated: Dict[str, List[Dict[str, Any]]] = {}
        for task_id, config in self.configs.items():
            scenarios = []
            for scenario_id in range(config.scenario_count):
                seed = self._seed_for(task_id, scenario_id)
                if task_id == "easy":
                    scenario = self.generate_easy_scenario(scenario_id, seed, config.max_steps)
                elif task_id == "medium":
                    scenario = self.generate_medium_scenario(scenario_id, seed, config.max_steps)
                else:
                    scenario = self.generate_hard_scenario(scenario_id, seed, config.max_steps)
                scenarios.append(scenario)
                output_path = SCENARIO_DIR / f"{task_id}_{scenario_id}.json"
                output_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
            generated[task_id] = scenarios
        return generated

    def generate_easy_scenario(self, scenario_id: int, seed: int, max_steps: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        nodes = [
            self._node("nginx-lb", "load_balancer", rng),
            self._node("api-gateway", "api_gateway", rng),
            self._node("auth-service", "microservice", rng),
            self._node("order-service", "microservice", rng),
            self._node("payment-service", "microservice", rng),
            self._node("inventory-service", "microservice", rng),
            self._node("postgres-primary", "database", rng),
            self._node("postgres-replica", "database", rng),
            self._node("redis-cache", "cache", rng),
            self._node("session-cache", "cache", rng),
            self._node("rabbitmq", "message_queue", rng),
            self._node("s3-compatible-storage", "storage", rng),
        ]
        edges = [
            self._edge("postgres-primary", "auth-service", "db_connection", rng),
            self._edge("postgres-primary", "order-service", "db_connection", rng),
            self._edge("postgres-primary", "payment-service", "db_connection", rng),
            self._edge("postgres-replica", "inventory-service", "db_connection", rng),
            self._edge("redis-cache", "auth-service", "cache_read", rng),
            self._edge("session-cache", "order-service", "cache_read", rng),
            self._edge("rabbitmq", "inventory-service", "async_queue", rng),
            self._edge("auth-service", "api-gateway", "sync_rpc", rng),
            self._edge("order-service", "api-gateway", "sync_rpc", rng),
            self._edge("payment-service", "api-gateway", "sync_rpc", rng),
            self._edge("inventory-service", "api-gateway", "sync_rpc", rng),
            self._edge("api-gateway", "nginx-lb", "sync_rpc", rng),
            self._edge("s3-compatible-storage", "payment-service", "sync_rpc", rng),
        ]
        root_nodes = ["postgres-primary"]
        failure_types = ["oom"]
        optimal_actions = [
            self._action_stub(1, "hypothesize_root", target_node_id="postgres-primary", hypothesis={"root_node": "postgres-primary"}),
            self._action_stub(2, "execute_heal", target_node_id="postgres-primary", heal_strategy="failover"),
            self._action_stub(3, "execute_heal", target_node_id="order-service", heal_strategy="smart_heal"),
            self._action_stub(4, "execute_heal", target_node_id="payment-service", heal_strategy="smart_heal"),
        ]
        return self._build_scenario(
            scenario_id=scenario_id,
            task_id="easy",
            seed=seed,
            nodes=nodes,
            edges=edges,
            root_nodes=root_nodes,
            failure_types=failure_types,
            red_herrings=[],
            healing_traps=[],
            expected_optimal_actions=optimal_actions,
            max_steps=max_steps,
        )

    def generate_medium_scenario(self, scenario_id: int, seed: int, max_steps: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        nodes = [
            self._node("nginx-lb", "load_balancer", rng),
            self._node("api-gateway", "api_gateway", rng),
            self._node("rabbitmq", "message_queue", rng),
            self._node("cdn-edge", "cdn", rng),
            self._node("s3-compatible-storage", "storage", rng),
            self._node("auth-service-a", "microservice", rng),
            self._node("order-service-a", "microservice", rng),
            self._node("payment-service-a", "microservice", rng),
            self._node("inventory-service-a", "microservice", rng),
            self._node("cart-service-a", "microservice", rng),
            self._node("pricing-service-a", "microservice", rng),
            self._node("checkout-worker-a", "microservice", rng),
            self._node("postgres-primary-a", "database", rng),
            self._node("postgres-replica-a", "database", rng),
            self._node("redis-cache-a", "cache", rng),
            self._node("checkout-cache-a", "cache", rng),
            self._node("notification-service-b", "microservice", rng),
            self._node("search-service-b", "microservice", rng),
            self._node("recommendation-engine-b", "microservice", rng),
            self._node("analytics-pipeline-b", "microservice", rng),
            self._node("fraud-detector-b", "microservice", rng),
            self._node("event-consumer-b", "microservice", rng),
            self._node("stream-worker-b", "microservice", rng),
            self._node("catalog-service-b", "microservice", rng),
            self._node("postgres-primary-b", "database", rng),
            self._node("redis-cache-b", "cache", rng),
            self._node("elasticsearch-b", "database", rng),
            self._node("user-profile-service-c", "microservice", rng),
            self._node("canary-api-c", "microservice", rng),
            self._node("canary-worker-c", "microservice", rng),
            self._node("config-service-c", "microservice", rng),
            self._node("postgres-primary-c", "database", rng),
            self._node("redis-cache-c", "cache", rng),
            self._node("session-cache-c", "cache", rng),
            self._node("admin-portal-c", "microservice", rng),
        ]
        edges = [
            self._edge("cdn-edge", "nginx-lb", "sync_rpc", rng),
            self._edge("nginx-lb", "api-gateway", "sync_rpc", rng),
            self._edge("postgres-primary-a", "auth-service-a", "db_connection", rng),
            self._edge("postgres-primary-a", "order-service-a", "db_connection", rng),
            self._edge("postgres-primary-a", "payment-service-a", "db_connection", rng),
            self._edge("redis-cache-a", "cart-service-a", "cache_read", rng),
            self._edge("checkout-cache-a", "checkout-worker-a", "cache_read", rng),
            self._edge("auth-service-a", "api-gateway", "sync_rpc", rng),
            self._edge("order-service-a", "api-gateway", "sync_rpc", rng),
            self._edge("payment-service-a", "api-gateway", "sync_rpc", rng),
            self._edge("inventory-service-a", "api-gateway", "sync_rpc", rng),
            self._edge("cart-service-a", "api-gateway", "sync_rpc", rng),
            self._edge("pricing-service-a", "api-gateway", "sync_rpc", rng),
            self._edge("payment-service-a", "rabbitmq", "async_queue", rng, bias=0.85),
            self._edge("checkout-worker-a", "rabbitmq", "async_queue", rng, bias=0.88),
            self._edge("rabbitmq", "event-consumer-b", "async_queue", rng, bias=0.95),
            self._edge("event-consumer-b", "notification-service-b", "sync_rpc", rng),
            self._edge("event-consumer-b", "analytics-pipeline-b", "sync_rpc", rng),
            self._edge("notification-service-b", "api-gateway", "sync_rpc", rng),
            self._edge("search-service-b", "api-gateway", "sync_rpc", rng),
            self._edge("recommendation-engine-b", "api-gateway", "sync_rpc", rng),
            self._edge("analytics-pipeline-b", "fraud-detector-b", "sync_rpc", rng),
            self._edge("redis-cache-b", "recommendation-engine-b", "cache_read", rng),
            self._edge("postgres-primary-b", "catalog-service-b", "db_connection", rng),
            self._edge("elasticsearch-b", "search-service-b", "db_connection", rng),
            self._edge("canary-api-c", "api-gateway", "sync_rpc", rng),
            self._edge("canary-worker-c", "config-service-c", "sync_rpc", rng),
            self._edge("postgres-primary-c", "user-profile-service-c", "db_connection", rng),
            self._edge("redis-cache-c", "admin-portal-c", "cache_read", rng),
            self._edge("session-cache-c", "user-profile-service-c", "cache_read", rng),
            self._edge("s3-compatible-storage", "analytics-pipeline-b", "sync_rpc", rng),
        ]
        red_herrings = ["canary-api-c", "canary-worker-c", "config-service-c"]
        root_nodes = ["postgres-primary-a"]
        failure_types = ["oom"]
        optimal_actions = [
            self._action_stub(1, "hypothesize_root", target_node_id="postgres-primary-a", hypothesis={"root_node": "postgres-primary-a"}),
            self._action_stub(2, "circuit_break", target_edge=("rabbitmq", "event-consumer-b")),
            self._action_stub(3, "execute_heal", target_node_id="postgres-primary-a", heal_strategy="failover"),
            self._action_stub(4, "submit_diagnosis", diagnosis={"root_causes": ["postgres-primary-a"], "blast_radius": ["cluster-a", "cluster-b"]}),
        ]
        return self._build_scenario(
            scenario_id=scenario_id,
            task_id="medium",
            seed=seed,
            nodes=nodes,
            edges=edges,
            root_nodes=root_nodes,
            failure_types=failure_types,
            red_herrings=red_herrings,
            healing_traps=[],
            expected_optimal_actions=optimal_actions,
            max_steps=max_steps,
        )

    def generate_hard_scenario(self, scenario_id: int, seed: int, max_steps: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        regions = ["us-east-1", "eu-west-1", "ap-south-1"]
        regional_nodes: List[Dict[str, Any]] = []
        regional_edges: List[Dict[str, Any]] = []
        for region in regions:
            regional_nodes.extend(
                [
                    self._node(f"auth-service-{region}", "microservice", rng),
                    self._node(f"order-service-{region}", "microservice", rng),
                    self._node(f"payment-service-{region}", "microservice", rng),
                    self._node(f"inventory-service-{region}", "microservice", rng),
                    self._node(f"notification-service-{region}", "microservice", rng),
                    self._node(f"user-profile-service-{region}", "microservice", rng),
                    self._node(f"search-service-{region}", "microservice", rng),
                    self._node(f"recommendation-engine-{region}", "microservice", rng),
                    self._node(f"analytics-pipeline-{region}", "microservice", rng),
                    self._node(f"fraud-detector-{region}", "microservice", rng),
                    self._node(f"postgres-primary-{region}", "database", rng),
                    self._node(f"postgres-replica-{region}", "database", rng),
                    self._node(f"redis-cache-{region}", "cache", rng),
                    self._node(f"search-cache-{region}", "cache", rng),
                    self._node(f"rabbitmq-{region}", "message_queue", rng),
                    self._node(f"nginx-lb-{region}", "load_balancer", rng),
                    self._node(f"s3-compatible-storage-{region}", "storage", rng),
                    self._node(f"cdn-edge-{region}", "cdn", rng),
                    self._node(f"checkout-worker-{region}", "microservice", rng),
                    self._node(f"billing-worker-{region}", "microservice", rng),
                    self._node(f"catalog-service-{region}", "microservice", rng),
                    self._node(f"session-service-{region}", "microservice", rng),
                    self._node(f"feature-flag-service-{region}", "microservice", rng),
                    self._node(f"quota-manager-{region}", "microservice", rng),
                ]
            )
            regional_edges.extend(
                [
                    self._edge(f"postgres-primary-{region}", f"auth-service-{region}", "db_connection", rng),
                    self._edge(f"postgres-primary-{region}", f"order-service-{region}", "db_connection", rng),
                    self._edge(f"postgres-primary-{region}", f"payment-service-{region}", "db_connection", rng),
                    self._edge(f"postgres-primary-{region}", f"user-profile-service-{region}", "db_connection", rng),
                    self._edge(f"redis-cache-{region}", f"session-service-{region}", "cache_read", rng),
                    self._edge(f"search-cache-{region}", f"recommendation-engine-{region}", "cache_read", rng),
                    self._edge(f"rabbitmq-{region}", f"analytics-pipeline-{region}", "async_queue", rng),
                    self._edge(f"rabbitmq-{region}", f"notification-service-{region}", "async_queue", rng),
                    self._edge(f"auth-service-{region}", f"nginx-lb-{region}", "sync_rpc", rng),
                    self._edge(f"order-service-{region}", f"nginx-lb-{region}", "sync_rpc", rng),
                    self._edge(f"payment-service-{region}", f"nginx-lb-{region}", "sync_rpc", rng),
                    self._edge(f"inventory-service-{region}", f"nginx-lb-{region}", "sync_rpc", rng),
                    self._edge(f"notification-service-{region}", f"nginx-lb-{region}", "sync_rpc", rng),
                    self._edge(f"user-profile-service-{region}", f"nginx-lb-{region}", "sync_rpc", rng),
                    self._edge(f"search-service-{region}", f"nginx-lb-{region}", "sync_rpc", rng),
                    self._edge(f"recommendation-engine-{region}", f"nginx-lb-{region}", "sync_rpc", rng),
                    self._edge(f"analytics-pipeline-{region}", f"fraud-detector-{region}", "sync_rpc", rng),
                    self._edge(f"fraud-detector-{region}", f"checkout-worker-{region}", "sync_rpc", rng),
                    self._edge(f"catalog-service-{region}", f"search-service-{region}", "sync_rpc", rng),
                    self._edge(f"feature-flag-service-{region}", f"payment-service-{region}", "sync_rpc", rng),
                    self._edge(f"quota-manager-{region}", f"api-proxy-{region}", "sync_rpc", rng) if False else self._edge(f"quota-manager-{region}", f"billing-worker-{region}", "sync_rpc", rng),
                    self._edge(f"s3-compatible-storage-{region}", f"analytics-pipeline-{region}", "sync_rpc", rng),
                    self._edge(f"cdn-edge-{region}", f"nginx-lb-{region}", "sync_rpc", rng),
                ]
            )

        global_nodes = [
            self._node("global-api-gateway", "api_gateway", rng),
            self._node("global-nginx-lb", "load_balancer", rng),
            self._node("global-rabbitmq", "message_queue", rng),
            self._node("global-cdn-edge", "cdn", rng),
            self._node("audit-storage", "storage", rng),
            self._node("deployment-controller", "microservice", rng),
            self._node("config-control-plane", "microservice", rng),
            self._node("traffic-manager", "microservice", rng),
        ]
        global_edges = [
            self._edge("global-cdn-edge", "global-nginx-lb", "sync_rpc", rng),
            self._edge("global-nginx-lb", "global-api-gateway", "sync_rpc", rng),
            self._edge("global-rabbitmq", "deployment-controller", "async_queue", rng),
            self._edge("config-control-plane", "deployment-controller", "sync_rpc", rng),
            self._edge("audit-storage", "deployment-controller", "sync_rpc", rng),
            self._edge("traffic-manager", "global-api-gateway", "sync_rpc", rng),
        ]
        for region in regions:
            global_edges.extend(
                [
                    self._edge(f"nginx-lb-{region}", "global-api-gateway", "sync_rpc", rng),
                    self._edge(f"rabbitmq-{region}", "global-rabbitmq", "async_queue", rng, bias=0.92),
                    self._edge("deployment-controller", f"feature-flag-service-{region}", "sync_rpc", rng),
                    self._edge("config-control-plane", f"session-service-{region}", "sync_rpc", rng),
                    self._edge("traffic-manager", f"nginx-lb-{region}", "sync_rpc", rng),
                ]
            )

        nodes = regional_nodes + global_nodes
        edges = regional_edges + global_edges
        trap_one = f"payment-service-{regions[0]}"
        trap_two = f"search-service-{regions[1]}"
        trap_target_one = f"fraud-detector-{regions[0]}"
        trap_target_two = f"recommendation-engine-{regions[1]}"
        for node in nodes:
            if node["node_id"] == trap_one:
                node["metadata"]["restart_trap_target"] = trap_target_one
            if node["node_id"] == trap_two:
                node["metadata"]["restart_trap_target"] = trap_target_two

        red_herrings = [
            f"feature-flag-service-{regions[2]}",
            f"quota-manager-{regions[2]}",
            f"session-service-{regions[2]}",
            f"catalog-service-{regions[2]}",
            f"analytics-pipeline-{regions[1]}",
            "deployment-controller",
        ]
        root_nodes = [f"postgres-primary-{regions[0]}", f"rabbitmq-{regions[1]}"]
        failure_types = ["oom", "timeout"]
        optimal_actions = [
            self._action_stub(1, "hypothesize_root", target_node_id=root_nodes[0], hypothesis={"root_node": root_nodes[0]}),
            self._action_stub(2, "hypothesize_root", target_node_id=root_nodes[1], hypothesis={"root_node": root_nodes[1]}),
            self._action_stub(3, "circuit_break", target_edge=(f"rabbitmq-{regions[1]}", "global-rabbitmq")),
            self._action_stub(4, "execute_heal", target_node_id=root_nodes[0], heal_strategy="failover"),
            self._action_stub(5, "scale_out", target_node_id=f"payment-service-{regions[0]}"),
        ]
        return self._build_scenario(
            scenario_id=scenario_id,
            task_id="hard",
            seed=seed,
            nodes=nodes,
            edges=edges,
            root_nodes=root_nodes,
            failure_types=failure_types,
            red_herrings=red_herrings,
            healing_traps=[trap_one, trap_two],
            expected_optimal_actions=optimal_actions,
            max_steps=max_steps,
        )

    def _build_scenario(
        self,
        scenario_id: int,
        task_id: str,
        seed: int,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        root_nodes: Sequence[str],
        failure_types: Sequence[str],
        red_herrings: Sequence[str],
        healing_traps: Sequence[str],
        expected_optimal_actions: Sequence[Dict[str, Any]],
        max_steps: int,
    ) -> Dict[str, Any]:
        graph = TemporalKnowledgeGraph(nodes=nodes, edges=edges, seed=seed)
        simulator = CascadeSimulator(
            graph=graph,
            seed=seed,
            red_herring_nodes=red_herrings,
            acceleration_after_step=15 if task_id == "hard" else None,
            acceleration_factor=1.35 if task_id == "hard" else 1.0,
        )
        simulator.inject_failures(root_nodes, failure_types)
        if red_herrings:
            simulator.seed_red_herrings(red_herrings)
        simulator.run(max_steps)
        final_snapshot = simulator.graph.full_snapshot()
        return {
            "scenario_id": scenario_id,
            "task_id": task_id,
            "seed": seed,
            "graph": {"nodes": nodes, "edges": edges},
            "failure_injection": {
                "root_nodes": list(root_nodes),
                "injection_time": 0,
                "failure_types": list(failure_types),
            },
            "red_herrings": list(red_herrings),
            "cascade_trajectory": simulator.trajectory,
            "healing_traps": list(healing_traps),
            "ground_truth_blast_radius": final_snapshot.blast_radius,
            "expected_optimal_actions": list(expected_optimal_actions),
        }

    def _node(self, node_id: str, node_type: str, rng: random.Random) -> Dict[str, Any]:
        latency_base = {
            "database": 18.0,
            "cache": 5.0,
            "message_queue": 12.0,
            "load_balancer": 8.0,
            "cdn": 7.0,
            "storage": 22.0,
        }.get(node_type, 14.0)
        deployment_history = [f"deploy-{i}" for i in range(rng.randint(1, 3))]
        return {
            "node_id": node_id,
            "node_type": node_type,
            "baseline_health": 1.0,
            "current_health": 1.0,
            "failure_state": "healthy",
            "failure_type": None,
            "degraded_since": None,
            "deployment_history": deployment_history,
            "metrics": {
                "latency_ms": round(latency_base + rng.random() * 10.0, 2),
                "error_rate": round(rng.random() * 0.03, 4),
                "cpu_utilization": round(0.2 + rng.random() * 0.25, 4),
                "memory_utilization": round(0.25 + rng.random() * 0.25, 4),
            },
            "metadata": {"replica_count": 2 if node_type == "microservice" else 1},
        }

    def _edge(
        self,
        source_id: str,
        target_id: str,
        dependency_type: str,
        rng: random.Random,
        bias: float = 1.0,
    ) -> Dict[str, Any]:
        baseline_weight = min(1.0, max(0.72, bias * (0.82 + rng.random() * 0.16)))
        return {
            "source_id": source_id,
            "target_id": target_id,
            "dependency_type": dependency_type,
            "baseline_weight": round(baseline_weight, 4),
            "current_weight": round(baseline_weight, 4),
            "circuit_breaker_state": False,
            "latency_p99_ms": round(25.0 + rng.random() * 30.0, 2),
            "error_rate": round(rng.random() * 0.03, 4),
            "metadata": {},
        }

    def _action_stub(
        self,
        step: int,
        action_type: str,
        target_node_id: str | None = None,
        target_edge: tuple[str, str] | None = None,
        hypothesis: Dict[str, Any] | None = None,
        heal_strategy: str | None = None,
        diagnosis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "step": step,
            "action": {
                "action_type": action_type,
                "target_node_id": target_node_id,
                "target_edge": list(target_edge) if target_edge else None,
                "hypothesis": hypothesis,
                "heal_strategy": heal_strategy,
                "diagnosis": diagnosis,
                "reasoning": "synthetic optimal action",
            },
        }

    def _seed_for(self, task_id: str, scenario_id: int) -> int:
        base = {"easy": 1100, "medium": 2200, "hard": 3300}[task_id]
        return base + scenario_id


def main() -> None:
    generator = ScenarioGenerator()
    generated = generator.generate_all()
    summary = {task_id: len(scenarios) for task_id, scenarios in generated.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
