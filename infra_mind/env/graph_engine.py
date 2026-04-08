from __future__ import annotations

import copy
import math
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from env.models import Action, EdgeHealth, GraphSnapshot, NodeHealth, NodeType


DECAY_RATES = {
    "sync_rpc": 0.15,
    "async_queue": 0.03,
    "db_connection": 0.20,
    "cache_read": 0.08,
}


class TemporalKnowledgeGraph:
    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        seed: int = 0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.graph = nx.DiGraph()
        self.seed = seed
        self.current_time = timestamp or datetime.utcnow()
        self.visibility_radius = 2
        self.node_visibility: Dict[str, int] = {}
        self.snapshot_history: List[GraphSnapshot] = []
        self.action_log: List[Dict[str, Any]] = []

        for node in nodes:
            self.graph.add_node(
                node["node_id"],
                type=node["node_type"],
                baseline_health=node.get("baseline_health", 1.0),
                current_health=node.get("current_health", node.get("baseline_health", 1.0)),
                failure_state=node.get("failure_state", "healthy"),
                failure_type=node.get("failure_type"),
                degraded_since=node.get("degraded_since"),
                deployment_history=list(node.get("deployment_history", [])),
                metrics=copy.deepcopy(node.get("metrics", {})),
                metadata=copy.deepcopy(node.get("metadata", {})),
                recovery_history=[],
            )

        for edge in edges:
            self.graph.add_edge(
                edge["source_id"],
                edge["target_id"],
                dependency_type=edge["dependency_type"],
                baseline_weight=edge.get("baseline_weight", 1.0),
                current_weight=edge.get("current_weight", edge.get("baseline_weight", 1.0)),
                circuit_breaker_state=edge.get("circuit_breaker_state", False),
                latency_p99_ms=edge.get("latency_p99_ms", 15.0),
                error_rate=edge.get("error_rate", 0.0),
                metadata=copy.deepcopy(edge.get("metadata", {})),
            )

        self._record_snapshot()

    @classmethod
    def from_snapshot(
        cls,
        snapshot: GraphSnapshot,
        seed: int = 0,
    ) -> "TemporalKnowledgeGraph":
        nodes = []
        edges = []
        for node in snapshot.nodes:
            nodes.append(
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "baseline_health": 1.0,
                    "current_health": node.health_score,
                    "failure_state": "failed"
                    if node.is_failed
                    else "degraded"
                    if node.is_degraded
                    else "healthy",
                    "failure_type": node.failure_type,
                    "degraded_since": node.time_since_degradation,
                    "metrics": {
                        "latency_ms": node.latency_ms,
                        "error_rate": node.error_rate,
                        "cpu_utilization": node.cpu_utilization,
                        "memory_utilization": node.memory_utilization,
                    },
                }
            )
        for edge in snapshot.edges:
            edges.append(
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "dependency_type": edge.dependency_type,
                    "baseline_weight": 1.0,
                    "current_weight": edge.health_weight,
                    "circuit_breaker_state": edge.is_circuit_broken,
                    "latency_p99_ms": edge.latency_p99_ms,
                    "error_rate": edge.error_rate,
                }
            )
        graph = cls(nodes=nodes, edges=edges, seed=seed, timestamp=snapshot.timestamp)
        graph.snapshot_history = [snapshot]
        return graph

    def clone(self) -> "TemporalKnowledgeGraph":
        node_payload = []
        edge_payload = []
        for node_id, attrs in self.graph.nodes(data=True):
            node_payload.append(
                {
                    "node_id": node_id,
                    "node_type": attrs["type"],
                    "baseline_health": attrs["baseline_health"],
                    "current_health": attrs["current_health"],
                    "failure_state": attrs["failure_state"],
                    "failure_type": attrs.get("failure_type"),
                    "degraded_since": attrs.get("degraded_since"),
                    "deployment_history": copy.deepcopy(attrs.get("deployment_history", [])),
                    "metrics": copy.deepcopy(attrs.get("metrics", {})),
                    "metadata": copy.deepcopy(attrs.get("metadata", {})),
                }
            )
        for source, target, attrs in self.graph.edges(data=True):
            edge_payload.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "dependency_type": attrs["dependency_type"],
                    "baseline_weight": attrs["baseline_weight"],
                    "current_weight": attrs["current_weight"],
                    "circuit_breaker_state": attrs["circuit_breaker_state"],
                    "latency_p99_ms": attrs["latency_p99_ms"],
                    "error_rate": attrs["error_rate"],
                    "metadata": copy.deepcopy(attrs.get("metadata", {})),
                }
            )
        clone = TemporalKnowledgeGraph(node_payload, edge_payload, seed=self.seed, timestamp=self.current_time)
        clone.visibility_radius = self.visibility_radius
        clone.node_visibility = copy.deepcopy(self.node_visibility)
        clone.snapshot_history = copy.deepcopy(self.snapshot_history)
        clone.action_log = copy.deepcopy(self.action_log)
        return clone

    def set_visibility_center(self, node_id: str, radius: int = 2) -> None:
        self.visibility_radius = radius
        self.node_visibility[node_id] = max(radius, self.node_visibility.get(node_id, 0))

    def get_most_anomalous_node(self) -> str:
        ranked = sorted(
            self.graph.nodes(data=True),
            key=lambda item: (
                item[1]["current_health"],
                -item[1].get("metrics", {}).get("error_rate", 0.0),
                -item[1].get("metrics", {}).get("latency_ms", 0.0),
            ),
        )
        return ranked[0][0]

    def get_subgraph_around(self, node_id: str, radius: int) -> GraphSnapshot:
        visited = {node_id}
        frontier = deque([(node_id, 0)])
        while frontier:
            current, depth = frontier.popleft()
            if depth >= radius:
                continue
            neighbors = set(self.graph.predecessors(current)) | set(self.graph.successors(current))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, depth + 1))

        nodes = [self._node_to_model(candidate) for candidate in visited]
        edges = []
        for source, target in self.graph.edges():
            if source in visited and target in visited:
                edges.append(self._edge_to_model(source, target))
        return GraphSnapshot(
            timestamp=self.current_time,
            nodes=sorted(nodes, key=lambda item: item.node_id),
            edges=sorted(edges, key=lambda item: (item.source_id, item.target_id)),
            cascade_wavefront=self.get_cascade_wavefront(),
            blast_radius=self.get_blast_radius(),
        )

    def get_temporal_snapshots(self, n: int = 5) -> List[GraphSnapshot]:
        return copy.deepcopy(self.snapshot_history[-n:])

    def update_edge_weights(self, elapsed_seconds: float = 30.0) -> None:
        for source, target, attrs in self.graph.edges(data=True):
            if attrs.get("circuit_breaker_state"):
                attrs["current_weight"] = 0.0
                continue
            target_state = self.graph.nodes[target]["failure_state"]
            source_state = self.graph.nodes[source]["failure_state"]
            failing = target_state in {"degraded", "failed"} or source_state in {"degraded", "failed"}
            if not failing:
                attrs["current_weight"] = min(attrs["baseline_weight"], attrs["current_weight"] + 0.01)
                attrs["error_rate"] = max(0.0, attrs["error_rate"] - 0.01)
                continue
            decay = DECAY_RATES.get(attrs["dependency_type"], 0.05)
            baseline = attrs["baseline_weight"]
            attrs["current_weight"] = max(0.0, baseline * math.exp(-decay * (elapsed_seconds / 30.0)))
            attrs["error_rate"] = min(1.0, 1.0 - attrs["current_weight"])
            attrs["latency_p99_ms"] = max(attrs["latency_p99_ms"], 25.0 + (1.0 - attrs["current_weight"]) * 180.0)

    def apply_action(self, action: Action) -> Tuple[bool, str]:
        success = False
        side_effects = "Action had no effect."

        if action.action_type == "explore_node":
            if not action.target_node_id or action.target_node_id not in self.graph:
                side_effects = "Target node not found for exploration."
            else:
                current = self.node_visibility.get(action.target_node_id, self.visibility_radius)
                self.node_visibility[action.target_node_id] = current + 1
                self.visibility_radius = max(self.visibility_radius, current + 1)
                success = True
                side_effects = f"Expanded visibility around {action.target_node_id} to radius {current + 1}."

        elif action.action_type == "query_dependency":
            if not action.target_edge or not self.graph.has_edge(*action.target_edge):
                side_effects = "Dependency edge not found."
            else:
                attrs = self.graph.edges[action.target_edge]
                success = True
                side_effects = (
                    f"Dependency {action.target_edge[0]}->{action.target_edge[1]} "
                    f"weight={attrs['current_weight']:.2f}, circuit_broken={attrs['circuit_breaker_state']}."
                )

        elif action.action_type == "hypothesize_root":
            candidate = (action.hypothesis or {}).get("root_node") or action.target_node_id
            if candidate and candidate in self.graph:
                self.graph.nodes[candidate]["metadata"]["hypothesized_root"] = True
                success = True
                side_effects = f"Stored {candidate} as a root-cause hypothesis."
            else:
                side_effects = "Root-cause hypothesis did not reference a valid node."

        elif action.action_type == "execute_heal":
            success, side_effects = self._heal_node(action.target_node_id, action.heal_strategy or "smart_heal")

        elif action.action_type == "circuit_break":
            if not action.target_edge or not self.graph.has_edge(*action.target_edge):
                side_effects = "Circuit-break target edge not found."
            else:
                edge = self.graph.edges[action.target_edge]
                edge["circuit_breaker_state"] = True
                edge["current_weight"] = 0.0
                edge["error_rate"] = min(1.0, edge["error_rate"] + 0.05)
                success = True
                side_effects = f"Circuit breaker tripped on {action.target_edge[0]}->{action.target_edge[1]}."

        elif action.action_type == "rollback_deployment":
            if not action.target_node_id or action.target_node_id not in self.graph:
                side_effects = "Rollback target node not found."
            else:
                node = self.graph.nodes[action.target_node_id]
                history = node.get("deployment_history", [])
                if history:
                    rolled_back = history.pop()
                    node["current_health"] = min(1.0, node["current_health"] + 0.18)
                    self._normalize_node_metrics(action.target_node_id)
                    success = True
                    side_effects = f"Rolled back deployment {rolled_back} on {action.target_node_id}."
                else:
                    side_effects = f"No deployment history available on {action.target_node_id}."

        elif action.action_type == "scale_out":
            if not action.target_node_id or action.target_node_id not in self.graph:
                side_effects = "Scale-out target node not found."
            else:
                node = self.graph.nodes[action.target_node_id]
                node["metadata"]["replica_count"] = node["metadata"].get("replica_count", 1) + 1
                node["current_health"] = min(1.0, node["current_health"] + 0.12)
                metrics = node["metrics"]
                metrics["cpu_utilization"] = max(0.0, metrics.get("cpu_utilization", 0.5) - 0.18)
                metrics["memory_utilization"] = max(0.0, metrics.get("memory_utilization", 0.5) - 0.12)
                success = True
                side_effects = f"Scaled out {action.target_node_id} to {node['metadata']['replica_count']} replicas."

        elif action.action_type == "restart_service":
            if not action.target_node_id or action.target_node_id not in self.graph:
                side_effects = "Restart target node not found."
            else:
                node = self.graph.nodes[action.target_node_id]
                node["metadata"]["last_restart"] = self.current_time.isoformat()
                health_before = node["current_health"]
                node["current_health"] = max(0.0, node["current_health"] - 0.08)
                self._normalize_node_metrics(action.target_node_id, restart=True)
                success = True
                side_effects = (
                    f"Restarted {action.target_node_id}; transient health dip "
                    f"{health_before:.2f}->{node['current_health']:.2f}."
                )

        elif action.action_type == "submit_diagnosis":
            success = True
            side_effects = "Final diagnosis submitted."

        self.action_log.append(
            {
                "time": self.current_time.isoformat(),
                "action": action.model_dump(),
                "success": success,
                "side_effects": side_effects,
            }
        )
        self._record_snapshot()
        return success, side_effects

    def get_cascade_wavefront(self) -> List[str]:
        wavefront = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs["failure_state"] != "degraded":
                continue
            has_failed_neighbor = any(
                self.graph.nodes[neighbor]["failure_state"] == "failed"
                for neighbor in set(self.graph.predecessors(node_id)) | set(self.graph.successors(node_id))
            )
            if has_failed_neighbor:
                wavefront.append(node_id)
        return sorted(set(wavefront))

    def get_blast_radius(self) -> List[str]:
        return sorted(
            node_id
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs["current_health"] < 0.7
        )

    def advance_time(self, step_seconds: float = 30.0) -> None:
        self.current_time += timedelta(seconds=step_seconds)
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs["failure_state"] == "recovering":
                attrs["current_health"] = min(1.0, attrs["current_health"] + 0.1)
                if attrs["current_health"] >= 0.85:
                    attrs["failure_state"] = "healthy"
                    attrs["failure_type"] = None
                    attrs["degraded_since"] = None
            self._normalize_node_metrics(node_id)
        self.update_edge_weights(step_seconds)
        self._record_snapshot()

    def inject_failure(self, node_id: str, failure_type: str, severity: float = 0.5) -> None:
        if node_id not in self.graph:
            return
        node = self.graph.nodes[node_id]
        node["failure_state"] = "failed" if severity >= 0.55 else "degraded"
        node["failure_type"] = failure_type
        node["current_health"] = max(0.0, node["current_health"] - severity)
        node["degraded_since"] = 0.0
        self._normalize_node_metrics(node_id)
        self.update_edge_weights()
        self._record_snapshot()

    def full_snapshot(self) -> GraphSnapshot:
        return GraphSnapshot(
            timestamp=self.current_time,
            nodes=sorted([self._node_to_model(node_id) for node_id in self.graph.nodes], key=lambda item: item.node_id),
            edges=sorted(
                [self._edge_to_model(source, target) for source, target in self.graph.edges],
                key=lambda item: (item.source_id, item.target_id),
            ),
            cascade_wavefront=self.get_cascade_wavefront(),
            blast_radius=self.get_blast_radius(),
        )

    def _heal_node(self, node_id: Optional[str], strategy: str) -> Tuple[bool, str]:
        if not node_id or node_id not in self.graph:
            return False, "Heal target node not found."
        node = self.graph.nodes[node_id]
        delta = {
            "smart_heal": 0.22,
            "patch": 0.18,
            "failover": 0.25,
            "restart": 0.14,
        }.get(strategy, 0.16)
        before = node["current_health"]
        node["current_health"] = min(1.0, node["current_health"] + delta)
        node["failure_state"] = "recovering" if node["current_health"] < 0.85 else "healthy"
        node["failure_type"] = None if node["current_health"] >= 0.75 else node["failure_type"]
        node["degraded_since"] = 0.0 if node["failure_state"] != "healthy" else None
        node["recovery_history"].append({"time": self.current_time.isoformat(), "strategy": strategy})
        self._normalize_node_metrics(node_id)
        improved = node["current_health"] - before
        return improved > 0.0, f"Healed {node_id} with {strategy}; health {before:.2f}->{node['current_health']:.2f}."

    def _normalize_node_metrics(self, node_id: str, restart: bool = False) -> None:
        node = self.graph.nodes[node_id]
        health = node["current_health"]
        metrics = node["metrics"]
        metrics["latency_ms"] = max(5.0, 20.0 + (1.0 - health) * 280.0 + (30.0 if restart else 0.0))
        metrics["error_rate"] = min(1.0, max(metrics.get("error_rate", 0.0), 1.0 - health))
        metrics["cpu_utilization"] = min(1.0, max(0.05, metrics.get("cpu_utilization", 0.3), 0.25 + (1.0 - health) * 0.8))
        metrics["memory_utilization"] = min(
            1.0,
            max(0.05, metrics.get("memory_utilization", 0.3), 0.2 + (1.0 - health) * 0.75),
        )

    def _node_to_model(self, node_id: str) -> NodeHealth:
        attrs = self.graph.nodes[node_id]
        metrics = attrs["metrics"]
        degraded_since = attrs.get("degraded_since")
        if degraded_since is not None and isinstance(degraded_since, (int, float)):
            time_since = degraded_since
        else:
            time_since = None
        return NodeHealth(
            node_id=node_id,
            node_type=NodeType(attrs["type"]),
            health_score=round(attrs["current_health"], 4),
            latency_ms=round(metrics.get("latency_ms", 10.0), 2),
            error_rate=round(metrics.get("error_rate", 0.0), 4),
            cpu_utilization=round(metrics.get("cpu_utilization", 0.2), 4),
            memory_utilization=round(metrics.get("memory_utilization", 0.2), 4),
            is_degraded=attrs["failure_state"] in {"degraded", "recovering"},
            is_failed=attrs["failure_state"] == "failed",
            failure_type=attrs.get("failure_type"),
            time_since_degradation=time_since,
        )

    def _edge_to_model(self, source: str, target: str) -> EdgeHealth:
        attrs = self.graph.edges[(source, target)]
        return EdgeHealth(
            source_id=source,
            target_id=target,
            dependency_type=attrs["dependency_type"],
            health_weight=round(attrs["current_weight"], 4),
            latency_p99_ms=round(attrs["latency_p99_ms"], 2),
            error_rate=round(attrs["error_rate"], 4),
            is_circuit_broken=attrs["circuit_breaker_state"],
        )

    def _record_snapshot(self) -> None:
        self.snapshot_history.append(self.full_snapshot())
        if len(self.snapshot_history) > 25:
            self.snapshot_history = self.snapshot_history[-25:]
