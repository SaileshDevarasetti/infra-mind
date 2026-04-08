from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    API_GATEWAY = "api_gateway"
    MICROSERVICE = "microservice"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    LOAD_BALANCER = "load_balancer"
    CDN = "cdn"
    STORAGE = "storage"


class NodeHealth(BaseModel):
    node_id: str
    node_type: NodeType
    health_score: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    cpu_utilization: float = Field(ge=0.0, le=1.0)
    memory_utilization: float = Field(ge=0.0, le=1.0)
    is_degraded: bool
    is_failed: bool
    failure_type: Optional[str] = Field(
        default=None,
        description='"oom" | "timeout" | "crash" | "saturation" | None',
    )
    time_since_degradation: Optional[float] = Field(default=None, ge=0.0)


class EdgeHealth(BaseModel):
    source_id: str
    target_id: str
    dependency_type: str = Field(
        description='"sync_rpc" | "async_queue" | "db_connection" | "cache_read"'
    )
    health_weight: float = Field(ge=0.0, le=1.0)
    latency_p99_ms: float = Field(ge=0.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    is_circuit_broken: bool


class GraphSnapshot(BaseModel):
    timestamp: datetime
    nodes: List[NodeHealth]
    edges: List[EdgeHealth]
    cascade_wavefront: List[str]
    blast_radius: List[str]


class Observation(BaseModel):
    task_id: str
    task_name: str
    task_description: str
    graph_snapshot: GraphSnapshot
    visible_subgraph_radius: int
    historical_snapshots: List[GraphSnapshot]
    available_actions: List[str]
    action_budget_remaining: int
    time_elapsed_seconds: float
    step_number: int
    previous_action_result: Optional[str]


class Action(BaseModel):
    action_type: Literal[
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
    target_node_id: Optional[str] = None
    target_edge: Optional[Tuple[str, str]] = None
    hypothesis: Optional[Dict[str, Any]] = None
    heal_strategy: Optional[str] = None
    diagnosis: Optional[Dict[str, Any]] = None
    reasoning: str


class Reward(BaseModel):
    step_reward: float
    cumulative_reward: float
    episode_reward: Optional[float] = None
    counterfactual_score: Optional[float] = None
    feedback: str
    penalty_reasons: List[str]
    bonus_reasons: List[str]
    done: bool
    success: bool
