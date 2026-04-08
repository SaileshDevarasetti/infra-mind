from __future__ import annotations

from tasks.base_task import BaseTask, TaskMetadata


class MediumTask(BaseTask):
    metadata = TaskMetadata(
        task_id="medium",
        task_name="Multi-Hop Cascade Diagnosis",
        task_description=(
            "A failure begins in cluster A, spreads into cluster B through an async queue, "
            "while cluster C contains misleading degraded canary nodes."
        ),
        action_budget=30,
        max_steps=30,
    )

    def evaluate_success(self, diagnosis, final_blast_radius, health_ratio) -> bool:
        true_roots = set(self.scenario["failure_injection"]["root_nodes"])
        guessed_roots = set(diagnosis.get("root_causes", []))
        guessed_blast = set(diagnosis.get("blast_radius", []))
        true_blast = set(self.scenario["ground_truth_blast_radius"])
        queue_isolation = any(
            "rabbitmq" in source and "event-consumer-b" in target
            for source, target in diagnosis.get("isolated_edges", [])
        )
        bounded = abs(len(guessed_blast) - len(true_blast)) <= 2
        return true_roots.issubset(guessed_roots) and bounded and queue_isolation and health_ratio >= 0.65
