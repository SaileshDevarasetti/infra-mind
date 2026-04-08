from __future__ import annotations

from tasks.base_task import BaseTask, TaskMetadata


class EasyTask(BaseTask):
    metadata = TaskMetadata(
        task_id="easy",
        task_name="Single Service Failure Detection",
        task_description=(
            "A single database OOM crash fans out through tightly coupled services. "
            "The agent must identify the root node and heal most of the impacted path."
        ),
        action_budget=20,
        max_steps=20,
    )

    def evaluate_success(self, diagnosis, final_blast_radius, health_ratio) -> bool:
        roots = set(diagnosis.get("root_causes", []))
        true_roots = set(self.scenario["failure_injection"]["root_nodes"])
        healed_count = sum(1 for node_id in self.scenario["ground_truth_blast_radius"] if node_id not in final_blast_radius)
        return true_roots.issubset(roots) and healed_count >= 3 and health_ratio >= 0.7
