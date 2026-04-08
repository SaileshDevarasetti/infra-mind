from __future__ import annotations

from tasks.base_task import BaseTask, TaskMetadata


class HardTask(BaseTask):
    metadata = TaskMetadata(
        task_id="hard",
        task_name="Multi-Root Self-Healing Under Constraints",
        task_description=(
            "Two simultaneous failures unfold across a multi-region estate with hidden restart traps "
            "and an accelerating cascade after step fifteen."
        ),
        action_budget=25,
        max_steps=25,
    )

    def evaluate_success(self, diagnosis, final_blast_radius, health_ratio) -> bool:
        true_roots = set(self.scenario["failure_injection"]["root_nodes"])
        guessed_roots = set(diagnosis.get("root_causes", []))
        avoided_traps = not any(node_id in diagnosis.get("trap_triggered_nodes", []) for node_id in self.scenario["healing_traps"])
        return true_roots.issubset(guessed_roots) and avoided_traps and health_ratio > 0.70
