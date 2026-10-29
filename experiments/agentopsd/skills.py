"""加载只提供给 Teacher 的固定 ALFWorld SkillBank。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "skills" / "alfworld"
SKILL_SOURCE_REPOSITORY = "https://github.com/ZethWang/AgentOPSD"
SKILL_SOURCE_COMMIT = "0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b"

# 本地 ALFWorld 数据集将单物体任务命名为 ``pick_and_place_simple``，
# 上游 SkillBank 使用的名称则是 ``pick_and_place``。
TASK_TYPE_ALIASES = {
    "pick_and_place_simple": "pick_and_place",
}


@dataclass(frozen=True)
class SkillSelection:
    """Resolved audit key and the exact text injected into the Teacher."""

    task_type: str
    skill_name: str
    key: str
    text: str


class SkillProvider:
    """Deterministically map ALFWorld task metadata to pinned skill text."""

    def __init__(self, skills_dir: str | Path = DEFAULT_SKILLS_DIR) -> None:
        self.skills_dir = Path(skills_dir).expanduser().resolve()
        mapping_path = self.skills_dir / "skill_mapping.json"
        with mapping_path.open(encoding="utf-8") as file:
            mapping = json.load(file)

        skill_files = mapping["skill_files"]
        task_to_skill = mapping["task_to_skill"]
        self.task_to_skill = {
            str(task_type): str(skill_name)
            for task_type, skill_name in task_to_skill.items()
        }
        self.skill_contents: dict[str, str] = {}
        for skill_name, filename in skill_files.items():
            path = self.skills_dir / str(filename)
            content = path.read_text(encoding="utf-8").strip()
            self.skill_contents[str(skill_name)] = content

    @staticmethod
    def canonical_task_type(task_type: str) -> str:
        value = str(task_type).strip()
        return TASK_TYPE_ALIASES.get(value, value)

    def resolve(self, task_type: str) -> SkillSelection:
        """Return ``general + task-specific`` text for an explicit task type."""

        canonical = self.canonical_task_type(task_type)
        skill_name = self.task_to_skill[canonical]
        task_skill = self.skill_contents[skill_name]
        general = self.skill_contents["general_skills"]
        return SkillSelection(
            task_type=str(task_type),
            skill_name=skill_name,
            key=f"general_skills+{skill_name}",
            text=f"{general}\n\n{task_skill}",
        )
