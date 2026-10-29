"""TEMPO macro-step 起点状态的表示、保存与仓库管理。

一个 MacroState 是"环境 + 交互历史"的完整快照：
- 环境侧通过重放 action_history 恢复（TextWorld 确定性）；
- token 侧通过保存的 token_prefix 续跑，保证下一轮 prompt
  仍然是历史真实 token 序列的严格前缀扩展。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from data import GameExample
from environment import EnvironmentState

if TYPE_CHECKING:
    from rollout import RolloutConfig, Trajectory


@dataclass
class MacroState:
    """一个可续跑的 macro-step 起点。"""

    example: GameExample
    task: str
    initial_observation: str
    # 环境侧快照：按顺序重放即可恢复到 current_observation。
    action_history: tuple[str, ...]
    # actor 侧快照：完整 chat 历史；初始状态时为 initial_messages。
    messages: list[dict[str, Any]] = field(default_factory=list)
    # 续跑用的真实 token 前缀；None 表示尚未生成（初始状态，rollout 现算）。
    token_prefix: list[int] | None = None
    # 该状态之前已消耗的交互轮数（用于 T 上限与 macro_index 推进）。
    round_index: int = 0
    macro_index: int = 0
    # 重放时必须使用生成该前缀时的同一环境种子。
    environment_seed: int = 0
    current_observation: str = ""
    current_admissible_actions: tuple[str, ...] = ()

    def fork_messages(self) -> list[dict[str, Any]]:
        """分支各自追加消息，不能共享同一个 list。"""
        return [dict(message) for message in self.messages]


def initial_macro_state(
    example: GameExample,
    state: EnvironmentState,
    *,
    messages: list[dict[str, Any]],
    environment_seed: int,
) -> MacroState:
    """从一次环境 reset 的结果构造第 0 个 macro-step 起点。"""
    return MacroState(
        example=example,
        task=state.task,
        initial_observation=state.observation,
        action_history=(),
        messages=messages,
        token_prefix=None,
        round_index=0,
        macro_index=0,
        environment_seed=environment_seed,
        current_observation=state.observation,
        current_admissible_actions=state.admissible_actions,
    )


class StateStore:
    """endpoint 状态的定长仓库，跨训练轮次均匀采样消费。

    每轮 TD 更新存入 batch 内所有非终局分支的终点，并按容量
    淘汰最旧的状态；这近似博客中"部分段末状态被保存，供后续
    rollout 从这里继续执行"的设定。
    """

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("capacity 必须大于等于 1")
        self._states: deque[MacroState] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def add(self, states: list[MacroState]) -> None:
        self._states.extend(states)

    def sample(self, count: int) -> list[MacroState]:
        """均匀抽出 count 个互不重复的状态；不足时有多少返回多少。"""
        count = min(count, len(self._states))
        indices = self._rng.sample(range(len(self._states)), count)
        return [self._states[index] for index in indices]

    def __len__(self) -> int:
        return len(self._states)


def endpoint_state(
    start_state: MacroState,
    trajectory: "Trajectory",
    *,
    config: "RolloutConfig",
) -> MacroState | None:
    """把一条停在 macro_boundary 的分支转成下一轮起点；其余情况返回 None。

    终局（won / 环境结束 / 轮数用尽 / token 超限）分支没有未来，
    不入库；macro_boundary 但 token 前缀已无续跑空间的分支同样丢弃。
    """
    if trajectory.stop_reason != "macro_boundary":
        return None
    if trajectory.next_prompt_tokens is None:
        return None
    round_index = start_state.round_index + len(trajectory.steps)
    if round_index >= config.max_episode_steps:
        return None
    if len(trajectory.next_prompt_tokens) >= config.max_trajectory_tokens:
        return None
    return MacroState(
        example=start_state.example,
        task=start_state.task,
        initial_observation=start_state.initial_observation,
        action_history=start_state.action_history
        + tuple(step.action for step in trajectory.steps),
        messages=[dict(message) for message in trajectory.messages],
        token_prefix=list(trajectory.next_prompt_tokens),
        round_index=round_index,
        macro_index=start_state.macro_index + 1,
        environment_seed=start_state.environment_seed,
        current_observation=trajectory.current_observation,
        current_admissible_actions=trajectory.current_admissible_actions,
    )


def boundary_states(
    start_state: MacroState,
    trajectory: "Trajectory",
    *,
    macro_rounds: int,
    max_trajectory_tokens: int,
) -> list[MacroState]:
    """提取轨迹在每个 H 轮边界上的非终局快照（warm-up 的 critic 训练点）。

    warm-up 阶段 rollout 会跑到终局；这里把 H、2H、... 轮处的中间状态
    重建出来，配合轨迹最终结局（MC return）构造无需 bootstrap 的 target。
    """
    from protocol import tool_message

    snapshots: list[MacroState] = []
    total_steps = len(trajectory.steps)
    for rounds in range(macro_rounds, total_steps + 1, macro_rounds):
        # 轨迹恰好在该边界终止则没有"从该状态继续"的语义。
        if total_steps == rounds and trajectory.stop_reason != "macro_boundary":
            break

        messages = start_state.fork_messages()
        for step in trajectory.steps[:rounds]:
            messages.append({"role": "assistant", "content": step.assistant_text})
            messages.append(tool_message(step.call_id, step.tool_response))

        if total_steps > rounds:
            token_prefix = list(trajectory.steps[rounds].prompt_tokens)
            admissible = trajectory.steps[rounds].admissible_actions
        else:
            if trajectory.next_prompt_tokens is None:
                break
            token_prefix = list(trajectory.next_prompt_tokens)
            admissible = trajectory.current_admissible_actions
        if len(token_prefix) >= max_trajectory_tokens:
            break

        snapshots.append(
            MacroState(
                example=start_state.example,
                task=start_state.task,
                initial_observation=start_state.initial_observation,
                action_history=start_state.action_history
                + tuple(step.action for step in trajectory.steps[:rounds]),
                messages=messages,
                token_prefix=token_prefix,
                round_index=start_state.round_index + rounds,
                macro_index=start_state.macro_index + rounds // macro_rounds,
                environment_seed=start_state.environment_seed,
                current_observation=trajectory.steps[rounds - 1].next_observation,
                current_admissible_actions=admissible,
            )
        )
    return snapshots
