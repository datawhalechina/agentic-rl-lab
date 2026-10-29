"""将同一个 ALFWorld 游戏实例化为一组独立的 TextWorld 环境。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import textworld
import textworld.envs.pddl.textgen as textgen
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from data import GameExample
from protocol import canonical_action


_TEXTWORLD_SYNC_LOCK = RLock()


@dataclass(frozen=True)
class EnvironmentState:
    """reset 后一条分支可见的初始状态。"""

    observation: str
    task: str
    admissible_actions: tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentStep:
    """执行一次文本动作后的环境结果。"""

    observation: str
    score: float
    done: bool
    won: bool
    admissible: bool
    admissible_actions: tuple[str, ...]


def extract_task(observation: str) -> str:
    """从 ALFWorld 初始 observation 中提取目标描述。"""
    marker = "Your task is to: "
    return observation.split(marker, 1)[1].strip()


def _column(infos: dict[str, Any], key: str, size: int, default: Any) -> list[Any]:
    value = infos.get(key)
    if value is None:
        return [default for _ in range(size)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    return values


def _textworld_eval_scope_compatibility(environment: Any) -> Any:
    """修复 CPython 3.13+ 中 TextWorld 动态变量不可见的问题。

    TextWorld 1.7.0 通过 ``locals().update(...)`` 把 ``r``、``o`` 等游戏
    变量注入 ``eval``。CPython 3.13 起函数局部变量采用已定义的新语义，
    这种写法不会再改变后续 ``eval`` 可见的作用域。该 wrapper 会在同步
    环境以及每个异步子进程内部执行，因此无需修改虚拟环境中的第三方包。
    """
    current_derive = textgen.EvalSymbol.derive
    if getattr(current_derive, "_alfworld_scope_compatible", False):
        return environment

    def derive(symbol: Any, context: dict[str, Any] | None = None) -> list[Any]:
        context = context or symbol.context
        namespace = vars(textgen).copy()
        namespace.update(context["variables"])
        value = eval(symbol.expression, namespace, namespace)
        return [textgen.TerminalSymbol(value)]

    derive._alfworld_scope_compatible = True  # type: ignore[attr-defined]
    textgen.EvalSymbol.derive = derive
    return environment


def _create_textworld_batch(
    game_file: Path,
    *,
    batch_size: int,
    max_turns: int,
    asynchronous: bool,
) -> Any:
    """使用 ALFWorld 官方 demangler/infos wrapper 注册固定游戏。"""
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
    )
    wrappers = [
        _textworld_eval_scope_compatibility,
        AlfredDemangler(shuffle=False),
        AlfredInfos,
    ]
    env_id = textworld.gym.register_games(
        [str(game_file)],
        request_infos,
        batch_size=batch_size,
        asynchronous=asynchronous,
        auto_reset=False,
        max_episode_steps=max_turns,
        wrappers=wrappers,
        name="agentopsd-alfworld",
    )
    return textworld.gym.make(env_id)


def _close_async_workers(environment: Any) -> bool:
    """让 TextWorld 1.7.0 的异步 worker 正常退出，而不是在析构时强杀。"""
    batch = getattr(environment, "batch_env", None)
    children = getattr(batch, "envs", None)
    if not children or not all(
        hasattr(child, "_pipe") and hasattr(child, "_process")
        for child in children
    ):
        return False

    child_type = type(children[0])
    current_destructor = child_type.__del__
    if not getattr(current_destructor, "_alfworld_close_compatible", False):
        original_destructor = current_destructor

        def safe_destructor(child: Any) -> None:
            if getattr(child, "_alfworld_closed", False):
                return
            original_destructor(child)

        safe_destructor._alfworld_close_compatible = True  # type: ignore[attr-defined]
        child_type.__del__ = safe_destructor

    for child in children:
        try:
            child._pipe.send(("close", "", ()))
        except (BrokenPipeError, EOFError, OSError):
            pass

    for child in children:
        child._process.join(timeout=2.0)
        if child._process.is_alive():
            child._process.terminate()
            child._process.join()
        child._alfworld_closed = True
        try:
            child._pipe.close()
        except OSError:
            pass

    # 避免 TextWorld 的 __del__ 再次向已经关闭的 pipe 发送 close。
    batch.envs.clear()
    environment.batch_env = None
    return True


class ALFWorldGroup:
    """持有同一游戏的 K 个独立状态，供组内 rollout 使用。"""

    def __init__(
        self,
        example: GameExample,
        group_size: int,
        max_turns: int,
        *,
        seed: int = 0,
        asynchronous: bool = True,
    ) -> None:
        self.example = example
        self.group_size = group_size
        # TextWorld 即使 asynchronous=True，也会让 batch_size=1 使用进程内
        # SyncBatchEnv；其中多个 TatSu parser 是模块级单例，不能跨线程调用。
        self._uses_in_process_textworld = group_size <= 1 or not asynchronous
        self._env = _create_textworld_batch(
            example.game_file,
            batch_size=group_size,
            max_turns=max_turns,
            asynchronous=asynchronous,
        )
        self._env.seed(seed)
        self._admissible_actions: list[tuple[str, ...]] = [tuple()] * group_size
        self._closed = False

    def reset(self) -> list[EnvironmentState]:
        """重置整组环境并返回每条分支的初始状态。"""
        if self._uses_in_process_textworld:
            with _TEXTWORLD_SYNC_LOCK:
                observations, infos = self._env.reset()
        else:
            observations, infos = self._env.reset()
        observations = [str(value) for value in observations]
        commands = _column(infos, "admissible_commands", self.group_size, [])
        self._admissible_actions = [
            tuple(canonical_action(str(action)) for action in actions)
            for actions in commands
        ]

        task = extract_task(observations[0])
        return [
            EnvironmentState(
                observation=observation,
                task=task,
                admissible_actions=self._admissible_actions[index],
            )
            for index, observation in enumerate(observations)
        ]

    def step(self, actions: list[str]) -> list[EnvironmentStep]:
        """每个分支执行一个动作；score 仅记录，不参与自定义奖励。"""
        normalized = [canonical_action(action) for action in actions]
        admissible = [
            action in current
            for action, current in zip(normalized, self._admissible_actions)
        ]
        if self._uses_in_process_textworld:
            with _TEXTWORLD_SYNC_LOCK:
                observations, scores, dones, infos = self._env.step(normalized)
        else:
            observations, scores, dones, infos = self._env.step(normalized)
        observations = [str(value) for value in observations]
        won = [bool(value) for value in _column(infos, "won", self.group_size, False)]
        commands = _column(infos, "admissible_commands", self.group_size, [])
        self._admissible_actions = [
            tuple(canonical_action(str(action)) for action in values)
            for values in commands
        ]

        return [
            EnvironmentStep(
                observation=observations[index],
                score=float(scores[index]),
                done=bool(dones[index]),
                won=won[index],
                admissible=admissible[index],
                admissible_actions=self._admissible_actions[index],
            )
            for index in range(self.group_size)
        ]

    def close(self) -> None:
        """释放 TextWorld 子进程。"""
        if not self._closed:
            try:
                if not _close_async_workers(self._env):
                    self._env.close()
            finally:
                self._closed = True

    def __enter__(self) -> "ALFWorldGroup":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
