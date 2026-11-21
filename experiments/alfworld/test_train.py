"""Unit tests for building the PPO Datum of one ALFWorld trajectory.

Run from the repo root (needs `uv sync --extra alfworld`):
    uv run python -m unittest discover -s experiments/alfworld
"""

import unittest
from pathlib import Path

import pytrio as trio

from data import GameExample
from rollout import StepRecord, Trajectory
from train import build_trajectory_datum


def trajectory(advantage: float) -> Trajectory:
    example = GameExample("train:demo", "train", Path("demo.tw-pddl"), "pick_and_place_simple")
    step = StepRecord(
        index=1, observation="o", admissible_actions=("look",),
        prompt_tokens=[1, 2, 3], completion_tokens=[4, 5], logprobs=[-0.1, -0.2],
        assistant_text="t", call_id="c", action="look", valid_format=True,
        admissible=True, valid_action=True, tool_response="r",
        next_observation="o2", score=0.0, done=True, won=True,
    )
    result = Trajectory(
        example=example, group_id="0:demo", game_index=0, group_index=0,
        task="t", initial_observation="o", current_observation="o2",
        current_admissible_actions=("look",), messages=[], steps=[step],
        done=True, won=True, reward=1.0,
    )
    result.advantage = advantage
    return result


class BuildTrajectoryDatumTest(unittest.TestCase):
    def test_builds_shifted_ppo_datum_with_pinned_pytrio(self):
        item = build_trajectory_datum(trajectory(0.5), max_sequence_tokens=64)

        self.assertIsInstance(item.datum, trio.Datum)
        inputs = item.datum.loss_fn_inputs
        self.assertEqual(item.datum.model_input.to_ints(), [1, 2, 3, 4])
        self.assertEqual(list(inputs["target_tokens"].data), [2, 3, 4, 5])
        self.assertEqual(list(inputs["advantages"].data), [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(len(inputs["logprobs"].data), 4)
        self.assertEqual(item.loss_tokens, 2)


if __name__ == "__main__":
    unittest.main()
