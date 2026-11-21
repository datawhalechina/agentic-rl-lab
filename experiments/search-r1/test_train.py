"""Unit tests for resuming train.py from a saved state.

PyTRIO, search, rollout and SwanLab are stubbed; train.main runs unmodified.

Run from the repo root:
    uv run python -m unittest discover -s experiments/search-r1
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import train
from rollout import AssistantTurn, Trajectory


class Done:
    def __init__(self, value=None):
        self.value = value

    def result(self):
        return self.value


class FakeTrainingClient:
    def __init__(self, log):
        self.log = log

    def get_tokenizer(self):
        return SimpleNamespace(eos_token="<|im_end|>")

    def save_weights_and_get_sampling_client(self):
        return object()

    def forward_backward(self, datums, loss_fn):
        return Done(SimpleNamespace(metrics={"loss_mean": 0.0}))

    def optim_step(self, params):
        return Done()

    def save_state(self, name):
        self.log.append(("save_state", name))
        return Done(SimpleNamespace(path=f"trio://run/training_state/{name}"))

    def save_weights_for_sampler(self, name):
        return Done(SimpleNamespace(path=f"trio://run/sampler_weights/{name}"))


class TrainResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "train.jsonl"
        self.data.write_text(
            "".join(
                json.dumps({"id": f"q{i}", "question": f"q {i}?", "answers": ["a"], "data_source": "nq"}) + "\n"
                for i in range(40)
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def parse(self, *extra):
        argv = [
            "train.py", "--max-steps", "4", "--questions-per-batch", "2",
            "--group-size", "2", "--save-every", "2", "--search-backend", "wikipedia",
            "--swanlab-mode", "disabled", "--run-name", "sr1", "--data", str(self.data),
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            return train.parse_args()

    def run_training(self, *extra):
        log = []

        def fake_rollout_batch(*, examples, **kwargs):
            log.append(("batch", [example.id for example in examples]))
            trajectories = []
            for question_index, example in enumerate(examples):
                for group_index, reward in enumerate([1.0, -0.1]):
                    trajectory = Trajectory(
                        example=example,
                        group_index=group_index,
                        messages=[],
                        question_index=question_index,
                    )
                    trajectory.turns.append(AssistantTurn([1, 2, 3], [4, 5], [-0.1, -0.1], "Answer: a"))
                    trajectory.reward, trajectory.advantage, trajectory.done = reward, reward - 0.45, True
                    trajectories.append(trajectory)
            return trajectories

        service = SimpleNamespace(
            create_lora_training_client=lambda **kwargs: log.append(("new",)) or FakeTrainingClient(log),
            create_training_client_from_state=lambda path: log.append(("weights_only", path))
            or FakeTrainingClient(log),
            create_training_client_from_state_with_optimizer=lambda path: log.append(("with_optimizer", path))
            or FakeTrainingClient(log),
        )
        swanlab = SimpleNamespace(
            init=lambda **kwargs: SimpleNamespace(finish=lambda: None),
            log=lambda metrics, step: log.append(("swanlab_step", step)),
        )
        search = SimpleNamespace(stats=SimpleNamespace(metrics=dict))
        args = self.parse(*extra)
        with (
            mock.patch.object(train.trio, "ServiceClient", lambda: service),
            mock.patch.object(train, "rollout_batch", fake_rollout_batch),
            mock.patch.object(train, "create_search_client", lambda *a, **k: search),
            mock.patch.object(train, "swanlab", swanlab),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            train.main(args)
        return log

    @staticmethod
    def events(log, kind):
        return [event[1] for event in log if event[0] == kind]

    def test_resume_restores_optimizer_and_continues_from_start_step(self):
        fresh = self.run_training()
        resumed = self.run_training(
            "--resume-state", "trio://run/training_state/sr1-step-2-state", "--start-step", "2"
        )

        self.assertIn(("with_optimizer", "trio://run/training_state/sr1-step-2-state"), resumed)
        self.assertEqual(self.events(resumed, "batch"), self.events(fresh, "batch")[2:])
        self.assertEqual(self.events(resumed, "save_state"), ["sr1-step-4-state", "sr1-final-state"])
        self.assertEqual(self.events(resumed, "swanlab_step"), [2, 3])

    def test_start_step_requires_resume_state(self):
        with self.assertRaises(SystemExit):
            self.parse("--start-step", "2")

    def test_start_step_must_be_before_max_steps(self):
        with self.assertRaises(SystemExit):
            self.parse("--resume-state", "trio://x", "--start-step", "4")


if __name__ == "__main__":
    unittest.main()
