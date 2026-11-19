"""Offline mathematical, baseline-parity and rollout tests; never call a service."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CHAPTER = Path(__file__).resolve().parents[1]
BASELINE = CHAPTER.parent / "alfworld"
if not (BASELINE / "train.py").is_file():
    # Keep offline checks usable in a local checkout from before the repository
    # moved its chapters under experiments/. New-layout checkouts use the path above.
    BASELINE = CHAPTER.parents[1] / "08-alfworld"
sys.path.insert(0, str(CHAPTER))

from advantages import assign_advantages
from analysis import load_evaluation, validate_pair, training_budgets, paired_delta
from data import GameExample, shuffled_games, take_batch
from environment import EnvironmentState, EnvironmentStep
from experiment import dataset_fingerprint, sampler_from_manifest
from protocol import build_next_prompt, build_prompt, parse_assistant
from reward import assign_rewards, discounted_returns
from rollout import RolloutConfig, StepRecord, Trajectory, rollout_batch
import analysis
import train
import eval as evaluate


def original_module(name: str):
    spec = importlib.util.spec_from_file_location(f"original_{name}", BASELINE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trajectory(observations, *, won=False, invalid=(), group="game", branch=0):
    example = GameExample(group, "train", Path("/synthetic/game.tw-pddl"), "pick_and_place_simple")
    result = Trajectory(example, group, 0, branch, "put apple in fridge", observations[0], observations[-1], (), [])
    prefix = [100, 101]
    for i, observation in enumerate(observations):
        completion = [200 + 2 * i, 201 + 2 * i]
        valid = i not in invalid
        result.steps.append(StepRecord(
            index=i + 1, observation=observation, admissible_actions=("look",),
            prompt_tokens=list(prefix), completion_tokens=completion, logprobs=[-0.2, -0.4],
            assistant_text="synthetic", call_id=str(i), action="look", valid_format=valid,
            admissible=valid, valid_action=valid, tool_response="result", next_observation="after",
            score=0, done=i == len(observations) - 1, won=won and i == len(observations) - 1,
        ))
        prefix = prefix + completion + [300 + i]
    result.next_prompt_tokens = prefix
    result.won = won
    result.done = True
    assign_rewards(result)
    return result


class FakeTokenizer:
    eos_token = None

    def encode(self, text, **kwargs):
        return [ord(char) + 1 for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token - 1) for token in tokens)

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text)


class FakeSampler:
    def __init__(self):
        self.calls = 0
        self.tokenizer = FakeTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    async def sample_async(self, *, num_samples, **kwargs):
        sequences = []
        for i in range(num_samples):
            text = "bad format" if self.calls == 0 and i == 0 else '<tool_call>{"name":"alfworld_step","arguments":{"action":"open fridge 1"}}</tool_call>'
            tokens = self.tokenizer.encode(text)
            sequences.append(SimpleNamespace(tokens=tokens, logprobs=[-0.3] * len(tokens), text=text))
        self.calls += 1
        return SimpleNamespace(sequences=sequences)


class FakeEnvironment:
    def __init__(self, example, group_size, max_steps, **kwargs):
        self.size = group_size
        self.won = [False] * group_size
        self.closed = False

    def reset(self):
        return [EnvironmentState("fridge is closed", "open the fridge", ("open fridge 1",)) for _ in range(self.size)]

    def step(self, actions):
        results = []
        for i, action in enumerate(actions):
            valid = action == "open fridge 1"
            self.won[i] = self.won[i] or valid
            results.append(EnvironmentStep("fridge is open" if self.won[i] else "fridge is closed", 0, self.won[i], self.won[i], valid, ("open fridge 1",)))
        return results

    def close(self):
        self.closed = True


def synthetic_dataset(root: Path):
    for split in ("train", "valid_seen", "valid_unseen"):
        folder = root / "json_2.1.1" / split / "synthetic"
        folder.mkdir(parents=True)
        (folder / "game.tw-pddl").write_text('{"solvable":true}')
        (folder / "traj_data.json").write_text('{"task_type":"pick_and_place_simple"}')


class AlgorithmTests(unittest.TestCase):
    def test_reward_conservation_and_discount(self):
        tr = trajectory(["s0", "s1", "s2"], won=True, invalid=(0, 2))
        self.assertAlmostEqual(tr.reward, 0.8)
        self.assertAlmostEqual(sum(step.reward for step in tr.steps), tr.reward)
        self.assertEqual(discounted_returns([0, 0, 1], 0.5), [0.25, 0.5, 1])
        self.assertAlmostEqual(discounted_returns([-0.1, 0, 0.9], 0.5)[0], 0.125)
        failed = trajectory(["s"], invalid=(0,))
        self.assertAlmostEqual(failed.reward, -0.1)

    def test_macro_matches_original_with_unequal_lengths(self):
        batch = [trajectory(["a"], won=True), trajectory(["a", "b", "c"], invalid=(1,)), trajectory(["a", "x"], won=True)]
        original = copy.deepcopy(batch)
        original_module("advantages").assign_advantages(original)
        assign_advantages(batch, algorithm="grpo")
        self.assertEqual([tr.advantage for tr in batch], [tr.advantage for tr in original])
        for tr in batch:
            self.assertTrue(all(step.advantage == tr.advantage for step in tr.steps))

    def test_hand_computed_micro_returns(self):
        a = trajectory(["root", "a"], won=True)
        b = trajectory(["root", "b", "c"], won=True, branch=1)
        stats = assign_advantages([a, b], algorithm="gigpo", gamma=0.5)
        self.assertEqual(a.advantage, 0)
        self.assertAlmostEqual(a.steps[0].micro_advantage, 0.125)
        self.assertAlmostEqual(b.steps[0].micro_advantage, -0.125)
        self.assertEqual(a.steps[1].micro_advantage, 0)
        self.assertAlmostEqual(stats.nonzero_micro_fraction, 2 / 5)
        self.assertEqual(stats.noninitial_matched_fraction, 0)

    def test_no_cross_game_or_post_action_grouping(self):
        a = trajectory(["same"], won=True, group="game-a")
        b = trajectory(["same"], group="game-b")
        assign_advantages([a, b], algorithm="gigpo")
        self.assertEqual(a.steps[0].micro_advantage, 0)
        b.group_id = a.group_id
        b.steps[0].action = "different action"
        b.steps[0].next_observation = "different future"
        assign_advantages([a, b], algorithm="gigpo")
        self.assertAlmostEqual(a.steps[0].micro_advantage, 0.5)

    def test_repeated_visits_and_stricter_anchor(self):
        a = trajectory(["same", "same"], won=True)
        stats = assign_advantages([a], algorithm="gigpo", gamma=0.5)
        self.assertEqual(stats.matched_step_fraction, 1)
        self.assertEqual(stats.cross_trajectory_step_fraction, 0)
        self.assertAlmostEqual(a.steps[0].micro_advantage, -0.25)
        a.steps[1].admissible_actions = ("open fridge 1",)
        assign_advantages([a], algorithm="gigpo", anchor_mode="observation_actions")
        self.assertEqual(a.steps[0].micro_advantage, 0)

    def test_step_weight_zero_matches_original_datums(self):
        batch = [trajectory(["root", "a"], won=True), trajectory(["root", "b", "c"], invalid=(0,))]
        assign_advantages(batch, algorithm="gigpo", step_weight=0)
        baseline = original_module("train")
        for tr in batch:
            # Supply the missing SDK validation method only while running the
            # original builder. Its token/advantage construction is unchanged.
            with patch.object(train.trio.Datum, "validate_loss_inputs", lambda datum, loss: train.validate_ppo_datum(datum), create=True):
                old = baseline.build_trajectory_datum(tr, max_sequence_tokens=12000)
            new = train.build_trajectory_datum(tr, max_sequence_tokens=12000)
            self.assertEqual(old.datum.model_dump(), new.datum.model_dump())

    def test_nonfinite_training_signals_are_rejected(self):
        tr = trajectory(["root"], won=True)
        tr.steps[0].advantage = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            train.build_trajectory_datum(tr, max_sequence_tokens=12000)
        tr.steps[0].advantage = 1.0
        tr.steps[0].logprobs[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            train.build_trajectory_datum(tr, max_sequence_tokens=12000)

    def test_zero_macro_micro_signal_is_not_dropped(self):
        batch = [trajectory(["root", "a"], won=True), trajectory(["root", "b", "c"], won=True)]
        assign_advantages(batch, algorithm="gigpo", gamma=0.5)
        self.assertEqual([tr.advantage for tr in batch], [0, 0])
        self.assertEqual(len(train.build_training_datums(batch, max_sequence_tokens=12000)), 2)

    def test_token_mask_and_old_logprob_alignment(self):
        tr = trajectory(["root", "next"], won=True)
        tr.steps[0].advantage, tr.steps[1].advantage = 0.25, -0.5
        datum = train.build_trajectory_datum(tr, max_sequence_tokens=12000).datum
        self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [101, 200, 201, 300, 202, 203, 301])
        self.assertEqual(datum.loss_fn_inputs["advantages"].data, [0, 0.25, 0.25, 0, -0.5, -0.5, 0])
        self.assertEqual(len(datum.loss_fn_inputs["logprobs"].data), 7)
        tr.steps[1].prompt_tokens[0] = 999
        with self.assertRaisesRegex(ValueError, "前缀"):
            train.build_trajectory_datum(tr, max_sequence_tokens=12000)

    def test_prefix_closing_overlap_and_parser(self):
        tokenizer = FakeTokenizer()
        messages = [{"role": "user", "content": "task"}]
        prompt = build_prompt(tokenizer, messages)
        completion = tokenizer.encode("output</assistant>")
        nxt = build_next_prompt(tokenizer, messages, prompt, completion, {"role": "tool", "content": "feedback"})
        self.assertEqual(nxt[:len(prompt) + len(completion)], prompt + completion)
        self.assertEqual(tokenizer.decode(nxt).count("</assistant>"), 1)
        self.assertFalse(parse_assistant("open fridge 1").valid_format)

    def test_original_protocol_and_environment_are_unchanged(self):
        for filename in ("protocol.py", "environment.py"):
            self.assertEqual((CHAPTER / filename).read_bytes(), (BASELINE / filename).read_bytes())

    def test_rollout_invalid_penalty_and_recovery(self):
        example = GameExample("train:fake", "train", Path("/fake"), "pick_and_place_simple")
        sampler = FakeSampler()
        batch = rollout_batch(sampler, sampler.tokenizer, [example], RolloutConfig(group_size=2, max_steps=3, max_assistant_tokens=2048, algorithm="gigpo"), environment_factory=FakeEnvironment)
        a, b = batch.trajectories
        self.assertTrue(a.won and b.won)
        self.assertEqual([a.invalid_action_count, b.invalid_action_count], [1, 0])
        self.assertEqual([len(a.steps), len(b.steps)], [2, 1])
        self.assertAlmostEqual(a.reward, 0.9)
        self.assertEqual(a.steps[1].prompt_tokens[:len(a.steps[0].prompt_tokens)], a.steps[0].prompt_tokens)
        self.assertEqual(len(train.build_training_datums(batch.trajectories, max_sequence_tokens=12000)), 2)

    def test_data_order_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            synthetic_dataset(root)
            games = shuffled_games(root, "train", 42)
            self.assertEqual([g.id for g in games], [g.id for g in original_module("data").shuffled_games(root, "train", 42)])
            self.assertEqual(take_batch(list("ABCDE"), 4, 2), ["E", "A"])
            before = dataset_fingerprint(games)
            games[0].game_file.write_text('{"solvable":true,"changed":1}')
            self.assertNotEqual(before, dataset_fingerprint(games))

    def test_offline_train_checkpoint_eval_chain(self):
        class Future:
            def __init__(self, value): self.value = value
            def result(self): return self.value
        class Client:
            def get_tokenizer(self): return FakeTokenizer()
            def save_weights_and_get_sampling_client(self): return FakeSampler()
            def forward_backward(self, data, **kwargs):
                self.seen = len(data)
                return Future(SimpleNamespace(metrics={"loss_mean": 0.1}))
            def optim_step(self, params): return Future(None)
            def save_state(self, name): return Future(SimpleNamespace(path=f"trio://fake/training_state/{name}"))
            def save_weights_for_sampler(self, name): return Future(SimpleNamespace(path=f"trio://fake/sampler_weights/{name}"))
        client = Client()
        service = SimpleNamespace(create_lora_training_client=lambda **kw: client, create_sampling_client=lambda **kw: FakeSampler())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            synthetic_dataset(root / "data")
            args = train.parse_args(["--algorithm", "gigpo", "--data-root", str(root / "data"), "--output-dir", str(root / "run"), "--max-steps", "1", "--games-per-batch", "1", "--group-size", "2", "--save-every", "0", "--swanlab-mode", "disabled"])
            with patch("train.trio.ServiceClient", return_value=service), patch("rollout.ALFWorldGroup", FakeEnvironment), patch("train.swanlab.init"), patch("train.swanlab.log"), patch("train.swanlab.finish"):
                train.main(args)
                run = json.loads((root / "run/run.json").read_text())
                self.assertEqual(run["status"], "completed")
                self.assertEqual(run["optimizer_updates"], 1)
                self.assertEqual(client.seen, 2)
                self.assertIn("sampler_weights", sampler_from_manifest(root / "run/run.json", 1))
                output = root / "eval.jsonl"
                eval_args = evaluate.parse_args(["--data-root", str(root / "data"), "--output", str(output), "--checkpoint-manifest", str(root / "run/run.json"), "--checkpoint-step", "1"])
                evaluate.main(eval_args)
                result = load_evaluation(output)
                self.assertEqual(len(result[0]), 2)
                self.assertTrue(validate_pair(result, result))
                budgets = training_budgets({"GiGPO": root / "run/run.json"}, {"GiGPO": result})
                self.assertEqual(budgets["GiGPO"]["iterations"], 1)
                self.assertEqual(budgets["GiGPO"]["optimizer_updates"], 1)
                wrong_weights = copy.deepcopy(result)
                wrong_weights[1]["model_path"] = "trio://unrelated/checkpoint"
                with self.assertRaisesRegex(ValueError, "weights"):
                    training_budgets({"GiGPO": root / "run/run.json"}, {"GiGPO": wrong_weights})
                other = copy.deepcopy(result)
                other[1]["evaluation"]["config"]["temperature"] = 1.0
                with self.assertRaisesRegex(ValueError, "temperature"):
                    validate_pair(result, other)
                incomplete = root / "partial.jsonl"
                incomplete.write_text(output.read_text().splitlines()[0] + "\n")
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    load_evaluation(incomplete)
                with patch.object(sys, "argv", ["analysis.py", "--inputs", f"GRPO={output}", f"GiGPO={output}", "--runs", f"GiGPO={root / 'run/run.json'}", "--bootstrap-samples", "100", "--output-dir", str(root / "analysis"), "--figure", str(root / "synthetic.png")]):
                    analysis.main()
                report = json.loads((root / "analysis/comparison.json").read_text())
                self.assertTrue(report["metadata_verified"])
                self.assertEqual(report["paired_comparisons"]["GiGPO"]["all"]["delta_percentage_points"], 0)
                self.assertTrue((root / "analysis/comparison.csv").is_file())
                self.assertTrue((root / "synthetic.png").read_bytes().startswith(b"\x89PNG"))

                # An earlier evaluated checkpoint must count only its own cost,
                # and it must not be compared against a later checkpoint.
                metric = json.loads((root / "run/metrics.jsonl").read_text())
                second = {**metric, "update": 2, "train/optimizer_updates": 2}
                (root / "run/metrics.jsonl").write_text(json.dumps(metric) + "\n" + json.dumps(second) + "\n")
                run["config"]["max_steps"] = run["completed_updates"] = run["optimizer_updates"] = 2
                run["checkpoints"].append({"update": 2, "sampler_weights": "trio://fake/later"})
                (root / "run/run.json").write_text(json.dumps(run))
                budgets = training_budgets({"early": root / "run/run.json"}, {"early": result})
                self.assertEqual(budgets["early"]["iterations"], 1)
                self.assertEqual(budgets["early"]["sampled_tokens"], metric["budget/sampled_tokens"])
                later = copy.deepcopy(result)
                later[1]["model_path"] = "trio://fake/later"
                later[1]["evaluation"]["config"]["checkpoint_step"] = 2
                with self.assertRaisesRegex(ValueError, "different iteration budgets"):
                    training_budgets({"early": root / "run/run.json", "later": root / "run/run.json"}, {"early": result, "later": later})

    def test_paired_success_delta_counts(self):
        base = {str(i): {"won": value} for i, value in enumerate([True, False, False, True])}
        treatment = {str(i): {"won": value} for i, value in enumerate([True, True, True, False])}
        result = paired_delta(base, treatment, list(base), 100)
        self.assertEqual(result["delta_percentage_points"], 25)
        self.assertEqual(result["new_successes"], 2)
        self.assertEqual(result["lost_successes"], 1)


if __name__ == "__main__":
    unittest.main()
