"""Unit tests for how rollout_batch picks the policy snapshot ID.

Run from the repo root (needs `uv sync --extra alfworld`):
    uv run python -m unittest discover -s experiments/agentopsd
"""

import unittest
from types import SimpleNamespace

from pytrio.client.sampling import SamplingClient

from rollout import RolloutConfig, rollout_batch


def sampling_client(run_id: str) -> SamplingClient:
    # A real SamplingClient from the pinned pytrio; no request is ever sent.
    return SamplingClient(
        sampling_run_cuid=run_id,
        model_id="Qwen/Qwen3.5-4B",
        coordinator=None,
        limits=None,
        runtime=SimpleNamespace(event_loop=None),
    )


class RolloutSnapshotIdTest(unittest.TestCase):
    def test_snapshot_id_comes_from_the_sampling_client(self):
        batch = rollout_batch(sampling_client("run_demo"), None, [], RolloutConfig())
        self.assertEqual(batch.policy_snapshot_id, "run_demo")

    def test_explicit_snapshot_id_wins(self):
        batch = rollout_batch(
            sampling_client("run_demo"), None, [], RolloutConfig(), policy_snapshot_id="step-3"
        )
        self.assertEqual(batch.policy_snapshot_id, "step-3")


if __name__ == "__main__":
    unittest.main()
