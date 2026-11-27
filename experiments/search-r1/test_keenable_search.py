"""Keenable 搜索后端的离线单测（mock httpx，不发真实请求）。

从仓库根目录运行：
uv run python -m unittest discover -s experiments/search-r1 -p "test_*.py"
"""

import argparse
import os
import threading
import time
import unittest
from unittest import mock

import httpx

import search
from search import (
    KEENABLE_PUBLIC_SEARCH_ENDPOINT,
    KEENABLE_SEARCH_ENDPOINT,
    KeenableSearchClient,
    create_search_client,
)


def ok_response(results: list[dict]) -> httpx.Response:
    request = httpx.Request("POST", KEENABLE_SEARCH_ENDPOINT)
    return httpx.Response(200, json={"results": results}, request=request)


def error_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", KEENABLE_SEARCH_ENDPOINT)
    return httpx.Response(status, headers=headers or {}, request=request)


RESULTS = [
    {
        "title": "Beloved (novel)",
        "url": "https://en.wikipedia.org/wiki/Beloved_(novel)",
        "description": "",
        "snippet": "Beloved is a 1987 novel by Toni Morrison. " + "x" * 2000,
    },
    {"title": "Empty", "url": "https://example.com/empty", "snippet": ""},
]


class KeenableClientTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"KEENABLE_API_KEY": ""})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.sleep = mock.patch.object(search.time, "sleep").start()
        self.addCleanup(mock.patch.stopall)

    def client(self, **kwargs) -> KeenableSearchClient:
        kwargs.setdefault("min_request_interval", 0.0)
        return KeenableSearchClient(**kwargs)

    def test_keyless_uses_public_endpoint_without_key_header(self) -> None:
        with mock.patch.object(search.httpx, "post", return_value=ok_response(RESULTS)) as post:
            result = self.client().search("who wrote Beloved")
        self.assertTrue(result.ok)
        url = post.call_args.args[0]
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(url, KEENABLE_PUBLIC_SEARCH_ENDPOINT)
        self.assertEqual(headers, {"X-Keenable-Title": "agentic-rl-lab"})
        self.assertEqual(
            post.call_args.kwargs["json"], {"query": "who wrote Beloved", "max_results": 3}
        )

    def test_api_key_uses_keyed_endpoint(self) -> None:
        with mock.patch.dict(os.environ, {"KEENABLE_API_KEY": " keen_test "}):
            client = self.client()
        with mock.patch.object(search.httpx, "post", return_value=ok_response(RESULTS)) as post:
            client.search("q")
        self.assertFalse(client.keyless)
        self.assertEqual(post.call_args.args[0], KEENABLE_SEARCH_ENDPOINT)
        self.assertEqual(post.call_args.kwargs["headers"]["X-API-Key"], "keen_test")
        self.assertNotIn("keen_test", repr(client))

    def test_items_are_truncated_and_empty_snippets_dropped(self) -> None:
        with mock.patch.object(search.httpx, "post", return_value=ok_response(RESULTS)):
            result = self.client().search("q")
        self.assertEqual(len(result.items), 1)
        item = result.items[0]
        self.assertEqual(len(item.content), search.KEENABLE_MAX_CONTENT_CHARS)
        self.assertEqual(item.source, "en.wikipedia.org")
        self.assertEqual(item.url, RESULTS[0]["url"])

    def test_429_retries_after_short_retry_after(self) -> None:
        responses = [error_response(429, {"Retry-After": "2"}), ok_response(RESULTS)]
        with mock.patch.object(search.httpx, "post", side_effect=responses):
            client = self.client()
            result = client.search("q")
        self.assertTrue(result.ok)
        self.sleep.assert_any_call(2.0)
        self.assertEqual(client.stats.rate_limits, 1)

    def test_long_retry_after_returns_error_without_waiting(self) -> None:
        with mock.patch.object(
            search.httpx, "post", return_value=error_response(429, {"Retry-After": "1800"})
        ) as post:
            result = self.client().search("q")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 429)
        self.assertEqual(post.call_count, 1)
        self.sleep.assert_not_called()

    def test_5xx_backs_off_then_gives_up(self) -> None:
        with mock.patch.object(
            search.httpx, "post", return_value=error_response(503)
        ) as post:
            result = self.client(max_retries=2).search("q")
        self.assertFalse(result.ok)
        self.assertEqual(post.call_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1.0, 2.0])

    def test_timeout_is_reported(self) -> None:
        with mock.patch.object(
            search.httpx, "post", side_effect=httpx.ReadTimeout("slow")
        ):
            client = self.client(max_retries=0)
            result = client.search("q")
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "request timeout")
        self.assertEqual(client.stats.timeouts, 1)

    def test_rate_limit_spaces_concurrent_starts(self) -> None:
        mock.patch.stopall()  # 这里需要真实 sleep
        starts: list[float] = []
        lock = threading.Lock()

        def fake_post(*args, **kwargs):
            with lock:
                starts.append(time.monotonic())
            return ok_response(RESULTS)

        client = KeenableSearchClient(min_request_interval=0.11)
        with mock.patch.dict(os.environ, {"KEENABLE_API_KEY": ""}), mock.patch.object(
            search.httpx, "post", side_effect=fake_post
        ):
            threads = [threading.Thread(target=client.search, args=("q",)) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        starts.sort()
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        self.assertEqual(len(starts), 6)
        self.assertTrue(all(gap >= 0.1 for gap in gaps), gaps)

    def test_factory_builds_keenable_client(self) -> None:
        client = create_search_client("keenable", None, count=5)
        self.assertIsInstance(client, KeenableSearchClient)
        self.assertEqual(client.count, 5)
        self.assertEqual(client.timeout, search.DEFAULT_SEARCH_TIMEOUT["keenable"])


class EvalSummaryTest(unittest.TestCase):
    def test_summary_records_keenable_endpoint_and_count(self) -> None:
        try:
            import eval as eval_module
        except ImportError as error:  # pytrio 等训练依赖未安装时跳过
            self.skipTest(f"eval.py 依赖不可用: {error}")
        args = argparse.Namespace(
            base_model="Qwen/Qwen3.5-4B",
            model_path=None,
            search_backend="keenable",
            search_concurrency=8,
            search_timeout=30.0,
            search_count=3,
            search_api=KEENABLE_PUBLIC_SEARCH_ENDPOINT,
        )
        record = eval_module.summary_record(args, [], {})
        self.assertEqual(record["search_api"], KEENABLE_PUBLIC_SEARCH_ENDPOINT)
        self.assertEqual(record["search_count"], 3)
        self.assertEqual(record["search_text_field"], "snippet")


if __name__ == "__main__":
    unittest.main()
