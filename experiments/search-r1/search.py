"""提供 DeepSeek 官方 Search API 和 Wikipedia 两个搜索后端。"""

import gzip
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


SEARCH_BACKENDS = ("deepseek", "wikipedia")
DEEPSEEK_SEARCH_ENDPOINT = "https://api.deepseek.com/v1/tools/search"
WIKIPEDIA_SEARCH_ENDPOINT = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = (
    "agentic-rl-lab/0.1 "
    "(https://github.com/KMnO4-zx/agentic-rl-lab)"
)
DEFAULT_SEARCH_CONCURRENCY = {"deepseek": 16, "wikipedia": 3}
DEFAULT_SEARCH_TIMEOUT = {"deepseek": 60.0, "wikipedia": 15.0}


@dataclass(frozen=True)
class SearchItem:
    """保存一条搜索证据及其可选来源信息。"""

    title: str
    content: str
    source: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """保存一次搜索的证据或错误信息。"""

    ok: bool
    items: list[SearchItem]
    latency: float
    status: int | None = None
    error: str | None = None


@dataclass
class SearchStats:
    """累计并发搜索请求的运行指标。"""

    backend: str
    requests: int = 0
    successes: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    errors: int = 0
    latency_total: float = 0.0
    api_requests: int = 0
    billable_requests: int = 0
    result_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def metrics(self) -> dict[str, float]:
        """把累计计数转换成便于 SwanLab 记录的单次请求均值。"""
        with self._lock:
            denominator = max(self.requests, 1)
            metrics = {
                "search/success_rate": self.successes / denominator,
                "search/timeout_rate": self.timeouts / denominator,
                "search/429_rate": self.rate_limits / denominator,
                "search/error_rate": self.errors / denominator,
                "search/latency": self.latency_total / denominator,
                "search/results": self.result_count / denominator,
            }
            if self.backend == "deepseek":
                metrics.update(
                    {
                        "search/api_requests": self.api_requests / denominator,
                        "search/billable_requests": (
                            self.billable_requests / denominator
                        ),
                    }
                )
            return metrics


@dataclass
class DeepSeekSearchClient:
    """直接调用官方 Search API，把网页摘要和来源转成工具证据。"""

    api_key: str | None = field(default=None, repr=False)
    count: int = 3
    timeout: float = DEFAULT_SEARCH_TIMEOUT["deepseek"]
    max_retries: int = 1
    retry_delay: float = 1.0
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="deepseek")
    )

    def __post_init__(self) -> None:
        self.api_key = (
            self.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
        )
        if not self.api_key or not self.api_key.strip():
            raise ValueError("请在 experiments/search-r1/.env 或环境变量中设置 DEEPSEEK_API_KEY")
        self.api_key = self.api_key.strip()
        if not 1 <= self.count <= 50:
            raise ValueError("search_count 必须在 1 到 50 之间")

    @classmethod
    def from_env(
        cls, env_path: str | Path | None = None, **kwargs: Any
    ) -> "DeepSeekSearchClient":
        """读取项目 .env；已有的进程环境变量优先。"""
        if env_path:
            load_dotenv(env_path)
        return cls(**kwargs)

    def search(self, query: str) -> SearchResult:
        """搜索 Top N 网页；超时、408、429 和 5xx 最多有限重试。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        query = query.strip()
        if not query or len(query) > 128:
            return self._error_result(
                started, "query must contain 1 to 128 Unicode characters", 400
            )

        saw_timeout = False
        saw_rate_limit = False
        attempt = 0
        while True:
            try:
                with self.stats._lock:
                    self.stats.api_requests += 1
                response = httpx.post(
                    DEEPSEEK_SEARCH_ENDPOINT,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"query": query, "count": self.count, "need_content": False},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("results"), list
                ):
                    raise ValueError("invalid results")
                results = payload["results"]
                items: list[SearchItem] = []
                for item in results[: self.count]:
                    if not isinstance(item, dict):
                        raise ValueError("invalid result item")
                    # 优先使用 query 相关的短摘要，便于在小工具预算内读取多条证据。
                    content = str(item.get("snippet") or "").strip() or str(
                        item.get("summary") or ""
                    ).strip()
                    if content:
                        items.append(
                            SearchItem(
                                title=str(item.get("title") or "Untitled").strip(),
                                content=content,
                                source=str(item.get("site_name") or "").strip() or None,
                                url=str(item.get("url") or "").strip() or None,
                            )
                        )
                latency = time.perf_counter() - started
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += latency
                    self.stats.billable_requests += int(bool(results))
                    self.stats.result_count += len(items)
                return SearchResult(ok=True, items=items, latency=latency, status=200)
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                retryable = status in (408, 429) or status >= 500
                with self.stats._lock:
                    if status == 408 and not saw_timeout:
                        self.stats.timeouts += 1
                        saw_timeout = True
                    if status == 429 and not saw_rate_limit:
                        self.stats.rate_limits += 1
                        saw_rate_limit = True
                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {status}", status)
            except httpx.TimeoutException:
                with self.stats._lock:
                    if not saw_timeout:
                        self.stats.timeouts += 1
                        saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except httpx.HTTPError as error:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, type(error).__name__)
            except (ValueError, TypeError):
                return self._error_result(started, "invalid search response", 200)

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换为不会泄露密钥的工具结果。"""
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


@dataclass
class WikipediaSearchClient:
    """使用 Wikimedia Action API 搜索英文 Wikipedia 并返回页面正文片段。"""

    timeout: float = DEFAULT_SEARCH_TIMEOUT["wikipedia"]
    max_retries: int = 2
    retry_delay: float = 1.0
    min_request_interval: float = 0.31
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="wikipedia")
    )
    _rate_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _next_request_time: float = field(default=0.0, init=False, repr=False)

    def search(self, query: str) -> SearchResult:
        """搜索 Top 3 页面；所有并发调用共享约 200 RPM 的启动限速。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        saw_timeout = False
        saw_rate_limit = False
        attempt = 0
        while True:
            self._wait_for_rate_slot()
            try:
                result = self._request(query, started)
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += result.latency
                    self.stats.result_count += len(result.items)
                return result
            except urllib.error.HTTPError as error:
                if error.code == 429 and not saw_rate_limit:
                    with self.stats._lock:
                        self.stats.rate_limits += 1
                    saw_rate_limit = True
                retryable = error.code == 429 or error.code >= 500
                if retryable and attempt < self.max_retries:
                    retry_after = error.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else self.retry_delay * (2**attempt)
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {error.code}", error.code)
            except (TimeoutError, socket.timeout):
                if not saw_timeout:
                    with self.stats._lock:
                        self.stats.timeouts += 1
                    saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except urllib.error.URLError as error:
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    if not saw_timeout:
                        with self.stats._lock:
                            self.stats.timeouts += 1
                        saw_timeout = True
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        attempt += 1
                        continue
                    return self._error_result(started, "request timeout")
                return self._error_result(started, type(error).__name__)
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                return self._error_result(started, type(error).__name__)

    def _wait_for_rate_slot(self) -> None:
        """序列化请求启动时间，避免超过 Wikimedia 的识别客户端分钟限额。"""
        with self._rate_lock:
            now = time.monotonic()
            delay = self._next_request_time - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_request_time = now + self.min_request_interval

    def _request(self, query: str, started: float) -> SearchResult:
        """一次请求同时执行全文搜索并取得前三个页面的纯文本摘要。"""
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 3,
                "prop": "extracts|info",
                "explaintext": 1,
                "exintro": 1,
                "exchars": 1200,
                "inprop": "url",
                "redirects": 1,
                "utf8": 1,
            }
        )
        request = urllib.request.Request(
            f"{WIKIPEDIA_SEARCH_ENDPOINT}?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": WIKIPEDIA_USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            payload = json.loads(body.decode("utf-8"))
            pages = payload.get("query", {}).get("pages", [])
            if not isinstance(pages, list):
                raise TypeError("Wikipedia 搜索响应 pages 不是列表")
            if any(not isinstance(page, dict) for page in pages):
                raise TypeError("Wikipedia 搜索响应 page 不是对象")
            ordered_pages = sorted(
                pages,
                key=lambda page: int(page.get("index", 1_000_000)),
            )
            items = [
                SearchItem(
                    title=str(page.get("title") or "Untitled").strip(),
                    content=str(page.get("extract") or "").strip(),
                    source="Wikipedia",
                    url=str(page.get("fullurl") or "").strip(),
                )
                for page in ordered_pages
                if str(page.get("extract") or "").strip()
            ]
            return SearchResult(
                ok=True,
                items=items,
                latency=time.perf_counter() - started,
                status=response.status,
            )

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换成 rollout 可观察、不中断训练的搜索结果。"""
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


SearchClient = DeepSeekSearchClient | WikipediaSearchClient


def resolve_search_concurrency(backend: str, value: int | None) -> int:
    """返回用户设置或当前后端的默认搜索并发。"""
    if backend not in SEARCH_BACKENDS:
        raise ValueError(f"不支持的搜索后端: {backend}")
    concurrency = DEFAULT_SEARCH_CONCURRENCY[backend] if value is None else value
    if concurrency < 1:
        raise ValueError("search_concurrency 必须大于等于 1")
    return concurrency


def resolve_search_timeout(backend: str, value: float | None) -> float:
    """返回用户设置或当前后端的默认请求超时。"""
    if backend not in SEARCH_BACKENDS:
        raise ValueError(f"不支持的搜索后端: {backend}")
    timeout = DEFAULT_SEARCH_TIMEOUT[backend] if value is None else value
    if timeout <= 0:
        raise ValueError("search_timeout 必须大于 0")
    return timeout


def create_search_client(
    backend: str,
    env_path: str | Path | None = None,
    *,
    count: int = 3,
    timeout: float | None = None,
) -> SearchClient:
    """按名称创建搜索后端；两个后端共用同一套 rollout 接口。"""
    resolved_timeout = resolve_search_timeout(backend, timeout)
    if backend == "deepseek":
        return DeepSeekSearchClient.from_env(
            env_path,
            count=count,
            timeout=resolved_timeout,
        )
    if backend == "wikipedia":
        return WikipediaSearchClient(timeout=resolved_timeout)
    raise ValueError(f"不支持的搜索后端: {backend}")


def format_item(item: SearchItem, index: int) -> str:
    """按后端提供的信息格式化一条完整工具证据。"""
    lines = [
        f"[{index}] Title: {item.title}",
        f"    Content: {item.content}",
    ]
    if item.source:
        lines.append(f"    Source: {item.source}")
    if item.url:
        lines.append(f"    URL: {item.url}")
    return "\n".join(lines)
