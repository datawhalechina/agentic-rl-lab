"""作者的两个 XML 工具：google_search 用 DeepSeek Search API，browse_webpage 用 Jina Reader。

工具失败时不抛异常，而是把错误文本作为 observation 返回给模型，和作者一致。

Adapted from experiments/search-r1 and PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.
"""

import fcntl
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

from protocol import ToolCall, truncate_tool_text


DEEPSEEK_SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
DEEPSEEK_MAX_QUERY_CHARS = 128
JINA_RATE_LIMIT_FILE = Path(tempfile.gettempdir()) / "tournament_grpo_jina_global_rate_limit.state"


@dataclass(frozen=True)
class ToolResult:
    text: str
    ok: bool  # 只用于统计工具错误率；模型只看到 text


def proxy_from_env(*names: str) -> str | None:
    """按作者的变量优先级读取代理；只写 host:port 时补 http://。"""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value if "://" in value else "http://" + value
    return None


def wait_for_jina() -> None:
    """作者的跨进程限速：所有进程共享一个时间戳文件，默认 20 RPM（无 Key 的额度）。

    文件里记着最近一次预约的请求时间，新请求排在它之后 interval 秒。
    """
    interval = 60.0 / int(os.getenv("JINA_RATE_LIMIT_RPM", "20"))
    with JINA_RATE_LIMIT_FILE.open("a+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)  # 关闭文件时自动解锁
        file.seek(0)
        reserved = max(time.time(), float(file.read() or 0) + interval)
        file.seek(0)
        file.truncate()
        file.write(str(reserved))
    time.sleep(max(0.0, reserved - time.time()))


def read_page(response: httpx.Response, url: str) -> tuple[str, str, str]:
    """返回 (title, url, content)。

    Jina 返回 JSON 时按作者 _extract_webpage_payload 的字段顺序取正文，否则直接使用纯文本。
    """
    if "json" in response.headers.get("content-type", "").lower():
        payload = response.json()
        data = payload.get("data", payload)
        content = (data.get("content") or data.get("text") or data.get("markdown")
                   or data.get("body") or data.get("result") or "").strip()
        if content:
            return (data.get("title") or "").strip(), (data.get("url") or "").strip() or url, content
    return "", url, response.text.strip()


@dataclass
class ResearchTools:
    deepseek_api_key: str = field(repr=False)
    jina_api_key: str = field(default="", repr=False)
    search_proxy: str | None = None
    jina_proxy: str | None = None
    jina_reader_base_url: str = "https://r.jina.ai"
    timeout: float = 30.0
    search_count: int = 3  # 作者为 Top 10，本复现降为 Top 3

    @classmethod
    def from_env(cls, env_path: str | Path) -> "ResearchTools":
        load_dotenv(env_path)
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise ValueError("请在 experiments/tournament-grpo/.env 中设置 DEEPSEEK_API_KEY")
        return cls(
            key, os.getenv("JINA_API_KEY", "").strip(),
            search_proxy=proxy_from_env("DEEPSEEK_SEARCH_PROXY"),
            jina_proxy=proxy_from_env("JINA_READER_HTTPS_PROXY", "JINA_READER_PROXY", "JINA_HTTPS_PROXY"),
            jina_reader_base_url=os.getenv("JINA_READER_BASE_URL", "https://r.jina.ai"),
        )

    def search(self, query: str) -> ToolResult:
        """每条搜索结果按作者的 <snippet> 格式输出。"""
        # DeepSeek 限制 query 最多 128 字符；轨迹里仍保留模型生成的原始 query。
        query = query.strip()[:DEEPSEEK_MAX_QUERY_CHARS]
        if not query:
            return ToolResult("Error: query is required.", False)
        try:
            response = httpx.post(
                DEEPSEEK_SEARCH_URL,
                headers={"Authorization": f"Bearer {self.deepseek_api_key}"},
                # 搜索只给摘要，模型要读正文时再调用 browse_webpage。
                json={"query": query, "count": self.search_count, "need_content": False},
                timeout=self.timeout, proxy=self.search_proxy, trust_env=False,
            )
            response.raise_for_status()
            blocks = []
            for item in response.json()["results"][:self.search_count]:
                title = (item.get("title") or "").strip()
                url = (item.get("url") or "").strip()
                snippet = (item.get("snippet") or item.get("summary") or "").strip()
                date = (item.get("date") or "").strip()
                if not (title or url or snippet):
                    continue
                lines = [f"Title: {title}", f"URL: {url}", f"Search Snippet: {snippet}"]
                if date:
                    lines.append(f"Date: {date}")
                blocks.append("<snippet>\n" + "\n".join(lines) + "\n</snippet>")
            content = "\n".join(blocks) or "<snippet>\nNo search results found.\n</snippet>"
            return ToolResult(f"<tool_output>\n{content}\n</tool_output>", True)
        except Exception as exc:
            return ToolResult(f"Error performing google_search: {exc}", False)

    def browse(self, url: str) -> ToolResult:
        """调用 Jina Reader；429 时最多尝试 3 次。正文先截断一次，execute 再截断整个 observation。"""
        url = url.strip()
        if not url:
            return ToolResult("Error: url is required.", False)
        if "://" not in url:
            url = "https://" + url.lstrip("/")
        # 和作者一样在 try 外校验 URL，由 execute 返回通用错误文本。
        if not urlparse(url).netloc:
            raise ValueError(f"Invalid URL: {url}")
        headers = {"Authorization": f"Bearer {self.jina_api_key}"} if self.jina_api_key else {}
        try:
            for attempt in range(3):
                wait_for_jina()
                response = httpx.get(
                    f"{self.jina_reader_base_url.rstrip('/')}/{url}", headers=headers,
                    timeout=self.timeout, follow_redirects=True, proxy=self.jina_proxy, trust_env=False,
                )
                if response.status_code != 429:
                    break
                time.sleep(2 * 2**attempt)
            response.raise_for_status()
            title, url, content = read_page(response, url)
            if not content:
                raise ValueError("Jina Reader response did not include readable page content")
            text = (f"<tool_output>\n<webpage>\nTitle: {title or 'No title available'}\n"
                    f"URL: {url}\nContent: {truncate_tool_text(content)}\n</webpage>\n</tool_output>")
            return ToolResult(text, True)
        except Exception as exc:
            return ToolResult(f"Error performing browse_webpage: {exc}", False)

    def execute(self, call: ToolCall) -> ToolResult:
        """分发工具调用；未知工具和非法 URL 用作者的通用错误文本，最后整体再截断一次。"""
        try:
            if call.name == "google_search":
                result = self.search(call.arguments.get("query", ""))
            elif call.name == "browse_webpage":
                result = self.browse(call.arguments.get("url", ""))
            else:
                raise KeyError(call.name)
        except Exception as exc:
            return ToolResult(f"Error when executing tool: {exc}", False)
        return ToolResult(truncate_tool_text(result.text), result.ok)
