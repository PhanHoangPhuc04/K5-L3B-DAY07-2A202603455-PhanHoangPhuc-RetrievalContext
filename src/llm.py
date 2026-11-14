"""Client DeepSeek (API tương thích OpenAI, chỉ dùng thư viện chuẩn).

Key đọc lúc chạy từ biến môi trường DEEPSEEK_API_KEY hoặc file .env ở gốc repo (đã gitignore).
Không in key ra log. Lỗi mạng/429/5xx thử lại tối đa 3 lần; hết lượt thì ném lỗi thật, không giả kết quả.
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request

from .common import ROOT

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = os.environ.get("DS_MODEL", "deepseek-flash")
_CA = "/etc/ssl/cert.pem"
_CTX = ssl.create_default_context(cafile=_CA) if os.path.exists(_CA) else ssl.create_default_context()


class LLMError(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise LLMError("Thiếu DEEPSEEK_API_KEY (biến môi trường hoặc .env)")
    return key


def chat(messages: list[dict], tools: list[dict] | None = None, tool_choice: str | None = None,
         retries: int = 3, timeout: int = 120) -> tuple[dict, dict]:
    """Trả (message, meta). meta gồm usage, latency_s, finish_reason, attempts, model."""
    body = {"model": MODEL, "messages": messages, "temperature": 0,
            "thinking": {"type": "disabled"}, "max_tokens": 800}
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_key()}"}
    last = None
    for attempt in range(1, retries + 1):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(API_URL, data, headers)
            raw = urllib.request.urlopen(req, context=_CTX, timeout=timeout).read()
            j = json.loads(raw)
            choice = j["choices"][0]
            return choice["message"], {"usage": j.get("usage", {}), "latency_s": round(time.perf_counter() - t0, 3),
                                       "finish_reason": choice.get("finish_reason"), "attempts": attempt,
                                       "model": j.get("model", MODEL)}
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {msg}"
            if e.code not in (429, 500, 502, 503, 504):
                raise LLMError(last) from None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError) as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 * attempt)
    raise LLMError(f"hết {retries} lần thử: {last}")
