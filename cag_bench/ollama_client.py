
import time
from typing import Any, Dict, List
import requests

class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        retries: int = 1,
        retry_backoff: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff

    def _post_json(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        for attempt in range(self.retries + 1):
            try:
                return requests.post(url, json=payload, timeout=self.timeout)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt >= self.retries:
                    raise
                time.sleep(self.retry_backoff * (2**attempt))
        raise RuntimeError("Unreachable retry state")

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.1, num_ctx: int = 8192) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature, "num_ctx": num_ctx}}
        started = time.time()
        resp = self._post_json(url, payload)
        latency = time.time() - started
        resp.raise_for_status()
        data = resp.json()
        return {"content": data.get("message", {}).get("content", ""), "latency_seconds": latency, "raw": data}

def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))
