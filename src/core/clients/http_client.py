import json
import random
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable
from urllib import error, request


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None, body: str | None, request_id: str | None):
        super().__init__(message)
        self.status = status
        self.body = body
        self.request_id = request_id


class CircuitBreakerOpenError(RuntimeError):
    pass


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 5.0
    jitter_ratio: float = 0.2
    retryable_statuses: tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    retryable_exceptions: tuple[type[BaseException], ...] = (
        TimeoutError,
        socket.timeout,
        error.URLError,
    )

    def get_delay(self, attempt: int) -> float:
        if attempt <= 1:
            return 0.0
        exp_delay = self.base_delay_seconds * (2 ** (attempt - 2))
        capped = min(exp_delay, self.max_delay_seconds)
        jitter = capped * self.jitter_ratio * random.random()
        return capped + jitter


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_timeout_seconds: int = 30
    half_open_success_threshold: int = 1

    _state: str = "closed"
    _failure_count: int = 0
    _opened_at: float | None = None
    _half_open_successes: int = 0

    def allow_request(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if self._opened_at is None:
                return False
            if time.time() - self._opened_at >= self.reset_timeout_seconds:
                self._state = "half_open"
                self._half_open_successes = 0
                return True
            return False
        return True

    def record_success(self) -> None:
        if self._state == "half_open":
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_success_threshold:
                self._reset()
        else:
            self._reset()

    def record_failure(self) -> None:
        if self._state == "half_open":
            self._trip()
            return
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = "open"
        self._opened_at = time.time()
        self._failure_count = 0
        self._half_open_successes = 0

    def _reset(self) -> None:
        self._state = "closed"
        self._opened_at = None
        self._failure_count = 0
        self._half_open_successes = 0


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: str | None
    request_id: str | None

    def json(self) -> Any:
        if self.body is None or not self.body.strip():
            return None
        return json.loads(self.body)


class HttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        default_headers: dict[str, str] | None = None,
        timeout_seconds: float = 10.0,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        auto_idempotency: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_headers = default_headers or {}
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.auto_idempotency = auto_idempotency

    def request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> Any:
        response = self.request(
            method,
            path,
            json_body=json_body,
            headers=headers,
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        if 200 <= response.status < 300:
            return response.json()
        message = f"API request failed with status {response.status}"
        raise ApiError(message, response.status, response.body, response.request_id)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> HttpResponse:
        if self.circuit_breaker and not self.circuit_breaker.allow_request():
            raise CircuitBreakerOpenError("Circuit breaker is open")

        url = self._build_url(path)
        request_id = request_id or str(uuid.uuid4())
        merged_headers = self._build_headers(
            headers=headers or {},
            method=method,
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        payload = None
        if json_body is not None:
            payload = json.dumps(json_body).encode("utf-8")

        attempts = max(1, self.retry_policy.max_attempts)
        last_error: ApiError | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self._send(url, method, payload, merged_headers)
                if self.circuit_breaker:
                    if 200 <= response.status < 500:
                        self.circuit_breaker.record_success()
                    else:
                        self.circuit_breaker.record_failure()
                if response.status in self.retry_policy.retryable_statuses and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                return response
            except self.retry_policy.retryable_exceptions as exc:  # type: ignore[misc]
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                if attempt >= attempts:
                    raise ApiError(str(exc), None, None, request_id) from exc
                self._sleep_backoff(attempt)
            except ApiError as exc:
                last_error = exc
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                if exc.status in self.retry_policy.retryable_statuses and attempt < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise

        if last_error:
            raise last_error
        raise ApiError("API request failed without response", None, None, request_id)

    def _send(self, url: str, method: str, payload: bytes | None, headers: dict[str, str]) -> HttpResponse:
        req = request.Request(url, data=payload, headers=headers, method=method.upper())
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body_bytes = response.read()
                body = body_bytes.decode("utf-8") if body_bytes else None
                return HttpResponse(
                    status=response.status,
                    headers=self._normalize_headers(response.headers),
                    body=body,
                    request_id=headers.get("X-Request-Id"),
                )
        except error.HTTPError as exc:
            body_bytes = exc.read()
            body = body_bytes.decode("utf-8") if body_bytes else None
            raise ApiError(str(exc), exc.code, body, headers.get("X-Request-Id")) from exc

    def _build_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    def _build_headers(
        self,
        *,
        headers: dict[str, str],
        method: str,
        idempotency_key: str | None,
        request_id: str,
    ) -> dict[str, str]:
        merged = {**self.default_headers, **headers}
        merged.setdefault("Accept", "application/json")
        merged.setdefault("X-Request-Id", request_id)
        if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            merged.setdefault("Content-Type", "application/json")
            if self.auto_idempotency or idempotency_key:
                merged.setdefault("Idempotency-Key", idempotency_key or request_id)
        return merged

    def _normalize_headers(self, headers: Iterable[tuple[str, str]]) -> dict[str, str]:
        return {key: value for key, value in headers}

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.retry_policy.get_delay(attempt)
        if delay > 0:
            time.sleep(delay)
