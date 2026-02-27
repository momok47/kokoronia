from dataclasses import dataclass
from typing import Any, Literal

from .http_client import CircuitBreaker, HttpClient, RetryPolicy


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"role": self.role, "content": self.content}
        if self.timestamp:
            data["timestamp"] = self.timestamp
        return data


@dataclass(frozen=True)
class Topic:
    title: str
    summary: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Topic":
        return Topic(title=data["title"], summary=data["summary"])


@dataclass(frozen=True)
class TopicSuggestion:
    title: str
    summary: str
    confidence: float | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TopicSuggestion":
        return TopicSuggestion(
            title=data["title"],
            summary=data["summary"],
            confidence=data.get("confidence"),
        )


@dataclass(frozen=True)
class TopicSuggestRequest:
    user_id: str
    conversation: list[Message]
    topics_hint: list[str] | None = None
    max_suggestions: int | None = None
    context_tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "user_id": self.user_id,
            "conversation": [message.to_dict() for message in self.conversation],
        }
        if self.topics_hint is not None:
            payload["topics_hint"] = self.topics_hint
        if self.max_suggestions is not None:
            payload["max_suggestions"] = self.max_suggestions
        if self.context_tags is not None:
            payload["context_tags"] = self.context_tags
        return payload


@dataclass(frozen=True)
class TopicSuggestResponse:
    request_id: str
    created_at: str
    suggestions: list[TopicSuggestion]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TopicSuggestResponse":
        return TopicSuggestResponse(
            request_id=data["request_id"],
            created_at=data["created_at"],
            suggestions=[
                TopicSuggestion.from_dict(item) for item in data.get("suggestions", [])
            ],
        )


class ConversationApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        jwt_token: str | None = None,
        timeout_seconds: float = 10.0,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        auto_idempotency: bool = True,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["X-API-Key"] = api_key
        if jwt_token:
            headers["Authorization"] = f"Bearer {jwt_token}"
        self.http = HttpClient(
            base_url=base_url,
            default_headers=headers,
            timeout_seconds=timeout_seconds,
            retry_policy=retry_policy,
            circuit_breaker=circuit_breaker,
            auto_idempotency=auto_idempotency,
        )

    def suggest_topics(
        self,
        request_data: TopicSuggestRequest,
        *,
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ) -> TopicSuggestResponse:
        data = self.http.request_json(
            "POST",
            "/api/v1/topics/suggest",
            json_body=request_data.to_dict(),
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
        return TopicSuggestResponse.from_dict(data)
