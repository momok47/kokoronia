import uuid
from typing import List

from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.models import User
from .serializers import (
    TopicSuggestRequestSerializer,
    TopicSuggestResponseSerializer,
    UserSerializer,
)
from .permissions import TopicsWritePermission


class UserViewSet(viewsets.ModelViewSet):
    """
    ユーザーCRUD用エンドポイント。
    """

    queryset = User.objects.all().order_by("-created_at")
    serializer_class = UserSerializer


class TopicSuggestView(APIView):
    """
    会話ログとヒントから具体的な話題提案を返すエンドポイント。
    """

    permission_classes = [TopicsWritePermission]

    def post(self, request):
        req = TopicSuggestRequestSerializer(data=request.data)
        req.is_valid(raise_exception=True)
        data = req.validated_data

        user = resolve_user(data["user_id"])
        if user is None:
            return Response({"detail": "指定したユーザーが見つかりません。"}, status=status.HTTP_404_NOT_FOUND)

        suggestions = dummy_suggest_topics(
            conversation=data["conversation"],
            topics_hint=data.get("topics_hint") or [],
            max_suggestions=data.get("max_suggestions") or 3,
            context_tags=data.get("context_tags") or [],
        )

        res = TopicSuggestResponseSerializer(
            {
                "request_id": str(uuid.uuid4()),
                "created_at": timezone.now(),
                "suggestions": suggestions,
            }
        )
        return Response(res.data, status=status.HTTP_200_OK)


def resolve_user(user_id: str):
    """
    user_idがPKまたはaccount_idのどちらでも検索できるようにする。
    """
    try:
        return User.objects.get(pk=user_id)
    except User.DoesNotExist:
        try:
            return User.objects.get(account_id=user_id)
        except User.DoesNotExist:
            return None


def dummy_suggest_topics(
    conversation: List[dict],
    topics_hint: List[str],
    max_suggestions: int,
    context_tags: List[str],
):
    """
    シンプルなダミー提案: ヒントと会話末尾からタイトルを生成。
    """
    texts = [m.get("content", "") for m in conversation if m.get("content")]
    base_titles = list(topics_hint)
    if texts:
        base_titles.append(texts[-1][:50])
    if not base_titles:
        base_titles = ["フリートーク", "最近の関心ごと"]

    suggestions = []
    for idx, title in enumerate(base_titles[:max_suggestions]):
        suggestions.append(
            {
                "title": title or f"Suggestion {idx + 1}",
                "summary": f"{title} について掘り下げる提案",
                "confidence": max(0.0, 0.9 - 0.1 * idx),
            }
        )
    return suggestions[:max_suggestions]

