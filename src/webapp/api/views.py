import uuid

from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from accounts.models import User
from accounts.topic_suggestion_service import generate_next_topic_sentence
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

        suggestions = suggest_topics_with_gpt(
            user_id=data["user_id"],
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


def suggest_topics_with_gpt(
    user_id: str,
    conversation: list[dict],
    topics_hint: list[str],
    max_suggestions: int,
    context_tags: list[str],
):
    """画面表示と同じロジックで話題提案を1件返す。"""
    partner_user_id = context_tags[0] if context_tags else user_id
    sentence = generate_next_topic_sentence(user_id, partner_user_id)
    return [
        {
            "title": "次回のおすすめ話題",
            "summary": sentence,
        }
    ]

