from typing import Any

from rest_framework import serializers

from accounts.models import User


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, allow_blank=False)

    class Meta:
        model = User
        fields = [
            "id",
            "account_id",
            "email",
            "first_name",
            "last_name",
            "birth_date",
            "created_at",
            "updated_at",
            "password",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def create(self, validated_data: dict[str, Any]) -> User:
        password = validated_data.pop("password")
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user

    def update(self, instance: User, validated_data: dict[str, Any]) -> User:
        password = validated_data.pop("password", None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance


class MessageSerializer(serializers.Serializer):
    role = serializers.ChoiceField(choices=["user", "assistant"])
    content = serializers.CharField()
    timestamp = serializers.DateTimeField(required=False, allow_null=True)


class TopicSerializer(serializers.Serializer):
    title = serializers.CharField()
    summary = serializers.CharField()


class TopicSuggestionSerializer(serializers.Serializer):
    title = serializers.CharField()
    summary = serializers.CharField()
    confidence = serializers.FloatField(required=False, min_value=0.0, max_value=1.0)


class TopicExtractRequestSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    conversation = MessageSerializer(many=True)
    max_topics = serializers.IntegerField(required=False, min_value=1, max_value=20, default=5)


class TopicExtractResponseSerializer(serializers.Serializer):
    request_id = serializers.CharField()
    created_at = serializers.DateTimeField()
    topics = TopicSerializer(many=True)


class TopicSuggestRequestSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    conversation = MessageSerializer(many=True)
    topics_hint = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True,
    )
    max_suggestions = serializers.IntegerField(required=False, min_value=1, max_value=10, default=3)
    context_tags = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True,
    )


class TopicSuggestResponseSerializer(serializers.Serializer):
    request_id = serializers.CharField()
    created_at = serializers.DateTimeField()
    suggestions = TopicSuggestionSerializer(many=True)

