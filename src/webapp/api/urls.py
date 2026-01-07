from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import TopicExtractView, TopicSuggestView, UserViewSet

router = DefaultRouter()
router.register(r"v1/users", UserViewSet, basename="user")

urlpatterns = [
    path("", include(router.urls)),
    path("v1/topics/extract", TopicExtractView.as_view(), name="topic-extract"),
    path("v1/topics/suggest", TopicSuggestView.as_view(), name="topic-suggest"),
]