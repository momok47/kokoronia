from dataclasses import dataclass
from django.conf import settings
from rest_framework import authentication, exceptions


class ApiKeyAuthentication(authentication.BaseAuthentication):
    """
    シンプルなAPIキー認証。
    ヘッダ: X-API-Key: <key>
    """

    keyword = "X-API-Key"

    def authenticate(self, request):
        provided_key = request.headers.get(self.keyword)
        expected_key = getattr(settings, "API_KEY", None)

        if not expected_key:
            raise exceptions.AuthenticationFailed("APIキーが設定されていません。管理者に連絡してください。")

        if not provided_key:
            raise exceptions.AuthenticationFailed("APIキーが指定されていません。")

        if not secrets_safe_compare(provided_key, expected_key):
            raise exceptions.AuthenticationFailed("APIキーが正しくありません。")

        return (ApiKeyUser(), None)


def secrets_safe_compare(val1: str, val2: str) -> bool:
    """
    タイミング攻撃を避けるための一定時間比較。
    """
    if val1 is None or val2 is None:
        return False
    if len(val1) != len(val2):
        return False
    result = 0
    for x, y in zip(val1.encode(), val2.encode()):
        result |= x ^ y
    return result == 0


@dataclass
class ApiKeyUser:
    """
    APIキーだけで通すシンプルな疑似ユーザー。
    DRFのIsAuthenticated判定に通すため is_authenticated=True を返す。
    """

    def __post_init__(self):
        # 表示用の最低限の属性
        self.username = "api-key-user"

    @property
    def is_authenticated(self):
        return True