from dataclasses import dataclass
from typing import Optional, Tuple

import jwt
from django.conf import settings
from rest_framework import authentication, exceptions
from rest_framework.authentication import get_authorization_header


class JwtAuthentication(authentication.BaseAuthentication):
    """
    Bearerトークン(JWT)を検証する認証クラス。
    ヘッダ: Authorization: Bearer <token>
    """

    keyword = b"bearer"

    def authenticate(self, request) -> Optional[Tuple["JwtUser", dict]]:
        auth = get_authorization_header(request).split()

        # ヘッダ未指定 or 他スキームの場合はスキップ（他の認証に委ねる）
        if not auth:
            return None
        if auth[0].lower() != self.keyword:
            return None
        if len(auth) == 1:
            raise exceptions.AuthenticationFailed("トークンが指定されていません。")
        if len(auth) > 2:
            raise exceptions.AuthenticationFailed("無効なAuthorizationヘッダ形式です。")

        try:
            token = auth[1].decode()
        except UnicodeError:
            raise exceptions.AuthenticationFailed("トークンをデコードできません。")

        decoded = self._decode_token(token)
        user = JwtUser(decoded)
        return (user, decoded)

    def _decode_token(self, token: str) -> dict:
        alg = settings.JWT_ALGORITHM
        iss = settings.JWT_ISSUER
        aud = settings.JWT_AUDIENCE
        leeway = getattr(settings, "JWT_LEEWAY_SECONDS", 0)

        # 鍵の取得（HS系はシークレット、RS系は公開鍵）
        key = settings.JWT_SECRET
        if not key:
            raise exceptions.AuthenticationFailed("JWT_SECRET が設定されていません。")

        try:
            return jwt.decode(
                token,
                key=key,
                algorithms=[alg],
                issuer=iss,
                audience=aud,
                leeway=leeway,
                options={"require": ["exp", "iat"]},
            )
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed("トークンの有効期限が切れています。")
        except jwt.InvalidIssuerError:
            raise exceptions.AuthenticationFailed("issuer が一致しません。")
        except jwt.InvalidAudienceError:
            raise exceptions.AuthenticationFailed("audience が一致しません。")
        except jwt.InvalidTokenError as e:
            raise exceptions.AuthenticationFailed(f"無効なトークンです: {str(e)}")


@dataclass
class JwtUser:
    """
    JWTクレームを保持する疑似ユーザー。
    DRFのIsAuthenticated判定に通すため is_authenticated=True を返す。
    """

    claims: dict

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def username(self) -> str:
        # sub があれば優先して返す
        return str(self.claims.get("sub", "jwt-user"))
