from typing import Iterable, Optional

from rest_framework.permissions import BasePermission


def _normalize_scopes(raw_scopes: Optional[object]) -> set[str]:
    """
    scopeクレームをスペース区切り文字列 or 配列どちらでも扱えるよう正規化。
    """
    if raw_scopes is None:
        return set()
    if isinstance(raw_scopes, str):
        return set(s for s in raw_scopes.split() if s)
    if isinstance(raw_scopes, Iterable):
        return set(str(s) for s in raw_scopes if s)
    return set()


class ScopePermission(BasePermission):
    """
    JWTのscope（文字列 or 配列）に required_scopes をすべて含むか検証する。
    """

    required_scopes: Iterable[str] = ()

    def has_permission(self, request, view) -> bool:
        required = set(self.required_scopes or [])

        # required_scopes が無ければ制約なし
        if not required:
            return True

        claims = {}
        if isinstance(request.auth, dict):
            claims = request.auth
        elif hasattr(request.user, "claims"):
            claims = getattr(request.user, "claims", {}) or {}

        token_scopes = _normalize_scopes(claims.get("scope"))
        return required.issubset(token_scopes)


class TopicsReadPermission(ScopePermission):
    required_scopes = ("topics:read",)


class TopicsWritePermission(ScopePermission):
    required_scopes = ("topics:write",)
