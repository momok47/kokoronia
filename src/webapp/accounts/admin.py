from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, UserTopicScore  # 自作のカスタムユーザーモデルをインポート

# Djangoの標準になっているユーザー管理画面をカスタムユーザーモデルに合わせて調節したい

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    # 管理画面の並び順をuseridで指定
    ordering = ['account_id']
    list_display = ['account_id', 'email', 'is_staff', 'is_active']# 管理画面の一覧に表示される情報
    search_fields = ['account_id', 'email']

    # 管理画面で既存ユーザーを開いた時の，表示レイアウト
    fieldsets = (
        (None, {'fields': ('account_id', 'email', 'password')}),
        ('Personal Info', {'fields': ('first_name', 'last_name', 'birth_date')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login',)}),
    )

    # 新規ユーザー作成時の表示レイアウト
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('account_id', 'email', 'password1', 'password2', 'is_staff', 'is_active')}
        ),
    )

@admin.register(UserTopicScore)
class UserTopicScoreAdmin(admin.ModelAdmin):
    list_display = ['user', 'topic_label', 'score', 'count', 'updated_at']
    list_filter = ['topic_label', 'updated_at']
    search_fields = ['user__account_id', 'user__email', 'topic_label']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['user', 'topic_label']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')