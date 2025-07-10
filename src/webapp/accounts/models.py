from django.db import models
from django.db import models
from django.contrib.auth.models import (BaseUserManager,
                                        AbstractBaseUser,
                                        PermissionsMixin)
from django.utils.translation import gettext_lazy as _

# Djangoで，本サービスのユーザーモデル(=ログインする人)を作成する
class UserManager(BaseUserManager):
    def _create_user(self, email, account_id, password, **extra_fields):
        email = self.normalize_email(email)
        user = self.model(email=email, account_id=account_id, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)

        return user

    def create_user(self, email, account_id, password=None, **extra_fields):
        extra_fields.setdefault('is_active', True)
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        return self._create_user(
            email=email,
            account_id=account_id,
            password=password,
            **extra_fields,
        )

    def create_superuser(self, email, account_id, password, **extra_fields):
        extra_fields['is_active'] = True
        extra_fields['is_staff'] = True
        extra_fields['is_superuser'] = True
        return self._create_user(
            email=email,
            account_id=account_id,
            password=password,
            **extra_fields,
        )


class User(AbstractBaseUser, PermissionsMixin):
    """
    ユーザーの持つ情報
        account_id: アカウントID
        email: メールアドレス
        first_name/last_name: 名前
        birth_date: 誕生日
        is_superuser: スーパーユーザーかどうか
        is_staff: スタッフかどうか
        is_active: 有効なアカウントかどうか
        created_at: ユーザー作成日時
        updated_at: ユーザー更新日時
    """

    account_id = models.CharField(
        verbose_name=_("account_id"),
        unique=True,
        max_length=10
    )
    email = models.EmailField(
        verbose_name=_("email"),
        unique=True
    )
    first_name = models.CharField(
        verbose_name=_("first_name"),
        max_length=150,
        null=True,
        blank=False
    )
    last_name = models.CharField(
        verbose_name=_("last_name"),
        max_length=150,
        null=True,
        blank=False
    )
    birth_date = models.DateField(
        verbose_name=_("birth_date"),
        blank=True,
        null=True
    )
    is_superuser = models.BooleanField(
        verbose_name=_("is_superuser"),
        default=False
    )
    is_staff = models.BooleanField(
        verbose_name=_('staff status'),
        default=False,
    )
    is_active = models.BooleanField(
        verbose_name=_('active'),
        default=True,
    )
    created_at = models.DateTimeField(
        verbose_name=_("created_at"),
        auto_now_add=True
    )
    updated_at = models.DateTimeField(
        verbose_name=_("updated_at"),
        auto_now=True
    )

    objects = UserManager()

    USERNAME_FIELD = 'account_id' # ログイン時，ユーザー名の代わりにaccount_idを使用する
    REQUIRED_FIELDS = ['email']  # スーパーユーザー作成時にemailも設定する

    def __str__(self):
        return self.account_id


class UserTopicScore(models.Model):
    """
    ユーザーごとのトピックスコアを格納するモデル
    各ユーザーの各トピックに対するスコアの累積平均を保持
    """
    
    # 利用可能なトピックラベル（interests_extraction.pyと同期）
    TOPIC_CHOICES = [
        ("社会", "社会"),
        ("まなび", "まなび"),
        ("テクノロジー", "テクノロジー"),
        ("カルチャー", "カルチャー"),
        ("アウトドア", "アウトドア"),
        ("フード", "フード"),
        ("旅行おでかけ", "旅行おでかけ"),
        ("ライフスタイル", "ライフスタイル"),
        ("ビジネス", "ビジネス"),
        ("読書", "読書"),
        ("キャリア", "キャリア"),
        ("デザイン", "デザイン"),
        ("IT", "IT"),
        ("経済投資", "経済投資"),
        ("ネットワーク", "ネットワーク"),
    ]
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='topic_scores',
        verbose_name=_("ユーザー")
    )
    topic_label = models.CharField(
        max_length=20,
        choices=TOPIC_CHOICES,
        verbose_name=_("トピックラベル")
    )
    score = models.FloatField(
        default=0.0,
        verbose_name=_("スコア"),
        help_text="累積平均スコア"
    )
    count = models.PositiveIntegerField(
        default=0,
        verbose_name=_("更新回数"),
        help_text="スコアが更新された回数"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_("作成日時")
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_("更新日時")
    )
    
    class Meta:
        unique_together = ['user', 'topic_label']
        verbose_name = _("ユーザートピックスコア")
        verbose_name_plural = _("ユーザートピックスコア")
        indexes = [
            models.Index(fields=['user', 'topic_label']),
        ]
    
    def __str__(self):
        return f"{self.user.account_id} - {self.topic_label}: {self.score:.2f}"
    
    def update_score(self, new_score):
        """
        スコアを更新する（累積平均を計算）
        
        Args:
            new_score (float): 新しいスコア値
        
        Returns:
            float: 更新後のスコア値
        """
        if self.count == 0:
            # 初回の場合はそのまま格納
            self.score = new_score
            self.count = 1
        else:
            # 累積平均を計算: (現在のスコア * 回数 + 新しいスコア) / (回数 + 1)
            total_score = self.score * self.count + new_score
            self.count += 1
            self.score = total_score / self.count
        
        self.save()
        return self.score
    
    @classmethod
    def update_user_topic_score(cls, user, topic_label, new_score):
        """
        指定されたユーザーとトピックのスコアを更新する
        
        Args:
            user (User): 対象ユーザー
            topic_label (str): トピックラベル
            new_score (float): 新しいスコア値
        
        Returns:
            tuple: (UserTopicScore instance, created: bool)
        """
        topic_score, created = cls.objects.get_or_create(
            user=user,
            topic_label=topic_label,
            defaults={'score': 0.0, 'count': 0}
        )
        
        final_score = topic_score.update_score(new_score)
        
        return topic_score, created
    
    @classmethod
    def get_user_topic_matrix(cls, user):
        """
        指定されたユーザーのトピック×スコアマトリックスを取得
        
        Args:
            user (User): 対象ユーザー
        
        Returns:
            dict: {topic_label: score, ...}
        """
        user_scores = cls.objects.filter(user=user).values('topic_label', 'score')
        topic_matrix = {choice[0]: 0.0 for choice in cls.TOPIC_CHOICES}
        
        for score_data in user_scores:
            topic_matrix[score_data['topic_label']] = score_data['score']
        
        return topic_matrix