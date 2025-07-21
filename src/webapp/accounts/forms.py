from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django import forms
from .models import User

# ユーザー登録フォームを作成する
class SignUpForm(UserCreationForm):
    account_id = forms.CharField(
        max_length=10,
        help_text="10文字まで入力可能だよ",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    first_name = forms.CharField(
        max_length=150,
        help_text="下の名前を入力してね",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    last_name = forms.CharField(
        max_length=150,
        help_text="上の名前を入力してね",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    birth_date = forms.DateField(
        help_text="誕生日をYYYY-MM-DDの形式で入力してね",
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        help_text="8文字以上の英数字で入力してね"
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        help_text="確認のため、もう一度同じパスワードを入力してね"
    )

    class Meta:
        model = User
        fields = ("account_id", "email", "first_name", "last_name", "birth_date")

# ログインフォームの作成
class LoginFrom(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'アカウントID'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'パスワード'})
    )
    
    class Meta:
        model = User