# 🎤 KOKORONIA - 音声対話分析システム

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.2-green.svg)](https://djangoproject.com)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Storage%20%7C%20Speech-orange.svg)](https://cloud.google.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 概要

**KOKORONIA**は、リアルタイム音声録音、文字起こし、興味分析を統合した次世代音声対話分析システムです。  
CA Tech Loungeでの活動でメンターの方と開発し、複数デバイスでの同時録音、Google Cloud Servicesを活用した高精度文字起こし、AI/MLによる感情・興味分析を提供します。

## ✨ 主な機能

🎙️ **デュアル音声録音**
- 複数デバイスでの同時録音
- リアルタイム音声データ取得
- WAVフォーマットでの高品質録音

☁️ **クラウド連携**
- Google Cloud Storage自動アップロード
- Google Speech-to-Text API文字起こし
- JSON形式での構造化データ保存

🤖 **AI分析機能**
- 興味・感情抽出
- ゼロショット学習による分類
- トランスフォーマーモデル活用

🌐 **Webインターフェース**
- Django REST Framework
- ユーザー認証システム
- レスポンシブWebUI

## 🏗️ プロジェクト構造

```
lounge/
├── 📦 src/                    # コア機能
│   ├── 🎤 core/
│   │   ├── audio/            # 音声録音モジュール
│   │   │   ├── device1_audio_recorder.py
│   │   │   └── device2_audio_recorder.py
│   │   ├── gcs/              # Google Cloud Storage
│   │   │   ├── gcs_uploader.py
│   │   │   └── transcribe_audio_from_gcs.py
│   │   └── analysis/         # AI分析エンジン
│   │       ├── interests_extraction.py
│   │       └── zero_shot_learning.py
│   └── 🌐 webapp/            # Django Webアプリケーション
│       ├── manage.py
│       ├── accounts/         # ユーザー管理
│       ├── project/          # プロジェクト設定
│       └── templates/        # HTMLテンプレート
├── 🚀 scripts/               # 実行スクリプト
│   └── main.py              # メインアプリケーション
├── 📖 docs/                  # ドキュメント
├── 🧪 tests/                 # テストファイル
├── 🎭 mock_data/             # モックデータ・サンプル
├── 🔧 others/                # その他のツール
└── 🔐 credentials/           # 認証情報
```

## 🛠️ 技術スタック

### **バックエンド**
- **Python 3.10+** - メイン言語
- **Django 5.2** - Webフレームワーク
- **PyAudio** - 音声録音
- **Google Cloud SDK** - クラウドサービス連携

### **AI/ML**
- **Transformers** - 自然言語処理
- **PyTorch** - 機械学習フレームワーク
- **MeCab + UniDic** - 日本語形態素解析
- **Pandas** - データ処理

### **インフラ**
- **Google Cloud Storage** - ファイル保存
- **Google Speech-to-Text** - 音声認識
- **SQLite** - データベース（開発環境）

## 🚀 セットアップ

### **1. 前提条件**

- Python 3.10以上
- Google Cloud Platform アカウント
- 音声入力デバイス（2台推奨）

### **2. インストール**

```bash
# リポジトリクローン
git clone https://github.com/momok47/lounge.git
cd lounge

# 仮想環境作成・有効化
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係インストール
pip install -e .
```

### **3. Google Cloud 設定**

```bash
# Google Cloud SDKインストール
# https://cloud.google.com/sdk/docs/install

# 認証設定
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# サービスアカウントキー設置
export GOOGLE_APPLICATION_CREDENTIALS="credentials/your-service-account.json"
```

### **4. Django設定**

```bash
# データベース初期化
cd src/webapp
python manage.py migrate

# スーパーユーザー作成
python manage.py createsuperuser

# 開発サーバー起動
python manage.py runserver
```

## 📱 使用方法

### **音声録音・分析**

```bash
# メインアプリケーション実行
python scripts/main.py
```

1. 利用可能な録音デバイス一覧が表示
2. デバイス1・2を選択し、ユーザーIDを入力
3. 録音開始（Enter押下で停止）
4. 自動で文字起こし・分析実行
5. 結果がGoogle Cloud Storageに保存

### **Webインターフェース**

```bash
# Webアプリケーション起動
cd src/webapp
python manage.py runserver
```

- **ユーザー登録・ログイン**: `http://localhost:8000/accounts/`
- **管理画面**: `http://localhost:8000/admin/`
- **メインダッシュボード**: `http://localhost:8000/`

## 🧪 テスト

```bash
# 全テスト実行
python -m pytest tests/

# 特定モジュールテスト
python -m pytest tests/test_audio.py
python -m pytest tests/test_analysis.py
```

## 📊 設定

### **環境変数**

`.env`ファイルを作成：

```env
# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=credentials/your-service-account.json
GCS_BUCKET_NAME=your-bucket-name

# Django
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# データベース
DATABASE_URL=sqlite:///db.sqlite3
```

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. フィーチャーブランチ作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエスト作成

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 👥 チーム

- **[@momok47](https://github.com/momok47)** - メイン開発者
- **CA Tech Loungeメンターさん** - プロジェクトサポート

## 🙋‍♀️ サポート

問題や質問がある場合は、[Issues](https://github.com/momok47/lounge/issues)を作成してください。

## 📚 関連ドキュメント

- [API Documentation](docs/api.md)
- [開発者ガイド](docs/development.md)
- [デプロイメント](docs/deployment.md)
- [よくある質問](docs/faq.md)
