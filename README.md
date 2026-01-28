# 🎤 KOKORONIA - 音声対話分析システム

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.2-green.svg)](https://djangoproject.com)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Storage%20%7C%20Speech-orange.svg)](https://cloud.google.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌱 概要

KOKORONIAは**何気ない会話からあなたの好きを知る**  
をコンセプトにした、音声会話の録音・分析のWebアプリです。

---

## ✨ 主な特徴
**🎙️ デュアル音声録音**  
- デバイスを2台使いながら同時に録音
- Webブラウザから録音・アップロード

**☁️ クラウド連携**  
- Google Cloud Storage自動アップロード
- Google Speech-to-Text を用いた高精度な文字起こし

**📊 AI分析**  
- 興味・関心トピックの自動抽出
- 棒グラフアイコンで分かりやすい分析結果表示
- 会話の「魅力」や「親密さ」をサポート

---

## ✅ 現状の実装ステータス

**実装済み**  
- Django Webアプリの基本構成とAPI（管理画面/認証の土台含む）
- GCS連携ユーティリティと音声・テキスト解析スクリプト
- OpenAI SFTの学習/評価スクリプト群
- Docker/Composeのローカル起動フロー

**要準備（設定/運用）**  
- 実運用の秘密情報（APIキー/サービスアカウント/DB認証情報の投入）
- GCPの有効化・権限設定（GCS/Speech-to-Text 等）
- 本番環境の運用設計（監視、バックアップ、ローテーション等）

**実装中**  
- 自動テスト/CI/CDの整備
- 音声会話の分析結果を用いたマッチング/話題提案システムの実装
- コンテナ化/サービス間通信の実装(RestAPI)
---

## 🏗️ プロジェクト構造

```
lounge/
├── 📦 src/                    # コア機能
│   ├── 🎤 core/
│   │   ├── audio/            # 音声録音モジュール
│   │   ├── gcs/              # Google Cloud Storage
│   │   └── analysis/         # AI分析エンジン
│   └── 🌐 webapp/            # Django Webアプリケーション
│       └── templates/        # HTMLテンプレート
├── 🚀 scripts/               # 実行スクリプト
├── 📖 docs/                  # ドキュメント
├── 🔧 others/                # その他のツール
└── 🔐 credentials/           # 認証情報（.gitignore対象）
```

---

## 🛠️ 技術スタック

- **Python 3.10+**
- **Django 5.2**
- **PyAudio**
- **Google Cloud SDK / Storage / Speech-to-Text**
- **Transformers, PyTorch, MeCab, UniDic, Pandas**
