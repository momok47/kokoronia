# 🎤 KOKORONIA - 音声対話分析システム

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.2-green.svg)](https://djangoproject.com)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Storage%20%7C%20Speech-orange.svg)](https://cloud.google.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌱 概要

**KOKORONIA**は  
「何気ない会話からあなたの好きを知る」  
をコンセプトにした、音声会話の録音・分析・マッチング提案Webアプリです。

---

## ✨ 主な特徴

**🌿 直感的なWeb UI**  
- 桜・深緑・クリーム色を基調としたやさしいデザイン
- スマホ・PC両対応のレスポンシブ設計
- シンプルなナビゲーションと分かりやすいボタン

**🎙️ デュアル音声録音**  
- 2台のマイクを同時に選択・録音
- Webブラウザから録音・アップロード

**☁️ クラウド連携**  
- Google Cloud Storage自動アップロード
- Google Speech-to-Text APIで高精度な文字起こし

**📊 AI分析**  
- 興味・関心トピックの自動抽出
- 棒グラフアイコンで分かりやすい分析結果表示
- 会話の「魅力」や「親密さ」をサポート

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
├── 🧪 tests/                 # テストファイル
├── 🎭 mock_data/             # モックデータ・サンプル
├── 🔧 others/                # その他のツール
└── 🔐 credentials/           # 認証情報
```

---

## 🛠️ 技術スタック

- **Python 3.10+**
- **Django 5.2**
- **PyAudio**
- **Google Cloud SDK / Storage / Speech-to-Text**
- **Transformers, PyTorch, MeCab, UniDic, Pandas**

---

## 🚀 使い方

1. リポジトリをクローンし、セットアップ

```bash
git clone https://github.com/momok47/lounge.git
cd lounge
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

2. サーバーを起動

```bash
cd src/webapp
python manage.py runserver
```

3. Webブラウザでアクセス

- アプリ本体: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- 管理画面: [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin)

---

## 📝 更新履歴

- 2024/07: UIを全面リニューアル。色・文言・アイコン・ボタン配置を刷新。
- 2024/07: 分析結果アイコンを棒グラフに変更。注意文デザイン改善。
- 2024/07: ナビゲーション・ヒーロー・カードタイトルの太字化。

---

## 👥 チーム・サポート

- **[@momok47](https://github.com/momok47)** - メイン開発者
- **CA Tech Loungeメンターさん** - プロジェクトサポート

問題や質問がある場合は、[Issues](https://github.com/momok47/lounge/issues)を作成してください。