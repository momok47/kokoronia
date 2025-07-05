## CA Tech Loungeでkokoroniaをローカルサービス化

loungeディレクトリ構造


lounge/
├── src/                    # 📦 コア機能
│   ├── core/
│   │   ├── audio/         # 🎤 音声録音
│   │   │   ├── device1_audio_recorder.py
│   │   │   └── device2_audio_recorder.py
│   │   ├── gcs/           # ☁️ Google Cloud Storage
│   │   │   ├── gcs_uploader.py
│   │   │   └── transcribe_audio_from_gcs.py
│   │   └── analysis/      # 🤖 AI分析
│   │       ├── interests_extraction.py
│   │       └── zero_shot_learning.py
│   └── webapp/            # 🌐 Django Web アプリ
│       ├── manage.py
│       ├── accounts/
│       ├── project/
│       └── templates/
├── scripts/               # 🚀 実行スクリプト
│   └── main.py
├── docs/                  # 📖 ドキュメント
│   └── ミーティング/
├── tests/                 # 🧪 テストファイル
├── mock_data/             # 🎭 モックデータ
├── others/                # 🔧 その他のツール
└── credentials/           # 🔐 認証情報
