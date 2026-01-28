# コマンド：python gcs_audio_player.py ファイル名.mp3
from google.cloud import storage
import webbrowser
import tempfile
import os
import argparse
from datetime import datetime, timedelta, UTC

# バケット名を定数として定義
BUCKET_NAME = "kokoronia"
AUDIO_DIR = "media/audio"
# サービスアカウントの秘密鍵ファイルのパス（環境変数で指定）
SERVICE_ACCOUNT_KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

def play_gcs_audio(
    bucket_name,
    blob_name,
    auto_cleanup=True
):
    """
    GCSの音声ファイルをブラウザで再生する
    入力:
        bucket_name: GCSバケット名
        blob_name: GCS上の音声ファイルパス
        auto_cleanup: 一時HTMLファイルを自動削除するかどうか
    """
    print(f"GCSの音声ファイル{blob_name.split("/")[-1]}をブラウザで再生します")
    try:
        if not SERVICE_ACCOUNT_KEY:
            raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS が設定されていません。")

        # ストレージクライアント(GCSと通信するためのインターフェース)作成
        client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # ファイルの存在確認
        if not blob.exists():
            raise FileNotFoundError(f"file not found in GCS: {blob_name}")

        # 署名付きURL(1時間だけ有効なURL)を作成
        public_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.now(UTC) + timedelta(hours=1),
            method="GET"
        )
        # print(f"public URL: {public_url}")

        # 一時的なHTMLファイルを作成
        html_content = f"""
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8">
            <title>Audio Preview</title>
            <style>
              body {{ font-family: Arial, sans-serif; margin: 20px; }}
              .audio-container {{ margin: 20px 0; }}
              audio {{ width: 100%; max-width: 500px; }}
            </style>
          </head>
          <body>
            <h2>ファイル名: {os.path.basename(blob_name)}</h2>
            <div class="audio-container">
              <audio controls autoplay>
                <source src="{public_url}" type="audio/mpeg">
                このブラウザは audio タグをサポートしていません。
              </audio>
            </div>
          </body>
        </html>
        """

        # 一時HTMLファイル作成 & ブラウザで開く(2回目以降もサイトが開く)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
            f.write(html_content)
            temp_html_path = f.name

        # 音声ファイルをブラウザで開く
        webbrowser.open(f"file://{temp_html_path}", new=2)

        # 自動クリーンアップ
        if auto_cleanup:
            try:
                os.unlink(temp_html_path)
            except Exception as e:
                print(f"failed to delete temporary file: {e}")

        return public_url

    except Exception as e:
        raise Exception(f"error occurred while processing audio file: {str(e)}")

if __name__ == "__main__":
    # コマンドライン引数の設定(--helpってするとこのメッセージが出る)
    parser = argparse.ArgumentParser(description='GCSの音声ファイルを再生できます')
    parser.add_argument('filename', help='再生する音声ファイル名（例: sample.mp3）')
    args = parser.parse_args()

    try:
        # ファイルパスを組み立てる
        file_path = f"{AUDIO_DIR}/{args.filename}"
        play_gcs_audio(
            bucket_name=BUCKET_NAME,
            blob_name=file_path,
            auto_cleanup=False  # 一時ファイルを保持する
        )
    except Exception as e:
        print(f"error: {e}")