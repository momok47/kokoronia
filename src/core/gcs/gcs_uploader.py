import os
from google.cloud import storage

def upload_to_gcs(bucket_name, data_content, destination_blob_name, content_type=None):
    """
    Args:
        bucket_name (str): アップロード先のGCSバケット名。
        data_content (bytes or str): アップロードするデータ（バイナリデータまたは文字列）。
        destination_blob_name (str): GCSバケット内での目的のファイル名（パスを含む）。
        content_type (str, optional): アップロードするデータのMIMEタイプ。指定しない場合は自動検出。
                                      wavファイル 'media/audio/', jsonファイル 'media/transcriptions/'

    Returns:
        str: アップロード成功： GCSのURI
             アップロード失敗： None
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # データ型に応じてアップロード方法を使い分け
        if isinstance(data_content, bytes):
            blob.upload_from_string(data_content, content_type=content_type)
        elif isinstance(data_content, str): # JSON文字列などの場合
            blob.upload_from_string(data_content, content_type=content_type)
        else:
            print("エラー: data_contentはbytesまたはstrである必要があります。")
            return None
        
        print(f"GCSにアップロード完了： gs://{bucket_name}/{destination_blob_name}")
        return f"gs://{bucket_name}/{destination_blob_name}"
    except Exception as e:
        print(f"GCSへのアップロード中にエラーが発生しました: {e}")
        return None

if __name__ == "__main__":
    # このスクリプトを直接実行した場合のテスト_WAVファイルのアップロードを想定
    bucket_name = "kokoronia"
    recordings_dir = "recordings"

    if not os.path.exists(recordings_dir):
        print(f"エラー: {recordings_dir}ディレクトリが見つかりません")
        exit()

    wav_files = [f for f in os.path.listdir(recordings_dir) if f.endswith('.wav')]
    if not wav_files:
        print(f"エラー: {recordings_dir}ディレクトリにWAVファイルが見つかりません")
        exit()

    latest_file = sorted(wav_files)[-1]
    local_file_path = os.path.join(recordings_dir, latest_file)

    try:
        # 絶対パスに変換
        if not os.path.isabs(local_file_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            local_file_path = os.path.join(current_dir, local_file_path)
        
        with open(local_file_path, 'rb') as f:
            wav_data = f.read()
        
        destination_blob_name = f"media/audio/{latest_file}"
        gcs_uri = upload_to_gcs(bucket_name, wav_data, destination_blob_name, content_type="audio/wav")
        
        if gcs_uri:
            print(f"テスト: 最新のWAVファイルをアップロードしました: {gcs_uri}")
        else:
            print("テスト: WAVファイルのアップロードに失敗しました。")

    except Exception as e:
        print(f"テスト実行中にエラーが発生しました: {e}")