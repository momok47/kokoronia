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