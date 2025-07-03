# Google Cloud Storage Uploader
# Original size: 3043 bytes
# GCSへのファイルアップロード機能

from google.cloud import storage
import os

class GCSUploader:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        
    def upload_file(self, source_file_name, destination_blob_name):
        '''ファイルをGCSにアップロード'''
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f'File {source_file_name} uploaded to {destination_blob_name}.')
        return blob.public_url

