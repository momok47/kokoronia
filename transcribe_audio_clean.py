# Audio Transcription from GCS
# Original size: 5576 bytes  
# Google Cloud Speechを使った音声転写機能

from google.cloud import speech
import io

class AudioTranscriber:
    def __init__(self):
        self.client = speech.SpeechClient()
        
    def transcribe_gcs_audio(self, gcs_uri):
        '''GCS上の音声ファイルを転写'''
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='ja-JP',
        )
        
        response = self.client.recognize(config=config, audio=audio)
        
        transcripts = []
        for result in response.results:
            transcripts.append(result.alternatives[0].transcript)
            
        return ' '.join(transcripts)

