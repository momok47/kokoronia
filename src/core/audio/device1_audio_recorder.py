import pyaudio
import wave
import time
import io
import threading
from threading import Event

def record_audio():
    # 録音パラメータ
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100

    # PyAudioオブジェクトの作成
    p = pyaudio.PyAudio()
    
    # 録音データを格納するリスト
    frames = []
    recording_event = Event()

    def record(frames_list, event):
        stream = p.open(format=sample_format,
                       channels=channels,
                       rate=sample_rate,
                       frames_per_buffer=chunk,
                       input=True)
        
        while event.is_set():
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                frames_list.append(data)
            except Exception as e:
                print(f"録音中にエラーが発生しました: {e}")
                break
        
        stream.stop_stream()
        stream.close()

    def check_input(frames_list, event):
        input("Enterを押して会話を開始^-^")
        event.set()  # 録音開始
        record_thread = threading.Thread(target=record, args=(frames_list, event))
        record_thread.start()
        input("録音中です！会話が終わったらEnterを押してね")
        event.clear()  # 録音停止
        record_thread.join()

    # 録音スレッドの開始
    input_thread = threading.Thread(target=check_input, args=(frames, recording_event))
    input_thread.start()
    input_thread.join()

    print("録音が完了しました")
    p.terminate()

    # メモリ上でWAVデータを作成
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    # ファイル名を生成（タイムスタンプ付き）
    filename = f"recording_{int(time.time())}.wav"
    
    return wav_buffer.getvalue(), filename

if __name__ == "__main__":
    record_audio()