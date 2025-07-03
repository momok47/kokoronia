import pyaudio
import wave
import time
import io
import threading
from threading import Event

def record_dual_audio(device_index_a, device_index_b):
    """
    Args:
        device_index_a (int): 最初の録音デバイスのインデックス。
        device_index_b (int): 2番目の録音デバイスのインデックス。

    Returns:
        tuple: (wav_buffer_a.getvalue(), filename_a, wav_buffer_b.getvalue(), filename_b)
               またはエラーが発生した場合は (None, None, None, None)
    """
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100

    p = pyaudio.PyAudio()

    # 録音データを格納するリスト (デバイスごとに用意)
    frames_a = []
    frames_b = []

    recording_event = Event()
    
    # 録音スレッドが終了したことを通知するイベント
    thread_finished_event_a = Event()
    thread_finished_event_b = Event()

    def record_stream(frames_list, event, device_index, stream_finished_event):
        try:
            stream = p.open(format=sample_format,
                           channels=channels,
                           rate=sample_rate,
                           frames_per_buffer=chunk,
                           input=True,
                           input_device_index=device_index)
            
            # manage_recording スレッドからの開始イベントを待機
            event.wait()

            # print(f"デバイス {device_index} で録音開始...")
            while event.is_set():
                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames_list.append(data)
                except Exception as e:
                    print(f"デバイス {device_index} で録音中にエラーが発生しました: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            # print(f"デバイス {device_index} の録音を停止しました。")
        except Exception as e:
            print(f"デバイス {device_index} のストリームオープン中にエラーが発生しました: {e}")
        finally:
            stream_finished_event.set() # このスレッドの処理が終了したことを通知

    def manage_recording(rec_event):
        input("\nEnterを押して会話を開始してね^-^")
        rec_event.set()  # 2端末録音開始イベントを設定
        
        input("録音中だよ，会話が終わったらEnterを押してね^-^")
        rec_event.clear() # 2端末録音停止イベントを設定
    
    # 各デバイスの録音スレッドを開始
    record_thread_a = threading.Thread(target=record_stream, 
                                       args=(frames_a, recording_event, device_index_a, thread_finished_event_a))
    record_thread_b = threading.Thread(target=record_stream, 
                                       args=(frames_b, recording_event, device_index_b, thread_finished_event_b))

    record_thread_a.start()
    record_thread_b.start()

    # 録音管理スレッドを開始
    manage_thread = threading.Thread(target=manage_recording, args=(recording_event,))
    manage_thread.start()
    
    # 録音管理スレッドの終了を待つ
    manage_thread.join()

    # 両方の録音スレッドが終了するのを待つ
    thread_finished_event_a.wait()
    thread_finished_event_b.wait()

    print("両方のデバイスでの録音が完了したよ")
    p.terminate() #

    wav_buffer_a = io.BytesIO()
    wav_buffer_b = io.BytesIO()
    filename_a = None
    filename_b = None

    try:
        # デバイスAのWAVデータを作成
        with wave.open(wav_buffer_a, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames_a))
        filename_a = f"recording_device_{device_index_a}_{int(time.time())}.wav"
        # print(f"ファイルを作成しました： {filename_a}")

        # デバイスBのWAVデータを作成
        with wave.open(wav_buffer_b, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames_b))
        filename_b = f"recording_device_{device_index_b}_{int(time.time())}.wav"
        # print(f"ファイルを作成しました： {filename_b}")

        return wav_buffer_a.getvalue(), filename_a, wav_buffer_b.getvalue(), filename_b

    except Exception as e:
        print(f"WAVファイルの作成中にエラーが発生しました: {e}")
        return None, None, None, None

if __name__ == "__main__":
    # 利用可能な録音デバイスの表示と選択は main.py で行う
    print("このスクリプトは単体で実行できないよ．")
    print("利用可能な録音デバイスの表示と選択は main.py でやってね．")