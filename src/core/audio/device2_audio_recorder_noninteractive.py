import pyaudio
import wave
import time
import io
import threading
from threading import Event

def record_dual_audio_noninteractive(device_index_a, device_index_b, duration_seconds=60):
    """
    2つのデバイスで同時に録音する関数（非対話型）
    
    Args:
        device_index_a (int): デバイス1のインデックス
        device_index_b (int): デバイス2のインデックス
        duration_seconds (int): 録音時間（秒）
        
    Returns:
        tuple: (wav_data_a, filename_a, wav_data_b, filename_b)
    """
    # 録音パラメータ
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100

    p = pyaudio.PyAudio()
    
    # 録音データを格納するリスト
    frames_a = []
    frames_b = []
    
    # スレッド同期用イベント
    recording_event = Event()
    thread_finished_event_a = Event()
    thread_finished_event_b = Event()

    def record_stream(frames, event, device_index, stream_finished_event):
        """指定されたデバイスから音声を録音"""
        stream = None
        try:
            print(f"デバイス {device_index} で録音を開始")
            
            stream = p.open(format=sample_format,
                          channels=channels,
                          rate=sample_rate,
                          frames_per_buffer=chunk,
                          input=True,
                          input_device_index=device_index)
            
            while event.is_set():
                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"読み込みエラー (デバイス {device_index}): {e}")
                    break
                    
            print(f"デバイス {device_index} の録音完了")
            
        except Exception as e:
            print(f"デバイス {device_index} のストリームオープン中にエラーが発生しました: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            stream_finished_event.set()  # このスレッドの処理が終了したことを通知

    def manage_recording_auto(rec_event, duration):
        """自動録音管理（指定秒数録音）"""
        print(f"\n🎙️ 録音開始（{duration}秒間）...")
        rec_event.set()  # 録音開始
        
        # 指定時間待機
        time.sleep(duration)
        
        print(f"\n⏹️ 録音終了（{duration}秒経過）")
        rec_event.clear()  # 録音終了

    # 各デバイスの録音スレッドを開始
    record_thread_a = threading.Thread(target=record_stream, 
                                       args=(frames_a, recording_event, device_index_a, thread_finished_event_a))
    record_thread_b = threading.Thread(target=record_stream, 
                                       args=(frames_b, recording_event, device_index_b, thread_finished_event_b))

    record_thread_a.start()
    record_thread_b.start()

    # 自動録音管理スレッドを開始
    manage_thread = threading.Thread(target=manage_recording_auto, args=(recording_event, duration_seconds))
    manage_thread.start()
    
    # 録音管理スレッドの終了を待つ
    manage_thread.join()

    # 両方の録音スレッドが終了するのを待つ
    thread_finished_event_a.wait()
    thread_finished_event_b.wait()

    print("✅ 両方のデバイスでの録音が完了")
    p.terminate()

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
        print(f"✅ ファイル作成: {filename_a}")

        # デバイスBのWAVデータを作成
        with wave.open(wav_buffer_b, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames_b))
        filename_b = f"recording_device_{device_index_b}_{int(time.time())}.wav"
        print(f"✅ ファイル作成: {filename_b}")

        return wav_buffer_a.getvalue(), filename_a, wav_buffer_b.getvalue(), filename_b

    except Exception as e:
        print(f"❌ WAVファイル作成エラー: {e}")
        return None, None, None, None


if __name__ == "__main__":
    # テスト実行
    print("=== 非対話型録音テスト ===")
    print("利用可能なデバイス:")
    
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            print(f"  {i}: {info.get('name')}")
    p.terminate()
    
    # テスト録音（デバイス0と2で5秒間）
    wav_a, file_a, wav_b, file_b = record_dual_audio_noninteractive(0, 2, duration_seconds=5)
    
    if wav_a and wav_b:
        print(f"✅ テスト成功: {len(wav_a)} bytes, {len(wav_b)} bytes")
    else:
        print("❌ テスト失敗") 