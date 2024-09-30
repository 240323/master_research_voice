import pyaudio
import wave
import json
import time
from vosk import Model, KaldiRecognizer



def record_audio(duration, sr=16000, framesize=8000):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr,
                     input=True, frames_per_buffer=framesize)
    frames = []

    print("録音を開始します...")
    for _ in range(int(sr / framesize * duration)):
        data = stream.read(framesize)
        frames.append(data)
    print("録音が終了しました。")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    audio_data = b''.join(frames)
    return audio_data

def recognize_speech_vosk(audio_data, model, sr=16000):
    rec = KaldiRecognizer(model, sr)
    start_time = time.perf_counter() #実行時間計測開始
    rec.AcceptWaveform(audio_data)
    result = rec.Result()
    end_time = time.perf_counter() #実行時間計測完了
    elasped_time = end_time - start_time
    print(f"音声認識の実行時間: {elasped_time:.4f} 秒")
    result_dict = json.loads(result)
    text = result_dict.get('text', '')
    return text

def classify_text(text):
    if '混雑' in text:
        return '歩行者の混雑状況'
    elif '狭い' in text:
        return '自転車通行帯の欠如・整備不足および車道の狭さ'
    elif '障害' in text:
        return '障害物のある通行帯'
    else:
        return '不明な入力'

if __name__ == "__main__":

    print("モデルをロードしています...")
    start_load_time = time.perf_counter()
    model = Model("C:\\vosk_model\\vosk-model-small-ja-0.22")
    end_load_time = time.perf_counter()
    load_elapsed_time = end_load_time - start_load_time
    print(f"モデルのロード時間: {load_elapsed_time:.4f} 秒")

    # 録音時間を指定（秒）
    duration = 3

    # 音声を録音
    audio_data = record_audio(duration)

    # 音声を認識
    text = recognize_speech_vosk(audio_data, model)
    if text:
        print(f"認識結果: {text}")

        # テキストを分類
        category = classify_text(text)
        print(f"分類結果: {category}")
    else:
        print("音声を認識できませんでした。もう一度お試しください。")
