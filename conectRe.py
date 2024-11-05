import pyaudio
import json
import time
from vosk import Model, KaldiRecognizer
import socket

# 音声認識の設定
def record_audio(sr=16000, framesize=4096):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sr,
                        input=True, frames_per_buffer=framesize)
    return stream

# ソケット通信を行う関数
def send_to_ue5(message):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("localhost", 12345))  # IPとポートを指定
        sock.sendall(message.encode('utf-8'))
        sock.close()
    except socket.error as e:
        print(f"ソケット通信エラー: {e}")

# 音声認識の処理ループ
def recognize_speech_vosk(stream, model, sr=16000):
    rec = KaldiRecognizer(model, sr)

    while True:
        data = stream.read(4096)
        if len(data) == 0:
            break

        # 音声認識
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get('text', '')

            if text:
                print(f"認識結果: {text}")
                classify_text(text)

# テキストを分類し、ソケットでUE5に送信
def classify_text(text):
    if '混雑' in text:
        print('歩行者の混雑状況 - Detected')
        send_to_ue5("crowd")
    elif '狭い' in text:
        print('自転車通行帯の欠如・整備不足および車道の狭さ - Detected')
        send_to_ue5("narrow")
    elif 'ハードル' in text:
        print('障害物のある通行帯 - Detected')
        send_to_ue5("hurdle")
    else:
        print('不明な入力')

if __name__ == "__main__":
    # モデルのロード
    print("モデルをロードしています...")
    start_load_time = time.perf_counter()
    model = Model("C:\\vosk_model\\vosk-model-small-ja-0.22")
    end_load_time = time.perf_counter()
    load_elapsed_time = end_load_time - start_load_time
    print(f"モデルのロード時間: {load_elapsed_time:.4f} 秒")

    # 音声入力のストリームを開始
    print("音声入力待機中...")
    stream = record_audio()

    # 音声認識のループを開始
    recognize_speech_vosk(stream, model)
