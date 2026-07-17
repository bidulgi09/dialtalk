# stt.py 맨 위
import os
import numpy as np # type: ignore[import]
import tensorflow as tf # type: ignore[import]
import whisper # type: ignore[import]
import json
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.tokenizer import tokenize
from src.preprocessing.encoder import encode, decode
from src.preprocessing.vocab_builder import load_vocab, invert_vocab

# 1. 전역적으로 Whisper 모델 로드 (가장 가벼운 tiny나 base 추천, 한국어에는 base나 small이 잘 맞습니다)
print("Whisper 모델을 로드하는 중...")
whisper_model = whisper.load_model("base") 

# 2. 방언 번역을 위한 사용자 정의 Keras 모델 및 vocab 로드 함수
def load_translation_assets(model_path: str, dialect_vocab_path: str, standard_vocab_path: str):
    """
    저장된 Keras 모델과 방언/표준어 vocab을 로드합니다.
    """
    from src.model.dialtalk_model import DialTalkModel
    
    print("방언 번역 모델 로드 중...")
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={"DialTalkModel": DialTalkModel}
    )
    
    # 3. 프로젝트의 고유한 vocab 로딩 로직 사용
    dialect_vocab = load_vocab(dialect_vocab_path)
    standard_vocab = load_vocab(standard_vocab_path)
        
    return model, dialect_vocab, standard_vocab

# 3. 메인 추론 함수
def predict_from_audio(
    audio_path: str, 
) -> list:
    """
    mp3 등 오디오 경로를 받아 Whisper로 텍스트화 한 뒤, 
    방언 번역 모델로 결과를 예측하여 원본 텍스트와 표준어 텍스트를 반환합니다.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

    # --- STEP 1: Whisper STT ---
    print(f"[{audio_path}] 음성 인식 시작...")
    result = whisper_model.transcribe(
        audio_path, 
        language="ko", 
        fp16=False,
        temperature=0.0,              # 일관된 결과를 위해 무작위성 제거 (0으로 고정)
        no_speech_threshold=0.6,       # 침묵/잡음을 음성으로 오인식하는 문턱값 조절
        condition_on_previous_text=False # 이전 인식 결과에 오염되어 영어로 반복되는 현상 방지
    )
    transcribed_text = result["text"].strip()
    print(f"👉 인식된 방언 텍스트: {transcribed_text}")

    if not transcribed_text:
        return ["음성이 인식되지 않았습니다.", ""]

    # --- STEP 2: 번역 모델 입력용 전처리 ---
    # 1. 텍스트 정제
    cleaned_text = clean_text(transcribed_text) 
    
    # 2. 토큰화
    tokens = tokenize(cleaned_text) 
    
    
    return tokens

if __name__ == "__main__":
    print("=== STT & 번역 파이프라인 실제 모델 테스트 시작 ===")

    # 1. 실제 경로 정의 (프로젝트 폴더 구조 기준)
    MODEL_PATH = "models/dialtalk_model.keras"
    DIAL_VOCAB_PATH = "models/dialect_vocab.json"
    STD_VOCAB_PATH = "models/standard_vocab.json"
    TEST_AUDIO_PATH = "test_audio.mp3"

    # 2. 실제 자원(모델 및 사전) 로드하기
    try:
        real_model, real_dialect_vocab, real_standard_vocab = load_translation_assets(
            model_path=MODEL_PATH,
            dialect_vocab_path=DIAL_VOCAB_PATH,
            standard_vocab_path=STD_VOCAB_PATH
        )
        print("✅ 실제 모델 및 vocab 로드 완료!\n")
    except Exception as e:
        print(f"❌ 실제 모델/사전 파일 로드 중 에러 발생: {e}")
        print("폴더 내에 'models/dialtalk_model.keras' 등의 파일이 실제로 있는지 확인해주세요.")
        exit(1)

    # 3. 실제 오디오 인식 및 번역 수행
    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"\n[오류] '{TEST_AUDIO_PATH}' 파일을 찾을 수 없습니다.")
        print("Whisper 연동을 테스트하려면 'dialtalk' 폴더 안에 'test_audio.mp3' 파일을 넣어주세요.")
    else:
        print("2. Whisper STT 및 번역 추론 실행...")
        try:
            final_result = predict_from_audio(
                audio_path=TEST_AUDIO_PATH
            )
            print("\n=== 최종 실행 결과 ===")
            print(f"인식된 원본 텍스트: {final_result[0]}")
            print(f"최종 표준어 번역: {final_result[1]}")
        except Exception as e:
            print(f"\n파이프라인 실행 중 에러 발생: {e}")