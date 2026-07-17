import os
import json
from google import genai
from google.genai import types

from src.stt import stt

# 1. API 키 설정 (신버전은 Client 객체 생성 시 api_key를 전달합니다)
client = genai.Client(api_key="AIzaSyABmWbLJl213TiE4EBbkbJsE4lHornwPrE")

# 2. 프롬프트를 변수로 저장
TRANSLATION_PROMPT_TEMPLATE = """
당신은 방언(사투리)을 표준어로 변환하고, 이를 다시 영어로 번역하는 전문 번역기입니다.
주어진 방언 토큰 리스트의 문맥을 파악하여 자연스러운 하나의 표준어 문장(또는 문단)으로 재구성하고, 이를 영어로 번역해주세요.

입력된 방언 토큰 리스트:
{tokens}

출력은 반드시 다음 JSON 스키마를 따라야 합니다:
{{
    "standard_korean": "표준어로 변환된 자연스러운 문장",
    "translated_text": "영어로 번역된 문장"
}}
"""

def translate_dialect_tokens(tokens: list[str]) -> dict:
    """
    토큰화된 방언 리스트를 받아 표준어와 영어 번역 결과를 포함하는 딕셔너리를 반환합니다.
    """
    if not tokens:
        return {"standard_korean": "", "translated_text": ""}

    # 리스트를 프롬프트에 삽입하기 좋게 문자열로 변환
    tokens_str = ", ".join(tokens)
    
    # 프롬프트 완성
    prompt = TRANSLATION_PROMPT_TEMPLATE.format(tokens=tokens_str)
    
    try:
        # 3. 신버전 API 호출 방식 (JSON 출력 강제 설정 추가)
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # 디버깅용: API 응답 원본 출력
        print(f"API 원본 응답: {response.text}\n")
        
        # 반환된 JSON 문자열을 파이썬 딕셔너리로 변환
        result_dict = json.loads(response.text)
        return result_dict
        
    except Exception as e:
        print(f"API 호출 또는 JSON 파싱 중 오류 발생: {e}")
        return None

# --- 실행 예시 ---
if __name__ == "__main__":
    # STT 후 토큰화되었다고 가정한 방언 리스트 예시
    sample_tokens = stt.predict_from_audio(
        audio_path="test_audio.mp3"
    )
    print(f"입력 토큰: {sample_tokens}\n")
    
    result = translate_dialect_tokens(sample_tokens)
    
    if result:
        print("--- 변환 결과 ---")
        print(f"표준어: {result.get('standard_korean')}")
        print(f"영어: {result.get('translated_text')}")
        print("\n--- 전체 JSON 데이터 ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))