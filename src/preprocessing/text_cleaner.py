"""
text_cleaner.py
DialTalk 전처리 파이프라인 - 1단계: 텍스트 정제(clean) 및 정규화(normalize)

역할 (설계문서 3.2 참고):
  1. 정제(clean): 특수문자/중복 공백 제거, 반복 음절 축약
  2. 정규화(normalize): 숫자·단위 표기 통일

★ 중요: 학습 데이터 구축(build_dataset.py)과 실시간 추론(pipeline.py) 양쪽에서
반드시 이 모듈의 clean_text() 함수를 그대로 호출해야 한다.
전처리 로직이 서로 달라지면 학습된 모델이 추론 시 낯선 입력을 받게 되어
성능이 급격히 떨어진다 (설계문서 3.4 참고).
"""

import re
import unicodedata

# 허용할 문자 집합: 한글(완성형+자모), 영문, 숫자, 기본 문장부호, 공백
_ALLOWED_PUNCT = ".,?!~()'\""
_ALLOWED_PATTERN = re.compile(
    r"[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9\s" + re.escape(_ALLOWED_PUNCT) + r"]"
)

# 3회 이상 반복되는 문자를 2회로 축약 (예: "아아아아아" -> "아아")
# 숫자는 제외한다 - "5000"의 "000"처럼 정상적인 숫자 반복까지 축약되면 값이 깨짐
_REPEAT_CHAR_PATTERN = re.compile(r"([^\d])\1{2,}")

# 2회 이상 연속 공백을 1칸으로
_MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")

# 숫자 + '원' 단위 사이 불필요한 공백 제거 (예: "5000 원" -> "5000원")
_NUMBER_UNIT_PATTERN = re.compile(r"(\d+)\s*원")


def normalize_unicode(text: str) -> str:
    """유니코드 정규화(NFC) - 자모 분리 등으로 인한 동일 문자 불일치 방지"""
    return unicodedata.normalize("NFC", text)


def remove_noise_characters(text: str) -> str:
    """한글/영문/숫자/기본 문장부호를 제외한 잡음 문자 제거 (이모지, 특수기호 등)"""
    return _ALLOWED_PATTERN.sub("", text)


def collapse_repeated_chars(text: str) -> str:
    """3회 이상 반복되는 문자를 2회로 축약 (구어체 늘임 표현 정리, 'ㅋㅋㅋㅋㅋ' -> 'ㅋㅋ')"""
    return _REPEAT_CHAR_PATTERN.sub(r"\1\1", text)


def normalize_numbers(text: str) -> str:
    """숫자+단위 표기 사이 공백 통일 (STT 결과의 띄어쓰기 편차 정리)"""
    return _NUMBER_UNIT_PATTERN.sub(r"\1원", text)


def collapse_whitespace(text: str) -> str:
    """중복 공백을 1칸으로 통일하고 앞뒤 공백 제거"""
    return _MULTI_SPACE_PATTERN.sub(" ", text).strip()


def clean_text(raw_text: str) -> str:
    """
    전체 정제 파이프라인 (설계문서 3.3의 clean_text 시그니처 구현)

    호출 순서가 중요하다:
      유니코드 정규화 -> 잡음 문자 제거 -> 반복 문자 축약 -> 숫자 정규화 -> 공백 정리

    Args:
        raw_text: Whisper STT 원문 또는 학습용 raw 텍스트 한 줄

    Returns:
        정제 및 정규화가 완료된 텍스트 (다음 단계인 tokenize()로 전달됨)
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = normalize_unicode(raw_text)
    text = remove_noise_characters(text)
    text = collapse_repeated_chars(text)
    text = normalize_numbers(text)
    text = collapse_whitespace(text)
    return text


if __name__ == "__main__":
    # 간단한 동작 확인용 샘플 (실제 유닛테스트는 tests/ 디렉토리에 별도 구성 권장)
    samples = [
        "단디 보소!!!  오천 원 이라예~~~~",
        "그거 매끼놔라...ㅋㅋㅋㅋㅋㅋㅋ  @@ 5000  원 인데",
        "아아아아 진짜 마음에 드네예    ",
        "",
    ]
    for s in samples:
        print(f"원본: {s!r}")
        print(f"정제: {clean_text(s)!r}")
        print()