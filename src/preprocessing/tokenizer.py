"""
tokenizer.py
DialTalk 전처리 파이프라인 - 2단계: 토큰화(tokenize)

형태소 분석기 선택: KoNLPy Okt (Open Korean Text)
  - Mecab-ko는 Windows 설치가 까다로움 (별도 빌드/사전 설치 필요)
  - Okt는 `pip install konlpy` + Java(JDK)만 있으면 바로 동작 -> 해커톤 일정에 적합
  - 필요시 나중에 Mecab으로 교체 가능하도록 tokenize() 시그니처만 고정해둔다

★ 중요 - norm/stem 옵션을 반드시 꺼둘 것:
  Okt는 morphs() 호출 시 norm=True로 두면 "그니까"->"그러니까"처럼
  구어체/방언을 자기 나름대로 표준화해버린다. 이러면 모델이 배워야 할
  방언->표준어 매핑을 토큰화 단계에서 미리 지워버리는 셈이라 절대 켜면 안 된다.
  stem=True도 마찬가지로 어미 원형을 복원해버려 방언 특유의 어미
  (-이소, -예 등)가 사라질 수 있으므로 끈다.

역할 (설계문서 3.2~3.3 참고):
  cleaned_text (text_cleaner.clean_text 결과) -> token 리스트
"""

try:
    from konlpy.tag import Okt  # type: ignore[import]
except Exception:  # pragma: no cover - fallback for environments without Konlpy
    Okt = None

_okt = Okt() if Okt is not None else None


def tokenize(cleaned_text: str) -> list[str]:
    """
    형태소 단위로 토큰화한다.

    Args:
        cleaned_text: text_cleaner.clean_text()를 거친 정제된 텍스트

    Returns:
        형태소 토큰 리스트. 입력이 비어 있으면 빈 리스트 반환.
    """
    if not cleaned_text:
        return []
    if _okt is None:
        return cleaned_text.split()
    return _okt.morphs(cleaned_text, norm=False, stem=False)


def tokenize_batch(cleaned_texts: list[str]) -> list[list[str]]:
    """여러 문장을 한 번에 토큰화 (build_dataset.py에서 코퍼스 전체 처리용)"""
    return [tokenize(t) for t in cleaned_texts]


if __name__ == "__main__":
    from text_cleaner import clean_text

    samples = [
        "단디 보소!!!  오천 원 이라예~~~~",
        "그거 매끼놔라...ㅋㅋㅋㅋㅋㅋㅋ  @@ 5000  원 인데",
        "아아아아 진짜 마음에 드네예    ",
        "",
    ]
    for s in samples:
        cleaned = clean_text(s)
        tokens = tokenize(cleaned)
        print(f"원본   : {s!r}")
        print(f"정제   : {cleaned!r}")
        print(f"토큰   : {tokens}")
        print()