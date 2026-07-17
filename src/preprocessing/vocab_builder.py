"""
vocab_builder.py
DialTalk 전처리 파이프라인 - 3단계: Vocab 구축 (학습 시에만 1회 실행)

역할 (설계문서 3.2, 3.4 참고):
  학습 코퍼스 전체의 토큰 리스트 -> {token: id} vocab 딕셔너리 생성.
  방언용 vocab과 표준어용 vocab은 반드시 "따로" 만들어야 한다
  (인코더/디코더가 서로 다른 언어 공간을 학습하므로 vocab을 공유하면 안 됨).

★ 중요: 이 모듈은 학습 파이프라인(build_dataset.py)에서만 호출한다.
  실시간 추론(pipeline.py)에서는 절대 build_vocab()을 다시 호출하지 않고,
  학습 때 저장해둔 vocab.json 파일을 load_vocab()으로 불러와 재사용해야 한다.
  추론 때마다 vocab이 달라지면 모델의 id 매핑이 어긋나서 완전히 오작동한다.
"""

import json
from collections import Counter

# 특수 토큰 - 순서가 곧 id이므로 한번 학습을 시작하면 절대 순서를 바꾸지 말 것
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3


def build_vocab(token_lists: list[list[str]], min_freq: int = 2) -> dict[str, int]:
    """
    토큰 리스트들(문장 단위)로부터 vocab을 생성한다.

    Args:
        token_lists: tokenize()를 거친 문장별 토큰 리스트의 리스트
                     예: [["단디", "보소"], ["오천", "원", "이라예"]]
        min_freq: 이 빈도 미만으로 등장한 토큰은 vocab에서 제외 (추론 시 <UNK> 처리됨)

    Returns:
        {token: id} 딕셔너리. 0~3번 id는 특수 토큰이 고정으로 차지한다.
    """
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    next_id = len(SPECIAL_TOKENS)

    # 빈도순으로 id를 배정 (필수는 아니지만 vocab 파일만 봐도 자주 쓰는 토큰을 알 수 있어 디버깅에 유용)
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = next_id
        next_id += 1

    return vocab


def save_vocab(vocab: dict[str, int], path: str) -> None:
    """vocab을 JSON 파일로 저장 (학습 후 1회 저장, 이후 추론에서 재사용)"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> dict[str, int]:
    """저장된 vocab.json을 로드 (추론 파이프라인 pipeline.py에서 사용)"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    """{token: id} -> {id: token} 역매핑 (decode 시 필요, encoder.py에서 사용 예정)"""
    return {idx: tok for tok, idx in vocab.items()}


if __name__ == "__main__":
    sample_sentences = [
        ["단디", "보소", ","],
        ["오천", "원", "이라예", "."],
        ["단디", "하다"],  # "단디"를 2번 등장시켜 min_freq=2 통과 확인
    ]
    vocab = build_vocab(sample_sentences, min_freq=2)

    print("생성된 vocab:")
    for tok, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"  {idx}: {tok!r}")

    print(f"\nvocab 크기: {len(vocab)}")
    print(f"'단디' id (2회 등장, 포함되어야 함): {vocab.get('단디')}")
    print(f"'보소' 포함 여부 (1회 등장, min_freq=2 미만이라 제외되어야 함): {'보소' in vocab}")

    id2token = invert_vocab(vocab)
    print(f"\n역매핑 확인: id {vocab['단디']} -> {id2token[vocab['단디']]!r}")