"""
encoder.py
DialTalk 전처리 파이프라인 - 4단계: 인코딩(encode) / 디코딩(decode)

역할 (설계문서 3.2, 3.3 참고):
  encode: 토큰 리스트 -> vocab id 시퀀스 -> 고정 길이 패딩 (모델 입력용)
  decode: id 시퀀스 -> 토큰 -> 문자열 (모델 출력을 사람이 읽는 텍스트로 복원)

이 모듈은 전처리 파이프라인의 마지막 단계이며, 학습(build_dataset.py)과
추론(pipeline.py) 양쪽에서 동일하게 사용된다.

★ add_start_end 사용 규칙:
  - 방언(input) 시퀀스: add_start_end=False  (인코더 입력이라 시작/끝 표시 불필요)
  - 표준어(target) 시퀀스: add_start_end=True (디코더가 <START>부터 생성을 시작해서
    <END>가 나오면 멈추도록 학습해야 하므로 반드시 붙여야 함)
"""

from .vocab_builder import (
    PAD_ID,
    START_ID,
    END_ID,
    UNK_ID,
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
)


def encode(
    tokens: list[str],
    vocab: dict[str, int],
    max_len: int,
    add_start_end: bool = False,
) -> list[int]:
    """
    토큰 리스트를 고정 길이의 정수 id 시퀀스로 변환한다 (post-padding).

    Args:
        tokens: tokenize()의 출력
        vocab: vocab_builder.build_vocab() 또는 load_vocab()의 결과
        max_len: 시퀀스 고정 길이 (설계문서 8장에서 데이터 분포 보고 확정 예정)
        add_start_end: True면 앞에 <START>, 뒤에 <END>를 붙인다 (표준어 타깃 전용)

    Returns:
        길이가 정확히 max_len인 정수 리스트. vocab에 없는 토큰은 <UNK> 처리.
    """
    ids = [vocab.get(tok, UNK_ID) for tok in tokens]

    if add_start_end:
        # <START>, <END>가 들어갈 두 자리를 미리 빼고 본문을 자른다
        ids = ids[: max(0, max_len - 2)]
        ids = [START_ID] + ids + [END_ID]
    else:
        ids = ids[:max_len]

    pad_len = max_len - len(ids)
    if pad_len > 0:
        ids = ids + [PAD_ID] * pad_len

    return ids


def decode(ids: list[int], id2token: dict[int, str], stop_at_end: bool = True) -> str:
    """
    정수 id 시퀀스를 사람이 읽는 문자열로 복원한다.

    Args:
        ids: 모델 출력(또는 encode() 결과) id 시퀀스
        id2token: vocab_builder.invert_vocab()의 결과 ({id: token})
        stop_at_end: True면 <END> 토큰을 만나는 즉시 그 뒤는 버린다
                     (추론 시 greedy decoding 결과에서 반드시 True로 써야
                     <END> 이후에 남은 <PAD>들이 출력에 섞이지 않는다)

    Returns:
        공백으로 이어붙인 문자열. 형태소 단위 토큰을 그대로 이어붙이므로
        완벽한 띄어쓰기는 아니며, 필요하면 후처리(띄어쓰기 교정기 등)를
        추가로 붙일 수 있다.
    """
    tokens = []
    for i in ids:
        token = id2token.get(i)
        if token is None:
            continue
        if stop_at_end and token == END_TOKEN_NAME:
            break
        if token in (PAD_TOKEN, START_TOKEN):
            continue
        tokens.append(token)
    return " ".join(tokens)


# decode()에서 END_TOKEN 문자열 비교용 (vocab_builder의 END_TOKEN을 그대로 참조)
from .vocab_builder import END_TOKEN as END_TOKEN_NAME


if __name__ == "__main__":
    from vocab_builder import build_vocab, invert_vocab

    dialect_sentences = [["단디", "보소", ","], ["오천", "원", "이라예", "."]]
    standard_sentences = [["자세히", "보세요", "."], ["오천", "원", "입니다", "."]]

    dialect_vocab = build_vocab(dialect_sentences, min_freq=1)
    standard_vocab = build_vocab(standard_sentences, min_freq=1)
    standard_id2token = invert_vocab(standard_vocab)

    MAX_LEN = 10

    # 방언 입력 인코딩 (add_start_end=False)
    input_ids = encode(dialect_sentences[0], dialect_vocab, MAX_LEN, add_start_end=False)
    print("방언 토큰   :", dialect_sentences[0])
    print("인코딩 결과 :", input_ids)

    # 표준어 타깃 인코딩 (add_start_end=True)
    target_ids = encode(standard_sentences[0], standard_vocab, MAX_LEN, add_start_end=True)
    print("\n표준어 토큰 :", standard_sentences[0])
    print("인코딩 결과 :", target_ids)

    # 디코딩 round-trip 확인
    decoded = decode(target_ids, standard_id2token)
    print("\n디코딩 결과 :", repr(decoded))