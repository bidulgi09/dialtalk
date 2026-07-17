"""
model/encoder.py
DialTalk 모델 - Encoder

역할 (설계문서 4.2 참고):
  방언 input_ids -> encoder_outputs (Attention이 참조할 전체 시퀀스)
                  -> encoder_final_state (Decoder 초기 상태 + ConfidenceHead 입력)

구조: Embedding -> Bidirectional GRU
  - Bidirectional을 쓰는 이유: 방언 어미 변형(-이소, -예 등)이 문장 뒤쪽에 몰리는
    경향이 있어, 양방향으로 문맥을 봐야 어미 패턴을 더 잘 잡아낼 수 있다.
  - mask_zero=True: PAD_ID=0 (vocab_builder.py의 PAD_ID와 반드시 일치해야 함)이라
    Embedding이 패딩 위치를 자동으로 마스킹해서 GRU 계산에서 무시하게 한다.
"""

import tensorflow as tf  # type: ignore[import]
from tensorflow.keras import layers # type: ignore[import]


class Encoder(tf.keras.layers.Layer):
    """
    Args (생성자):
        vocab_size: 방언 vocab 크기 (dialect_vocab.json 로드 후 len(vocab))
        embedding_dim: 임베딩 차원 (기본 128)
        gru_units: 단방향 GRU의 unit 수 (기본 256, 양방향이라 실제 출력은 512차원)

    call() 입력:
        input_ids: (batch_size, max_input_len) 정수 텐서

    call() 출력:
        encoder_outputs: (batch_size, max_input_len, gru_units*2)
            -> Attention 모듈이 여기서 매 디코딩 스텝마다 어디를 볼지 정렬(alignment)을 계산
        encoder_final_state: (batch_size, gru_units*2)
            -> Decoder GRU의 initial_state로 사용
            -> ConfidenceHead의 입력으로도 사용 (설계문서 4.2)
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128, gru_units: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="encoder_embedding",
        )
        self.bi_gru = layers.Bidirectional(
            layers.GRU(
                gru_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            ),
            merge_mode="concat",
            name="encoder_bi_gru",
        )

    def call(self, input_ids, training: bool = False):
        x = self.embedding(input_ids)  # (B, T_in, embedding_dim)

        # Bidirectional(GRU)의 반환값 순서: outputs, forward_state, backward_state
        encoder_outputs, state_fwd, state_bwd = self.bi_gru(x, training=training)

        # 정방향/역방향 최종 은닉 상태를 이어붙여서 하나의 벡터로 만든다
        encoder_final_state = tf.concat([state_fwd, state_bwd], axis=-1)  # (B, gru_units*2)

        return encoder_outputs, encoder_final_state


if __name__ == "__main__":
    # 더미 입력으로 shape만 빠르게 확인 (로컬에서 실행해서 shape이 주석과 일치하는지 확인해주세요)
    VOCAB_SIZE = 500
    MAX_INPUT_LEN = 10
    BATCH_SIZE = 4

    encoder = Encoder(vocab_size=VOCAB_SIZE, embedding_dim=128, gru_units=256)

    dummy_input = tf.random.uniform(
        (BATCH_SIZE, MAX_INPUT_LEN), minval=1, maxval=VOCAB_SIZE, dtype=tf.int32
    )
    outputs, final_state = encoder(dummy_input)

    print("encoder_outputs shape    :", outputs.shape)      # 기대: (4, 10, 512)
    print("encoder_final_state shape:", final_state.shape)  # 기대: (4, 512)