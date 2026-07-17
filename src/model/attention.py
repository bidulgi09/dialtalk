"""
model/attention.py
DialTalk 모델 - Bahdanau (Additive) Attention

역할 (설계문서 4.2 참고):
  디코더가 매 스텝 단어를 하나씩 생성할 때, 인코더가 만든 전체 시퀀스
  (encoder_outputs) 중 "지금 이 순간 어디를 봐야 하는지"를 학습으로 정렬(alignment)한다.

  예: "단디 보소, 오천 원 이라예" -> "자세히" 를 생성하는 순간엔 "단디"쪽에,
      "오천 원" 을 생성하는 순간엔 "오천 원"쪽에 attention_weights가 몰리도록 학습됨.

입력:
  decoder_hidden: (batch_size, decoder_units)      - 디코더의 현재(직전) 은닉 상태
  encoder_outputs: (batch_size, T_in, encoder_dim)  - 인코더 전체 시퀀스 출력

출력:
  context_vector: (batch_size, encoder_dim)   - encoder_outputs를 attention_weights로
                                                  가중합한 벡터. 디코더 입력에 concat됨.
  attention_weights: (batch_size, T_in, 1)    - 시각화/디버깅 및 confidence 보조 신호로 활용 가능
                                                  (설계문서 3.4 B안 참고)
"""

import tensorflow as tf # type: ignore[import]
from tensorflow.keras import layers # type: ignore[import]


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Args (생성자):
        units: attention score 계산용 내부 Dense 레이어의 차원

    score(s_t, h_i) = V^T * tanh(W1*h_i + W2*s_t)
    attention_weights = softmax(score)  (T_in 축으로)
    context_vector = sum(attention_weights * encoder_outputs)
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units, name="attn_W1")  # encoder_outputs에 적용
        self.W2 = layers.Dense(units, name="attn_W2")  # decoder_hidden에 적용
        self.V = layers.Dense(1, name="attn_V")        # score를 스칼라로 압축

    def call(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (B, decoder_units) -> (B, 1, decoder_units)로 확장해서
        # encoder_outputs의 시간축(T_in)과 브로드캐스팅 덧셈이 되도록 맞춘다
        decoder_hidden_expanded = tf.expand_dims(decoder_hidden, axis=1)  # (B, 1, decoder_units)

        score = self.V(
            tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden_expanded))
        )  # (B, T_in, 1)

        attention_weights = tf.nn.softmax(score, axis=1)  # (B, T_in, 1), T_in 축으로 정규화

        context_vector = attention_weights * encoder_outputs  # (B, T_in, encoder_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (B, encoder_dim)

        return context_vector, attention_weights


if __name__ == "__main__":
    # Encoder 출력 shape (4, 10, 512)를 가정한 더미 테스트
    BATCH_SIZE = 4
    T_IN = 10
    ENCODER_DIM = 512   # encoder.py의 gru_units(256) * 2 (양방향)
    DECODER_UNITS = 512

    dummy_encoder_outputs = tf.random.normal((BATCH_SIZE, T_IN, ENCODER_DIM))
    dummy_decoder_hidden = tf.random.normal((BATCH_SIZE, DECODER_UNITS))

    attention = BahdanauAttention(units=256)
    context_vector, attention_weights = attention(dummy_decoder_hidden, dummy_encoder_outputs)

    print("context_vector shape    :", context_vector.shape)     # 기대: (4, 512)
    print("attention_weights shape :", attention_weights.shape)  # 기대: (4, 10, 1)

    # attention_weights가 T_in 축에서 합이 1이 되는지 확인 (softmax 검증)
    weight_sums = tf.reduce_sum(attention_weights, axis=1)
    print("weight sums (모두 1에 가까워야 함):", weight_sums.numpy().flatten())