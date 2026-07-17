"""
model/decoder.py
DialTalk 모델 - Decoder (1-step)

역할 (설계문서 4.2 참고):
  표준어 토큰을 한 스텝씩 생성한다. 이 클래스의 call()은 "한 스텝"만 처리하며,
  학습 시에는 train.py에서 정답 시퀀스를 한 토큰씩 넣어주는 teacher forcing 방식으로
  MAX_OUTPUT_LEN번 반복 호출한다. 추론 시에는 pipeline.py에서 이전 스텝의 예측을
  다음 스텝 입력으로 넣는 greedy decoding 방식으로 반복 호출한다.

  (한 번에 전체 시퀀스를 생성하는 구조가 아니라 "1-step 함수"로 만든 이유:
   학습/추론에서 입력을 주는 방식이 다르기 때문 - teacher forcing vs greedy decoding.
   둘 다 이 클래스 하나를 재사용할 수 있게 설계.)

흐름 (한 스텝):
  1. decoder_input(단어 1개) -> Embedding
  2. Attention(이전 GRU 은닉 상태, encoder_outputs) -> context_vector
  3. Embedding 출력과 context_vector를 concat -> GRU 입력
  4. GRU -> 새로운 은닉 상태
  5. GRU 출력 -> Dense(softmax) -> 다음 토큰에 대한 확률 분포
"""

import tensorflow as tf # type: ignore[import]
from tensorflow.keras import layers # type: ignore[import]

try:
    from .attention import BahdanauAttention
except ImportError:  # pragma: no cover - allows running the file directly from src/model
    from attention import BahdanauAttention


class Decoder(tf.keras.layers.Layer):
    """
    Args (생성자):
        vocab_size: 표준어 vocab 크기 (standard_vocab.json 로드 후 len(vocab))
        embedding_dim: 임베딩 차원 (기본 128, encoder와 동일하게 맞춰도 되고 달라도 무방)
        decoder_units: 디코더 GRU의 unit 수 (기본 512, encoder_final_state 차원과 맞춰야
                        initial_state로 바로 넣을 수 있어 편함 - encoder gru_units*2 = 512)
        attention_units: BahdanauAttention 내부 Dense 차원 (기본 256)

    call() 입력:
        decoder_input: (batch_size, 1) - 이번 스텝에 넣을 토큰 1개 (teacher forcing 시 정답,
                                          추론 시 직전 스텝의 argmax 예측)
        decoder_hidden: (batch_size, decoder_units) - 직전 스텝의 GRU 은닉 상태
                                                        (첫 스텝은 encoder_final_state)
        encoder_outputs: (batch_size, T_in, encoder_dim) - attention이 참조할 인코더 전체 출력

    call() 출력:
        logits: (batch_size, vocab_size) - 이번 스텝에서 다음 토큰일 확률 분포 (softmax 전/후는 아래 참고)
        new_decoder_hidden: (batch_size, decoder_units) - 다음 스텝에 넘겨줄 은닉 상태
        attention_weights: (batch_size, T_in, 1) - 디버깅/시각화용
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        decoder_units: int = 512,
        attention_units: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decoder_units = decoder_units

        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="decoder_embedding",
        )
        self.attention = BahdanauAttention(units=attention_units)
        self.gru = layers.GRU(
            decoder_units,
            return_sequences=False,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            name="decoder_gru",
        )
        # vocab_size 차원으로 projection. softmax는 여기서 적용하지 않고 raw logits을 반환한다.
        # -> train.py에서 SparseCategoricalCrossentropy(from_logits=True)를 쓰면 되고,
        #    추론(pipeline.py)에서는 tf.nn.softmax 또는 argmax를 직접 적용하면 된다.
        self.fc = layers.Dense(vocab_size, name="decoder_output_projection")

    def call(self, decoder_input, decoder_hidden, encoder_outputs, training: bool = False):
        # 1) Attention: 직전 은닉 상태 기준으로 encoder_outputs 어디를 볼지 계산
        context_vector, attention_weights = self.attention(decoder_hidden, encoder_outputs)

        # 2) 이번 스텝 입력 토큰 임베딩
        x = self.embedding(decoder_input)  # (B, 1, embedding_dim)

        # 3) context_vector를 시간축(1)으로 확장해서 embedding과 concat
        context_vector_expanded = tf.expand_dims(context_vector, axis=1)  # (B, 1, encoder_dim)
        gru_input = tf.concat([context_vector_expanded, x], axis=-1)  # (B, 1, encoder_dim+embedding_dim)

        # 4) GRU 한 스텝 실행 (initial_state로 직전 은닉 상태를 넣는다)
        gru_output, new_decoder_hidden = self.gru(
            gru_input, initial_state=decoder_hidden, training=training
        )  # gru_output: (B, decoder_units)

        # 5) vocab_size 차원으로 projection (softmax 미적용, raw logits)
        logits = self.fc(gru_output)  # (B, vocab_size)

        return logits, new_decoder_hidden, attention_weights


if __name__ == "__main__":
    STD_VOCAB_SIZE = 300
    BATCH_SIZE = 4
    T_IN = 10
    ENCODER_DIM = 512
    DECODER_UNITS = 512  # encoder_final_state 차원(512)과 맞춰야 initial_state로 바로 씀

    decoder = Decoder(
        vocab_size=STD_VOCAB_SIZE,
        embedding_dim=128,
        decoder_units=DECODER_UNITS,
        attention_units=256,
    )

    # encoder.py의 출력 shape을 흉내낸 더미 데이터
    dummy_encoder_outputs = tf.random.normal((BATCH_SIZE, T_IN, ENCODER_DIM))
    dummy_initial_hidden = tf.random.normal((BATCH_SIZE, DECODER_UNITS))  # 첫 스텝 = encoder_final_state 역할

    # <START> 토큰(id=1)을 첫 입력으로 가정
    dummy_decoder_input = tf.constant([[1]] * BATCH_SIZE, dtype=tf.int32)  # (B, 1)

    logits, new_hidden, attn_weights = decoder(
        dummy_decoder_input, dummy_initial_hidden, dummy_encoder_outputs
    )

    print("logits shape           :", logits.shape)        # 기대: (4, 300)
    print("new_decoder_hidden shape:", new_hidden.shape)     # 기대: (4, 512)
    print("attention_weights shape :", attn_weights.shape)   # 기대: (4, 10, 1)

    # 다음 스텝 예측 토큰(argmax) 확인 - 실제 추론 때 이렇게 다음 입력을 만든다
    predicted_ids = tf.argmax(logits, axis=-1)
    print("predicted token ids (스텝1):", predicted_ids.numpy())