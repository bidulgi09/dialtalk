"""
model/dialtalk_model.py
DialTalk 모델 - 최종 통합 모델

Encoder + Decoder(반복 호출) + ConfidenceHead를 하나로 묶은 tf.keras.Model.
train.py와 pipeline.py 양쪽에서 이 클래스 하나만 가져다 쓰면 된다.

이 클래스는 두 가지 실행 모드를 명시적으로 분리해서 제공한다
(teacher forcing 학습과 greedy decoding 추론은 입력을 주는 방식 자체가 달라서,
 Keras 기본 call() 하나로 억지로 합치는 것보다 메서드를 나누는 편이 명확하다):

  - forward_train(): 학습 전용. teacher forcing으로 정답을 한 스텝씩 넣어가며
                      전체 타임스텝의 logits을 한 번에 반환한다. (train.py에서 사용)
  - infer():          추론 전용. <START>부터 시작해서 직전 예측을 다음 입력으로 넣는
                      greedy decoding을 수행한다. (pipeline.py에서 사용)

★ start_id/end_id를 vocab_builder.py에서 직접 import하지 않고 생성자 인자로 받는 이유:
  src/model/ 과 src/preprocessing/ 이 서로 다른 폴더라 상대 import가 번거롭다.
  대신 train.py/pipeline.py에서 vocab_builder.START_ID 값을 읽어와 이 모델 생성 시
  넘겨주는 방식으로 폴더 간 의존성을 없앴다. (다만 값 자체는 반드시 vocab_builder.py의
  START_ID=1, END_ID=2와 일치해야 한다 - 다르면 학습이 아예 안 된다.)
"""

import tensorflow as tf # type: ignore[import]

try:
    from .encoder import Encoder
    from .decoder import Decoder
    from .confidence_head import ConfidenceHead
except ImportError:  # pragma: no cover - allows running the file directly from src/model
    from encoder import Encoder
    from decoder import Decoder
    from confidence_head import ConfidenceHead


class DialTalkModel(tf.keras.Model):
    """
    Args (생성자):
        dialect_vocab_size: 방언(입력) vocab 크기
        standard_vocab_size: 표준어(타깃) vocab 크기
        start_id: vocab_builder.START_ID 값과 반드시 일치 (기본 1)
        end_id: vocab_builder.END_ID 값과 반드시 일치 (기본 2, 현재는 infer 조기종료에 미사용,
                추후 배치 내 개별 문장 조기종료 최적화 시 활용 예정)
        embedding_dim: 임베딩 차원 (기본 128)
        encoder_gru_units: encoder 단방향 GRU unit 수 (기본 256, 양방향이라 실제 512)
        decoder_units: decoder GRU unit 수 (기본 512 - encoder_final_state 차원과 일치시켜야
                        별도 projection 없이 바로 initial_state로 넣을 수 있음)
        attention_units: attention 내부 Dense 차원 (기본 256)
    """

    def __init__(
        self,
        dialect_vocab_size: int,
        standard_vocab_size: int,
        start_id: int = 1,
        end_id: int = 2,
        embedding_dim: int = 128,
        encoder_gru_units: int = 256,
        decoder_units: int = 512,
        attention_units: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 1. get_config()에서 꺼내 쓸 수 있도록 모든 설정 인자들을 인스턴스 변수로 바인딩합니다.
        self.dialect_vocab_size = dialect_vocab_size
        self.standard_vocab_size = standard_vocab_size
        self.start_id = start_id
        self.end_id = end_id
        self.embedding_dim = embedding_dim
        self.encoder_gru_units = encoder_gru_units
        self.decoder_units = decoder_units
        self.attention_units = attention_units

        # 2. 내부 서브 네트워크 정의
        self.encoder = Encoder(
            vocab_size=dialect_vocab_size,
            embedding_dim=embedding_dim,
            gru_units=encoder_gru_units,
        )
        self.decoder = Decoder(
            vocab_size=standard_vocab_size,
            embedding_dim=embedding_dim,
            decoder_units=decoder_units,
            attention_units=attention_units,
        )
        self.confidence_head = ConfidenceHead()


    def get_config(self):
        config = super().get_config()
        config.update({
            "dialect_vocab_size": self.dialect_vocab_size,
            "standard_vocab_size": self.standard_vocab_size,
            "start_id": self.start_id,
            "end_id": self.end_id,
            "embedding_dim": self.embedding_dim,
            "encoder_gru_units": self.encoder_gru_units,
            "decoder_units": self.decoder_units,
            "attention_units": self.attention_units,
        })
        return config

    # ★ 추가된 from_config 클래스 메서드 ★
    # 이 메서드는 나중에 tf.keras.models.load_model()을 통해 모델을 다시 불러올 때 활용됩니다.
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def forward_train(self, input_ids, target_ids, training: bool = True):
        """
        학습 전용 forward (teacher forcing).

        Args:
            input_ids: (B, T_in) 방언 입력
            target_ids: (B, T_out) 표준어 정답. encoder.py(전처리쪽)의 encode()에서
                        add_start_end=True로 만들어진 <START>...<END>...<PAD> 시퀀스여야 한다.
            training: True면 dropout 등 학습 모드로 동작

        Returns:
            all_logits: (B, T_out - 1, vocab_size)
                target_ids[:, 0]은 항상 <START>라서 "입력"으로만 쓰이고 예측 대상이 아니다.
                따라서 T_out - 1개의 스텝만 생성하며, 이는 target_ids[:, 1:]와 shape이 맞아
                손실 계산(SparseCategoricalCrossentropy)에 바로 사용할 수 있다.
            confidence_score: (B, 1)
        """
        encoder_outputs, encoder_final_state = self.encoder(input_ids, training=training)
        confidence_score = self.confidence_head(encoder_final_state, training=training)

        decoder_hidden = encoder_final_state
        target_len = target_ids.shape[1]  # 고정 길이 패딩이므로 정적 shape을 그대로 사용

        step_logits_list = []
        for t in range(target_len - 1):
            # t번째 정답 토큰을 입력으로 넣어 (t+1)번째 토큰을 예측하도록 학습 (teacher forcing)
            decoder_input = target_ids[:, t : t + 1]  # (B, 1)
            logits, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, training=training
            )
            step_logits_list.append(logits)

        all_logits = tf.stack(step_logits_list, axis=1)  # (B, T_out-1, vocab_size)

        return all_logits, confidence_score

    def infer(self, input_ids, max_output_len: int = 30):
        """
        추론 전용 forward (greedy decoding).

        Args:
            input_ids: (B, T_in) 방언 입력
            max_output_len: 최대 생성 길이 (설계문서 8장 MAX_OUTPUT_LEN과 동일 값 권장)

        Returns:
            predicted_ids: (B, max_output_len) 예측된 표준어 토큰 id 시퀀스.
                            <END> 이후에도 계속 토큰을 생성하지만, 전처리쪽 encoder.py의
                            decode(stop_at_end=True)가 <END> 이후를 잘라내므로 최종 결과엔
                            영향 없다.
            confidence_score: (B, 1)
        """
        encoder_outputs, encoder_final_state = self.encoder(input_ids, training=False)
        confidence_score = self.confidence_head(encoder_final_state, training=False)

        batch_size = tf.shape(input_ids)[0]
        decoder_hidden = encoder_final_state
        decoder_input = tf.fill((batch_size, 1), self.start_id)  # 첫 입력 = <START>

        predicted_tokens_list = []
        for _ in range(max_output_len):
            logits, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, training=False
            )
            predicted_token = tf.argmax(logits, axis=-1, output_type=tf.int32)  # (B,)
            predicted_tokens_list.append(predicted_token)

            # 다음 스텝 입력 = 이번 스텝 예측 (greedy)
            decoder_input = tf.expand_dims(predicted_token, axis=1)  # (B, 1)

        predicted_ids = tf.stack(predicted_tokens_list, axis=1)  # (B, max_output_len)

        return predicted_ids, confidence_score


if __name__ == "__main__":
    DIALECT_VOCAB_SIZE = 500
    STANDARD_VOCAB_SIZE = 400
    BATCH_SIZE = 4
    MAX_INPUT_LEN = 10
    MAX_OUTPUT_LEN = 12

    model = DialTalkModel(
        dialect_vocab_size=DIALECT_VOCAB_SIZE,
        standard_vocab_size=STANDARD_VOCAB_SIZE,
        start_id=1,
        end_id=2,
    )

    dummy_input_ids = tf.random.uniform(
        (BATCH_SIZE, MAX_INPUT_LEN), minval=1, maxval=DIALECT_VOCAB_SIZE, dtype=tf.int32
    )
    dummy_target_ids = tf.random.uniform(
        (BATCH_SIZE, MAX_OUTPUT_LEN), minval=1, maxval=STANDARD_VOCAB_SIZE, dtype=tf.int32
    )

    # 1) 학습 모드 (teacher forcing) 확인
    all_logits, confidence_score = model.forward_train(dummy_input_ids, dummy_target_ids, training=True)
    print("[forward_train]")
    print("all_logits shape      :", all_logits.shape)        # 기대: (4, 11, 400)  (12-1=11 스텝)
    print("confidence_score shape:", confidence_score.shape)  # 기대: (4, 1)

    # 2) 추론 모드 (greedy decoding) 확인
    predicted_ids, confidence_score2 = model.infer(dummy_input_ids, max_output_len=MAX_OUTPUT_LEN)
    print("\n[infer]")
    print("predicted_ids shape    :", predicted_ids.shape)      # 기대: (4, 12)
    print("confidence_score shape :", confidence_score2.shape)  # 기대: (4, 1)
    print("첫 샘플 예측 토큰 id    :", predicted_ids[0].numpy())

    # 3) 파라미터 개수 확인 (모델 크기 감 잡기용)
    total_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\n총 학습 파라미터 수: {total_params:,}")