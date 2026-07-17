"""
model/confidence_head.py
DialTalk 모델 - Confidence Head

역할 (설계문서 4.2, F-04 참고):
  encoder_final_state(문장 전체를 요약한 벡터) 하나만 보고,
  이 방언 문장을 얼마나 자신 있게 표준어로 변환할 수 있는지를 0~1 사이 값으로 예측한다.

  이 값이 0.7 미만이면 pipeline.py의 후처리에서 needs_reconfirmation=True로 판단하여
  PRD 화면3(F-04)의 "확인이 필요합니다" 재확인 팝업을 띄운다.

★ 설계상 위치: Decoder와 완전히 독립된 별도 브랜치다.
  즉 encoder_final_state 한 곳에서 갈라져 나오는 멀티태스크 학습 구조이며,
  Decoder가 몇 스텝을 생성하든 ConfidenceHead는 encoder_final_state 하나만 보고
  단 한 번만 계산하면 된다 (Decoder의 반복 스텝과 무관하게 독립적으로 동작).

입력:
  encoder_final_state: (batch_size, encoder_state_dim)  - encoder.py의 두 번째 출력

출력:
  confidence_score: (batch_size, 1)  - sigmoid로 0~1 범위 보장
"""

import tensorflow as tf # type: ignore[import]
from tensorflow.keras import layers # type: ignore[import]


class ConfidenceHead(tf.keras.layers.Layer):
    """
    Args (생성자):
        hidden_units_1: 첫 번째 Dense 레이어 차원 (기본 128)
        hidden_units_2: 두 번째 Dense 레이어 차원 (기본 64)
        dropout_rate: 과적합 방지용 dropout 비율 (기본 0.2)
                      -> confidence_label 데이터가 적을 가능성이 높아 dropout을 넣어둠
                         (설계문서 3.5: 학습 라벨 확보가 확실치 않은 부분이라 과적합 위험 큼)
    """

    def __init__(
        self,
        hidden_units_1: int = 128,
        hidden_units_2: int = 64,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_units_1, activation="relu", name="confidence_dense1")
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(hidden_units_2, activation="relu", name="confidence_dense2")
        self.dropout2 = layers.Dropout(dropout_rate)
        # sigmoid로 0~1 범위를 강제한다 (PRD의 confidence_score 스키마: 0.0~1.0)
        self.output_layer = layers.Dense(1, activation="sigmoid", name="confidence_output")

    def call(self, encoder_final_state, training: bool = False):
        x = self.dense1(encoder_final_state)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        confidence_score = self.output_layer(x)  # (B, 1)
        return confidence_score


if __name__ == "__main__":
    BATCH_SIZE = 4
    ENCODER_STATE_DIM = 512  # encoder.py의 encoder_final_state 차원과 일치해야 함

    confidence_head = ConfidenceHead(hidden_units_1=128, hidden_units_2=64, dropout_rate=0.2)

    dummy_encoder_final_state = tf.random.normal((BATCH_SIZE, ENCODER_STATE_DIM))

    # training=False (추론 모드) - dropout 비활성화
    confidence_score = confidence_head(dummy_encoder_final_state, training=False)

    print("confidence_score shape:", confidence_score.shape)  # 기대: (4, 1)
    print("confidence_score 값    :", confidence_score.numpy().flatten())

    # 모든 값이 0~1 범위 안에 있는지 확인 (sigmoid 검증)
    values = confidence_score.numpy().flatten()
    in_range = all(0.0 <= v <= 1.0 for v in values)
    print("모두 0~1 범위 안에 있는가:", in_range)