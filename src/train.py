import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np # type: ignore[import]
import tensorflow as tf # type: ignore[import]

from src.preprocessing.encoder import encode
from src.preprocessing.tokenizer import tokenize
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.vocab_builder import build_vocab, save_vocab
from src.model.dialtalk_model import DialTalkModel

ROOT = Path(__file__).parent.parent
def load_pairs(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "pairs" in data:
                data = data["pairs"]
            elif "data" in data:
                data = data["data"]
            else:
                raise ValueError("JSON 형식이 지원되지 않습니다. 'pairs' 또는 'data' 키가 필요합니다.")
        if isinstance(data, list):
            pairs = []
            for item in data:
                if isinstance(item, dict):
                    dialect = item.get("dialect") or item.get("src") or item.get("input")
                    standard = item.get("standard") or item.get("tgt") or item.get("target")
                    if dialect is not None and standard is not None:
                        pairs.append((str(dialect), str(standard)))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    pairs.append((str(item[0]), str(item[1])))
            return pairs
        raise ValueError("JSON 파일의 최상위 구조가 올바르지 않습니다.")

    if suffix in {".jsonl", ".txt"}:
        pairs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = line
                if isinstance(item, dict):
                    dialect = item.get("dialect") or item.get("src") or item.get("input")
                    standard = item.get("standard") or item.get("tgt") or item.get("target")
                    if dialect is not None and standard is not None:
                        pairs.append((str(dialect), str(standard)))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    pairs.append((str(item[0]), str(item[1])))
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        pairs.append((parts[0], parts[1]))
        return pairs

    raise ValueError(f"지원하지 않는 데이터 확장자입니다: {suffix}")


def prepare_dataset(
    pairs: List[Tuple[str, str]],
    max_input_len: int,
    max_output_len: int,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    # 수정 전 코드의 불필요한 중간 리스트(cleaned_dialect 등)를 없애고 한 번에 처리합니다.
    dialect_tokens = []
    standard_tokens = []

    # 리스트 컴프리헨션 4개를 단일 for문 1개로 압축
    cnt=0
    for d, s in pairs:
        if cnt%10000==0: print(cnt)
        dialect_tokens.append(tokenize(clean_text(d)))
        standard_tokens.append(tokenize(clean_text(s)))
        cnt+=1;
        

    dialect_vocab = build_vocab(dialect_tokens, min_freq=1)
    standard_vocab = build_vocab(standard_tokens, min_freq=1)

    input_ids = np.array(
        [encode(tokens, dialect_vocab, max_input_len, add_start_end=False) for tokens in dialect_tokens],
        dtype=np.int32,
    )
    target_ids = np.array(
        [encode(tokens, standard_vocab, max_output_len, add_start_end=True) for tokens in standard_tokens],
        dtype=np.int32,
    )

    return input_ids, target_ids, dialect_vocab, standard_vocab


def train_model(
    data_path: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 8,
    max_input_len: int = 16,
    max_output_len: int = 20,
    embedding_dim: int = 64,
    learning_rate: float = 1e-3,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print('......')
    pairs = load_pairs(data_path)
    if not pairs:
        raise ValueError("학습 데이터가 비어 있습니다.")
    print('......')
    input_ids, target_ids, dialect_vocab, standard_vocab = prepare_dataset(
        pairs,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
    )
    print('......')
    train_ds = tf.data.Dataset.from_tensor_slices((input_ids, target_ids)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print('......')
    model = DialTalkModel(
        dialect_vocab_size=len(dialect_vocab),
        standard_vocab_size=len(standard_vocab),
        start_id=1,
        end_id=2,
        embedding_dim=embedding_dim,
    )
    print('......')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    print('......')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print('......')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_input_ids, batch_target_ids in train_ds:
            with tf.GradientTape() as tape:
                logits, _ = model.forward_train(batch_input_ids, batch_target_ids, training=True)
                target_tokens = batch_target_ids[:, 1:]
                loss = loss_fn(target_tokens, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += float(loss.numpy())
        print('......')
        print(f"epoch {epoch + 1}/{epochs} - loss: {epoch_loss / max(1, len(input_ids) // batch_size):.4f}")

    save_vocab(dialect_vocab, str(output_dir / "dialect_vocab.json"))
    save_vocab(standard_vocab, str(output_dir / "standard_vocab.json"))
    
    model.build((None, 224, 224, 3))
    model.save(str(output_dir / "dialtalk_model.keras"))
    print(f"모델 저장 완료: {output_dir / 'dialtalk_model'}")


def main() -> None:
    print('asdf')
    parser = argparse.ArgumentParser(description="DialTalk 모델 학습 및 저장")
    print('.')
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "data.json")
    print('..')
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    print('...')
    parser.add_argument("--epochs", type=int, default=5)
    print('....')
    parser.add_argument("--batch-size", type=int, default=4)
    print('.....')
    parser.add_argument("--max-input-len", type=int, default=16)
    print('.....')
    parser.add_argument("--max-output-len", type=int, default=20)
    print('.....')
    parser.add_argument("--embedding-dim", type=int, default=64)
    print('.....')
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    print('.....')
    args = parser.parse_args()
    print('.....')
    train_model(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
