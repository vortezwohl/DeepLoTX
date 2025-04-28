# LOTC: Easy2UseLongTextClassifierTrainers

## Installation

- Install with pip

    ```
    pip install git+https://github.com/vortezwohl/LOTC-Easy2UseLongTextClassifierTrainers.git
    ```

- Install with uv

    ```
    uv add git+https://github.com/vortezwohl/LOTC-Easy2UseLongTextClassifierTrainers.git
    ```

## Quick Start

To train a binary classifier for text files:

```python
from lotc.trainer import FileBinaryClassifierTrainer
from lotc.embedding import long_text_embedding
from lotc.util.read_file import get_files, read_file

trainer = FileBinaryClassifierTrainer(
    max_length=1024,
    chunk_size=256,
    batch_size=4,
    train_ratio=0.9
)

pos_data_path = './data/pos'
neg_data_path = './data/neg'
pos_data = [read_file(x) for x in get_files(pos_data_path)[:2]]
neg_data = [read_file(x) for x in get_files(neg_data_path)[:2]]
model = trainer.train(pos_data, neg_data, num_epochs=20, learning_rate=2e-5)
model.save()

model = model.load()
model.predict(long_text_embedding('这是一个测试文本.', max_length=1024, chunk_size=256)[1])
```
