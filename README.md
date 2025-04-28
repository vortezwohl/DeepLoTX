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

pos_data = 'path/to/positives'
neg_data = 'path/to/negatives'
trainer = FileBinaryClassifierTrainer(
    model_name='example-model',
    max_length=4096,
    num_epochs=20,
    train_loss_threshold=4
)

trainer.train(pos_data, neg_data)
trainer.save()

model = trainer.load()
model.predict(long_text_embedding('这是一个测试文本.', max_length=4096, chunk_size=256)[1])
```
