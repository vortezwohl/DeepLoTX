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

pos = 'path/to/positives'
neg = 'path/to/negatives'
trainer = FileBinaryClassifierTrainer(
    model_name='FileBinaryClassifier_1',
    max_length=4096,
    num_epochs=10,
    train_loss_threshold=20.0
)
model = trainer.train(pos, neg)
trainer.save(model)
```
