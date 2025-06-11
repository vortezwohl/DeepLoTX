from deeplotx import TextBinaryClassifierTrainer, LongTextEncoder
from deeplotx.util import get_files, read_file

long_text_encoder = LongTextEncoder(
    max_length=2048,
    chunk_size=512,
    overlapping=128
)

trainer = TextBinaryClassifierTrainer(
    long_text_encoder=long_text_encoder,
    batch_size=4,
    train_ratio=0.9
)

pos_data_path = './data/pos'
neg_data_path = './data/neg'
pos_data = [read_file(x) for x in get_files(pos_data_path)]
neg_data = [read_file(x) for x in get_files(neg_data_path)]
model = trainer.train(pos_data, neg_data, num_epochs=20, learning_rate=2e-5, train_loss_threshold=1)
model.predict(long_text_encoder.encode('这是一个测试文本.').squeeze())
