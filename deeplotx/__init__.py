import logging
import os

__ROOT__ = os.path.dirname(os.path.abspath(__file__))

from .encoder import Encoder, LongTextEncoder, LongformerEncoder
from .nn import (
    BaseNeuralNetwork,
    FeedForward,
    LinearRegression,
    LogisticRegression,
    SoftmaxRegression,
    RecursiveSequential,
    LongContextRecursiveSequential,
    RoPE,
    Attention,
    RoFormerEncoder,
    AutoRegression,
    LongContextAutoRegression
)
from .trainer import TextBinaryClassifierTrainer

__AUTHOR__ = '吴子豪 / Vortez Wohl'
__EMAIL__ = 'vortez.wohl@gmail.com'
__GITHUB__ = 'https://github.com/vortezwohl'
__BLOG__ = 'https://vortezwohl.github.io'

logger = logging.getLogger('deeplotx')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s : %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger = logging.getLogger('deeplotx.trainer')
logger.setLevel(logging.DEBUG)
logger = logging.getLogger('deeplotx.embedding')
logger.setLevel(logging.DEBUG)
