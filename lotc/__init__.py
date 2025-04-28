import logging
import os

__ROOT__ = os.path.dirname(os.path.abspath(__file__))

from .encoder import BertEncoder
from .encoder import long_text_encoder
from .nn import LinearRegression, LogisticRegression, SoftmaxRegression
from .trainer import FileBinaryClassifierTrainer

__AUTHOR__ = '吴子豪 / Vortez Wohl'
__EMAIL__ = 'vortez.wohl@gmail.com'
__GITHUB__ = 'https://github.com/vortezwohl'
__BLOG__ = 'https://vortezwohl.github.io'

logger = logging.getLogger('lotc')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s : %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger = logging.getLogger('lotc.trainer')
logger.setLevel(logging.DEBUG)
logger = logging.getLogger('lotc.embedding')
logger.setLevel(logging.DEBUG)
