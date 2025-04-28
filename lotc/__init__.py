import logging
import os

import torch

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__ROOT__ = os.path.dirname(os.path.abspath(__file__))

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
