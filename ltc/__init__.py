import logging
import os

__ROOT__ = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger('ltc')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s : %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger = logging.getLogger('ltc.trainer')
logger.setLevel(logging.DEBUG)
logger = logging.getLogger('ltc.embedding')
logger.setLevel(logging.DEBUG)
