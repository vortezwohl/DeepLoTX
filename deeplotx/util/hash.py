import hashlib


def md5(text: str) -> str:
    _hash = hashlib.md5()
    text_bytes = text.encode('utf-8')
    _hash.update(text_bytes)
    return _hash.hexdigest()


def sha1(text: str) -> str:
    _hash = hashlib.sha1()
    text_bytes = text.encode('utf-8')
    _hash.update(text_bytes)
    return _hash.hexdigest()


def sha256(text: str) -> str:
    _hash = hashlib.sha256()
    text_bytes = text.encode('utf-8')
    _hash.update(text_bytes)
    return _hash.hexdigest()


def sha512(text: str) -> str:
    _hash = hashlib.sha512()
    text_bytes = text.encode('utf-8')
    _hash.update(text_bytes)
    return _hash.hexdigest()
