import hashlib


def md5(text: str) -> str:
    md5_hash = hashlib.md5()
    text_bytes = text.encode('utf-8')
    md5_hash.update(text_bytes)
    return md5_hash.hexdigest()
