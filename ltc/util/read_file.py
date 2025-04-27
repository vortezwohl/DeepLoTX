import os


def read_file(path: str, encoding: str = 'utf-8') -> str:
    try:
        with open(path, mode='r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(path, mode='r', encoding='gbk') as f:
                return f.read()
        except UnicodeDecodeError:
            pass


def get_files(path: str) -> list:
    entries = os.listdir(path)
    return [os.path.join(path, entry) for entry in entries if os.path.isfile(os.path.join(path, entry))]
