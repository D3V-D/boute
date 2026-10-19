"""Decode the standard Linux file-drag selection without TkDND conversion."""
from urllib.parse import unquote, urlsplit

from .core import InputError


def uri_paths(data: str) -> list[str]:
    paths = []
    for line in data.splitlines():
        if not line or line.startswith('#'):
            continue
        uri = urlsplit(line)
        if uri.scheme != 'file' or uri.netloc not in ('', 'localhost') or not uri.path.startswith('/'):
            raise InputError('The drag source did not provide a local file URI.')
        paths.append(unquote(uri.path, encoding='utf-8', errors='strict'))
    if not paths:
        raise InputError('The drag source did not provide any files.')
    return paths
