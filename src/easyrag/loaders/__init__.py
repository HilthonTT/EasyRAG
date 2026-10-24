"""Document loaders: turn files and sources into `Document` objects."""

from easyrag.loaders.directory import LOADERS_BY_SUFFIX, DirectoryLoader
from easyrag.loaders.base import BaseLoader
from easyrag.loaders.text import TextLoader
from easyrag.loaders.pdf import PDFLoader
from easyrag.loaders.html import HTMLLoader

__all__ = [
    "LOADERS_BY_SUFFIX",
    "BaseLoader",
    "DirectoryLoader",
    "TextLoader",
    "PDFLoader",
    "HTMLLoader",
]
