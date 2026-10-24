"""Document loaders: turn files and sources into `Document` objects."""

from easyrag.loaders.directory import DirectoryLoader
from easyrag.loaders.base import BaseLoader
from easyrag.loaders.text import TextLoader
from easyrag.loaders.pdf import PDFLoader
from easyrag.loaders.html import HTMLLoader

__all__ = ["BaseLoader", "DirectoryLoader", "TextLoader", "PDFLoader", "HTMLLoader"]
