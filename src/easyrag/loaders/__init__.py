"""Document loaders: turn files and sources into `Document` objects."""

from easyrag.loaders.base import BaseLoader
from easyrag.loaders.text import DirectoryLoader, TextLoader

__all__ = ["BaseLoader", "DirectoryLoader", "TextLoader"]
