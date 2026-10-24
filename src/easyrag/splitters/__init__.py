"""Text splitters: break documents into chunks for indexing."""

from easyrag.splitters.base import BaseSplitter
from easyrag.splitters.character import CharacterSplitter

__all__ = ["BaseSplitter", "CharacterSplitter"]
