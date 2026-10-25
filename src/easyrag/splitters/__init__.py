"""Text splitters: break documents into chunks for indexing."""

from easyrag.splitters.base import BaseSplitter
from easyrag.splitters.character import CharacterSplitter
from easyrag.splitters.recursive import RecursiveCharacterSplitter

__all__ = ["BaseSplitter", "CharacterSplitter", "RecursiveCharacterSplitter"]
