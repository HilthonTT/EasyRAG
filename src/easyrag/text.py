from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)

# A short English stopword list; enough to keep BM25 and hashing embeddings from matching on glue.
_STOPWORDS_TEXT = """
a an and are as at be but by can do does for from has have how i if in into is it its me my not
of on or our so than that the their them then there these they this to was we were what when where
which who why will with you your
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
