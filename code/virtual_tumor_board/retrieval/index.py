from collections import Counter
from dataclasses import dataclass
from math import log
from re import findall


@dataclass(frozen=True)
class Passage:
    source_key: str
    title: str
    text: str
    version: str
    effective_year: int


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    score: float
    matched_terms: tuple[str, ...]


class LexicalIndex:
    def __init__(self, passages: tuple[Passage, ...]) -> None:
        if not passages:
            raise ValueError("at least one passage is required")
        keys = [item.source_key for item in passages]
        if len(keys) != len(set(keys)):
            raise ValueError("passage source keys must be unique")
        self._passages = passages
        self._documents = tuple(_tokens(item.title + " " + item.text) for item in passages)
        self._frequencies = tuple(Counter(document) for document in self._documents)
        self._document_frequency = self._build_document_frequency()

    def search(self, query: str, limit: int = 5) -> tuple[SearchHit, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_terms = _tokens(query)
        if not query_terms:
            return ()
        hits = []
        for passage, frequencies in zip(self._passages, self._frequencies, strict=True):
            matched = tuple(sorted(set(query_terms) & set(frequencies)))
            score = sum(self._term_score(term, frequencies[term]) for term in query_terms)
            if score > 0.0:
                hits.append(SearchHit(passage, score, matched))
        hits.sort(key=lambda item: (-item.score, item.passage.source_key))
        return tuple(hits[:limit])

    def sources(self) -> tuple[str, ...]:
        return tuple(item.source_key for item in self._passages)

    def _term_score(self, term: str, frequency: int) -> float:
        documents = len(self._passages)
        document_frequency = self._document_frequency.get(term, 0)
        inverse = log((documents + 1.0) / (document_frequency + 1.0)) + 1.0
        return (1.0 + log(frequency)) * inverse if frequency else 0.0

    def _build_document_frequency(self) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for document in self._documents:
            frequency.update(set(document))
        return frequency


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(findall(r"[a-z0-9]+", text.lower()))
