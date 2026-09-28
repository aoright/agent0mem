import math
import re
from typing import List, Dict, Any, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenize English words, numbers, and CJK characters."""
    if not text:
        return []
    text = text.lower()
    # Match words (including numbers and underscores) or individual CJK characters
    tokens = re.findall(r"[\w]+|[\u4e00-\u9fff]", text)
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_ids: List[str] = []

    def fit(self, documents: List[Tuple[str, str]]):
        """Fit index with a list of (doc_id, text) tuples."""
        self.doc_ids = []
        self.doc_tokens = []
        self.doc_lens = []
        self.doc_freqs = {}

        total_length = 0
        for doc_id, text in documents:
            tokens = tokenize(text)
            self.doc_ids.append(doc_id)
            self.doc_tokens.append(tokens)
            l = len(tokens)
            self.doc_lens.append(l)
            total_length += l

            # Track doc frequencies
            seen_tokens = set(tokens)
            for t in seen_tokens:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.corpus_size = len(documents)
        self.avgdl = (total_length / self.corpus_size) if self.corpus_size > 0 else 0.0

        # Calculate IDF
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            # BM25 IDF formula with smoothing
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> List[Tuple[str, float]]:
        """Score all documents against query and return list of (doc_id, score) sorted desc."""
        if not self.corpus_size:
            return []

        query_tokens = tokenize(query)
        scores = [0.0] * self.corpus_size

        for q in query_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]

            for idx, doc in enumerate(self.doc_tokens):
                # Count frequency of q in doc
                tf = doc.count(q)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                scores[idx] += q_idf * (numerator / denominator)

        ranked = [(self.doc_ids[i], scores[i]) for i in range(self.corpus_size)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
