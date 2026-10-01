import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional


def get_word_stem(w: str) -> str:
    """Lightweight rule-based morphological stemmer for English words."""
    if len(w) <= 3:
        return w
    for suffix in ["ing", "ed", "ly", "tion", "able", "ment"]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            stem = w[:-len(suffix)]
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] in "bdfgmnprt":
                stem = stem[:-1]
            return stem
    if w.endswith("ies") and len(w) >= 5:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) >= 4:
        return w[:-1]
    return w


def tokenize(text: str) -> List[str]:
    """Code-aware and multi-lingual tokenizer for English, CJK, code identifiers, and symbols."""
    if not text:
        return []
    tokens = []
    # 1. Base English / ASCII alphanumeric words and identifiers
    en_words = re.findall(r"[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*", text.lower())
    for w in en_words:
        tokens.append(w)
        stem = get_word_stem(w)
        if stem != w and stem not in tokens:
            tokens.append(stem)
        if "_" in w or "." in w or "-" in w:
            for part in re.split(r"[_.-]+", w):
                if len(part) > 1 and part != w:
                    tokens.append(part)

    # 2. Extract camelCase sub-identifiers from original case
    camel_words = re.findall(r"[a-zA-Z0-9]+", text)
    for cw in camel_words:
        splits = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[0-9]+", cw)
        if len(splits) > 1:
            for s in splits:
                s_lower = s.lower()
                if len(s_lower) > 1 and s_lower not in tokens:
                    tokens.append(s_lower)

    # 3. Chinese / CJK unigrams and bigrams
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    for c in zh_chars:
        tokens.append(c)
    for i in range(len(zh_chars) - 1):
        tokens.append(zh_chars[i] + zh_chars[i + 1])

    return tokens


class BM25Index:
    """High-performance BM25 Index with inverted index postings."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_ids: List[str] = []
        self.doc_lens: List[int] = []
        # Inverted index: token -> list of (doc_index, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[Tuple[str, str]]):
        """Fit index with a list of (doc_id, text) tuples in O(N_tokens)."""
        self.corpus_size = len(documents)
        self.doc_ids = []
        self.doc_lens = []
        self.inverted_index = {}
        total_length = 0

        for idx, (doc_id, text) in enumerate(documents):
            self.doc_ids.append(doc_id)
            tokens = tokenize(text)
            l = len(tokens)
            self.doc_lens.append(l)
            total_length += l

            counts = Counter(tokens)
            for token, freq in counts.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((idx, freq))

        self.avgdl = (total_length / self.corpus_size) if self.corpus_size > 0 else 1.0

        # Calculate BM25 Robertson-Sparck Jones IDF
        self.idf = {}
        for token, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[token] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> List[Tuple[str, float]]:
        """Score documents against query using inverted index lookup with query-calibrated normalization."""
        if not self.corpus_size:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return [(doc_id, 0.0) for doc_id in self.doc_ids]

        scores = [0.0] * self.corpus_size
        max_possible = 0.0
        seen_q = set()
        default_idf = math.log(1.0 + (self.corpus_size + 0.5) / 0.5)

        for q in query_tokens:
            if q in seen_q:
                continue
            seen_q.add(q)
            q_idf = self.idf.get(q, default_idf)
            max_possible += q_idf * (self.k1 + 1.0)

            if q in self.inverted_index:
                for doc_idx, tf in self.inverted_index[q]:
                    doc_len = self.doc_lens[doc_idx]
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_idx] += q_idf * (numerator / denominator)

        if max_possible > 0:
            scores = [round(s / max_possible, 4) for s in scores]

        ranked = [(self.doc_ids[i], scores[i]) for i in range(self.corpus_size)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
