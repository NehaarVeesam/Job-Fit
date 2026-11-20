"""Small, explicit BM25 implementation over the same job text as dense search."""

import math
import re
from collections import Counter

TOKENIZER_VERSION = "jobfit-technical-tokens-v1"
# Keep punctuation that distinguishes technical skills. Do not split C++ into C.
TOKEN = re.compile(r"(?<![\w.+#])(?:\.net|[a-z0-9]+(?:[./-][a-z0-9]+)*(?:\+\+|#)?)(?![\w+#])", re.I)
ALIASES = {"nodejs": "node.js", "dotnet": ".net", "k8s": "kubernetes",
           "postgres": "postgresql", "sklearn": "scikit-learn"}


def tokenize(text):
    return [ALIASES.get(token.casefold(), token.casefold()) for token in TOKEN.findall(text)]


class BM25Index:
    """Positive-IDF BM25, with k1 saturation and b document-length normalization.

    IDF = log(1 + (N - df + 0.5)/(df + 0.5)). Query terms are deduplicated.
    Build only on eligible documents so all retrieval methods share a corpus.
    """

    def __init__(self, documents, *, k1=1.5, b=0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.counts = {}
        frequencies = Counter()
        for job_id, text in documents:
            if job_id in self.counts:
                raise ValueError("Duplicate job ID")
            terms = Counter(tokenize(text))
            self.counts[job_id] = terms
            frequencies.update(terms.keys())
        n = len(self.counts)
        self.lengths = {key: sum(value.values()) for key, value in self.counts.items()}
        self.average_length = sum(self.lengths.values()) / n if n else 0
        self.idf = {term: math.log1p((n - count + 0.5) / (count + 0.5))
                    for term, count in frequencies.items()}

    def search(self, query, *, k=30):
        if not 1 <= k <= 250:
            raise ValueError("k must be between 1 and 250")
        query_terms = sorted(set(tokenize(query)))
        scores = []
        for job_id, counts in self.counts.items():
            normalization = self.k1 * (1 - self.b + self.b * self.lengths[job_id] /
                                       (self.average_length or 1))
            score = sum(self.idf.get(term, 0) * counts[term] * (self.k1 + 1) /
                        (counts[term] + normalization) for term in query_terms if counts[term])
            if score > 0:
                scores.append((job_id, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:k]
