"""Character n-gram baselines to put the neural model's numbers into context."""

import math
from collections import Counter, defaultdict


class CharNGram:
    """Character n-gram model with add-k smoothing."""

    def __init__(self, n, k=0.1):
        self.n = n
        self.k = k

    def fit(self, text, vocab_size):
        self.vocab_size = vocab_size
        self.counts = defaultdict(Counter)
        padded = "\n" * (self.n - 1) + text
        for i in range(self.n - 1, len(padded)):
            self.counts[padded[i - self.n + 1:i]][padded[i]] += 1
        return self

    def prob(self, context, char):
        counter = self.counts.get(context, Counter())
        return (counter[char] + self.k) / (sum(counter.values()) + self.k * self.vocab_size)

    def cross_entropy(self, text):
        """Average negative log-likelihood per character (in nats, like Keras)."""
        padded = "\n" * (self.n - 1) + text
        total = 0.0
        for i in range(self.n - 1, len(padded)):
            total -= math.log(self.prob(padded[i - self.n + 1:i], padded[i]))
        return total / len(text)
