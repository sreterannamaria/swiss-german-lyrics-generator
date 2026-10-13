"""Metrics for generated text: known words and copying from the training data."""

import re


def words(text):
    return re.findall(r"[a-z0-9]+", text)


def known_word_rate(generated, training_text):
    """Share of generated words that also occur in the training text.

    An automatic stand-in for the manual "correct word" count used in the
    original study.
    """
    vocabulary = set(words(training_text))
    generated_words = words(generated)
    if not generated_words:
        return 0.0
    return sum(w in vocabulary for w in generated_words) / len(generated_words)


def copied_ngram_rate(generated, training_text, n=5):
    """Share of word n-grams in the generated text that appear verbatim in the training text.

    A high value means the model reproduces existing lyrics instead of writing new ones.
    """
    training_words = words(training_text)
    training_ngrams = {tuple(training_words[i:i + n]) for i in range(len(training_words) - n + 1)}
    generated_words = words(generated)
    ngrams = [tuple(generated_words[i:i + n]) for i in range(len(generated_words) - n + 1)]
    if not ngrams:
        return 0.0
    return sum(g in training_ngrams for g in ngrams) / len(ngrams)


def longest_copied_span(generated, training_text):
    """Length (in characters) of the longest substring of `generated` found in the training text."""
    longest = 0
    start = 0
    for end in range(1, len(generated) + 1):
        while start < end and generated[start:end] not in training_text:
            start += 1
        longest = max(longest, end - start)
    return longest
