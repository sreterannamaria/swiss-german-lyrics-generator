import math

from lyrics_gen.baselines import CharNGram
from lyrics_gen.data import Vocabulary, normalise, train_val_split
from lyrics_gen.metrics import copied_ngram_rate, known_word_rate, longest_copied_span


def test_normalise_replaces_umlauts_and_removes_punctuation():
    assert normalise("Är isch vom Amt ufbotte gsy!") == "aer isch vom amt ufbotte gsy\n"


def test_train_val_split_keeps_every_artist_and_cuts_at_line_breaks():
    corpus = {"a": "zeile eins\nzeile zwei\nzeile drei\n", "b": "eis\nzwoei\ndrue\n"}
    train, val = train_val_split(corpus, val_fraction=0.4)
    assert train + val != ""
    assert "zeile drei" in val and "drue" in val
    assert "zeile eins" in train and "eis" in train
    assert sorted(train + val) == sorted("".join(corpus.values()))


def test_vocabulary_round_trip_and_unknown_characters():
    vocab = Vocabulary("abc")
    assert vocab.decode(vocab.encode("cab")) == "cab"
    assert vocab.encode("x") == [0]


def test_ngram_model_is_a_probability_distribution():
    model = CharNGram(3).fit("abcabcabc", vocab_size=3)
    assert math.isclose(sum(model.prob("ab", c) for c in "abc"), 1.0)
    assert model.cross_entropy("abc") < math.log(3)


def test_copy_metrics():
    training = "i han es zuendhoelzli azuendt und das het e flamme gaeh"
    assert copied_ngram_rate("i han es zuendhoelzli azuendt", training, n=5) == 1.0
    assert copied_ngram_rate("mir hei e verein i ghoere derzue", training, n=5) == 0.0
    assert known_word_rate("das het xyz", training) == 2 / 3
    assert longest_copied_span("xx zuendhoelzli yy", training) == len(" zuendhoelzli ")
