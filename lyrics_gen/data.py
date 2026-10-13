"""Loading, normalising and splitting the lyrics corpus."""

import re
import unicodedata
from pathlib import Path

# Umlauts are written as digraphs, as in the original corpus ("ä" -> "ae").
UMLAUTS = {"ä": "ae", "ö": "oe", "ü": "ue", "à": "a", "è": "e", "é": "e"}


def normalise(text):
    """Lowercase, replace umlauts and keep only letters, digits, spaces and line breaks."""
    text = unicodedata.normalize("NFC", text).lower()
    for umlaut, replacement in UMLAUTS.items():
        text = text.replace(umlaut, replacement)
    text = re.sub(r"[^a-z0-9 \n]", "", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def load_corpus(data_dir):
    """Return {artist: normalised text} for every *.txt file in data_dir."""
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir} (see data/README.md).")
    return {f.stem: normalise(f.read_text(encoding="utf-8")) for f in files}


def train_val_split(corpus, val_fraction=0.1):
    """Hold out the last part of every artist's text as validation data.

    Splitting per artist keeps every artist in both sets. Cutting at a line break
    keeps the validation text from starting mid-line.
    """
    train_parts, val_parts = [], []
    for text in corpus.values():
        cut = int(len(text) * (1 - val_fraction))
        line_break = text.rfind("\n", 0, cut)
        if line_break > 0:
            cut = line_break + 1
        train_parts.append(text[:cut])
        val_parts.append(text[cut:])
    return "".join(train_parts), "".join(val_parts)


class Vocabulary:
    """Character <-> integer id mapping. Id 0 is reserved for unknown characters."""

    def __init__(self, text):
        self.chars = ["<unk>"] + sorted(set(text))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.index.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids if i != 0)
