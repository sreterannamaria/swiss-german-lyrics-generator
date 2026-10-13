"""Train and evaluate the character-level GRU models and write the results to results/.

Usage:
    python train.py --data-dir data --out-dir results
"""

import argparse
import json
import math
import time
from pathlib import Path

import keras
import matplotlib
import numpy as np
import tensorflow as tf

from lyrics_gen.baselines import CharNGram
from lyrics_gen.data import Vocabulary, load_corpus, train_val_split
from lyrics_gen.metrics import copied_ngram_rate, known_word_rate, longest_copied_span
from lyrics_gen.model import Generator, build_model, make_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEQ_LENGTH = 100
BATCH_SIZE = 64
SEEDS = ["i ", "und ", "das ", "isch ", "u "]  # start words used in the original study
TEMPERATURES = [0.2, 0.5, 0.8, 1.0]

CONFIGS = {
    # Same architecture and epochs as the original study (TensorFlow tutorial defaults).
    "original": dict(embedding_dim=256, rnn_units=1024, dropout=0.0, epochs=30, early_stopping=False),
    # Same model trained much longer, to show what happens once it starts to memorise.
    "original_long": dict(embedding_dim=256, rnn_units=1024, dropout=0.0, epochs=100, early_stopping=False),
    # Smaller network with dropout, stopped when the validation loss stops improving.
    "regularised": dict(embedding_dim=128, rnn_units=256, dropout=0.3, epochs=100, early_stopping=True),
}


def train(name, config, vocab, train_ids, val_ids):
    keras.utils.set_random_seed(42)
    model = build_model(len(vocab), config["embedding_dim"], config["rnn_units"], config["dropout"])
    callbacks = []
    if config["early_stopping"]:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True))

    start = time.time()
    history = model.fit(
        make_dataset(train_ids, SEQ_LENGTH, BATCH_SIZE, shuffle=True),
        validation_data=make_dataset(val_ids, SEQ_LENGTH, BATCH_SIZE, shuffle=False),
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=2,
    )
    h = history.history
    best_epoch = int(np.argmin(h["val_loss"]))
    used_epoch = best_epoch if config["early_stopping"] else len(h["loss"]) - 1
    return model, {
        "params": int(model.count_params()),
        "epochs_trained": len(h["loss"]),
        "training_minutes": round((time.time() - start) / 60, 1),
        "train_loss": [round(x, 4) for x in h["loss"]],
        "val_loss": [round(x, 4) for x in h["val_loss"]],
        "best_val_epoch": best_epoch + 1,
        "best_val_loss": round(h["val_loss"][best_epoch], 4),
        "final_model_epoch": used_epoch + 1,
        "final_model_train_loss": round(h["loss"][used_epoch], 4),
        "final_model_val_loss": round(h["val_loss"][used_epoch], 4),
        "final_model_val_perplexity": round(math.exp(h["val_loss"][used_epoch]), 2),
    }


def evaluate_generation(model, vocab, train_text):
    rng = np.random.default_rng(42)
    generator = Generator(model)
    results = {}
    for temperature in TEMPERATURES:
        samples = [generator.generate(vocab, seed, length=400, temperature=temperature, rng=rng) for seed in SEEDS]
        results[str(temperature)] = {
            "known_word_rate": round(float(np.mean([known_word_rate(s, train_text) for s in samples])), 3),
            "copied_5gram_rate": round(float(np.mean([copied_ngram_rate(s, train_text, 5) for s in samples])), 3),
            "longest_copied_span_chars": int(np.mean([longest_copied_span(s, train_text) for s in samples])),
            "samples": samples,
        }
    return results


LABELS = {"original_long": "GRU 1024 units", "regularised": "GRU 256 units with dropout"}
REPORTED_2024_TRAIN_LOSS = 0.61  # training loss after 30 epochs reported in the 2024 study


def plot(metrics, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"original_long": "#d35400", "regularised": "#2471a3"}
    for name, m in metrics["models"].items():
        if name not in LABELS:
            continue  # "original" equals the first 30 epochs of "original_long" (same seed)
        epochs = range(1, len(m["train_loss"]) + 1)
        ax.plot(epochs, m["train_loss"], color=colors[name], linestyle="--", label=f"{LABELS[name]}: training")
        ax.plot(epochs, m["val_loss"], color=colors[name], label=f"{LABELS[name]}: validation")

    best = min(metrics["baselines"].items(), key=lambda kv: kv[1]["val_loss"])
    ax.axhline(best[1]["val_loss"], color="grey", linestyle=":", label=f"best n-gram baseline ({best[0].split('_')[1].replace('gram', '-gram')}): validation")

    long_run = metrics["models"].get("original_long")
    if long_run:
        epoch = next(i for i, loss in enumerate(long_run["train_loss"], 1) if loss <= REPORTED_2024_TRAIN_LOSS)
        ax.annotate(f"training loss {REPORTED_2024_TRAIN_LOSS} (value reported in 2024)\nvalidation loss already rising",
                    xy=(epoch, long_run["train_loss"][epoch - 1]), xytext=(epoch + 8, 1.2),
                    fontsize=8, arrowprops=dict(arrowstyle="->", color="grey"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy per character (nats)")
    ax.set_title("Training vs. validation loss")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.2, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_outputs(metrics, out):
    """Write metrics.json (numbers only), samples.md and the loss plot.

    Samples are only published for models that do not reproduce the training lyrics,
    because those lyrics are protected by copyright.
    """
    samples = {}
    for model, by_temperature in metrics["generation"].items():
        for temperature, result in by_temperature.items():
            texts = result.pop("samples")
            if result["copied_5gram_rate"] < 0.05:
                samples.setdefault(model, {})[temperature] = texts

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Generated samples", "",
             "Start words as in the original study. Samples of the model trained for 100 epochs are not shown:",
             "most of their text is copied verbatim from the (copyrighted) training lyrics.", ""]
    for model, by_temperature in samples.items():
        lines += [f"## {model}", ""]
        for temperature in ("0.5", "0.8"):
            for text in by_temperature.get(temperature, [])[:2]:
                lines += [f"**Temperature {temperature}**", "", "```", text.strip()[:300], "```", ""]
    (out / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    plot(metrics, out / "loss_curves.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    corpus = load_corpus(args.data_dir)
    train_text, val_text = train_val_split(corpus)
    vocab = Vocabulary(train_text)
    train_ids, val_ids = vocab.encode(train_text), vocab.encode(val_text)

    metrics = {
        "data": {
            "artists": {artist: len(text) for artist, text in corpus.items()},
            "train_chars": len(train_text),
            "val_chars": len(val_text),
            "vocab_size": len(vocab),
        },
        "baselines": {},
        "models": {},
        "generation": {},
    }

    for n in (1, 3, 5):
        ce = CharNGram(n).fit(train_text, len(vocab)).cross_entropy(val_text)
        metrics["baselines"][f"char_{n}gram"] = {"val_loss": round(ce, 4), "val_perplexity": round(math.exp(ce), 2)}

    for name, config in CONFIGS.items():
        model, model_metrics = train(name, config, vocab, train_ids, val_ids)
        metrics["models"][name] = {"config": config, **model_metrics}
        metrics["generation"][name] = evaluate_generation(model, vocab, train_text)
        print(name, json.dumps({k: v for k, v in model_metrics.items() if "loss" not in k or "final" in k or "best" in k}))

    write_outputs(metrics, out)


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
