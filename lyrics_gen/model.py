"""Character-level GRU language model (Keras)."""

import numpy as np
import keras
import tensorflow as tf


def build_model(vocab_size, embedding_dim=256, rnn_units=1024, dropout=0.0):
    model = keras.Sequential([
        keras.Input(shape=(None,), dtype="int32"),
        keras.layers.Embedding(vocab_size, embedding_dim),
        keras.layers.GRU(rnn_units, return_sequences=True, dropout=dropout),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(vocab_size),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def make_dataset(ids, seq_length, batch_size, shuffle, seed=42):
    """Split the id sequence into (input, target) windows shifted by one character."""
    ds = tf.data.Dataset.from_tensor_slices(np.array(ids, dtype="int32"))
    ds = ds.batch(seq_length + 1, drop_remainder=True)
    ds = ds.map(lambda seq: (seq[:-1], seq[1:]))
    if shuffle:
        ds = ds.shuffle(10_000, seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class Generator:
    """Generates text one character at a time, carrying the GRU state from step to step.

    Re-uses the trained weights in a copy of the network whose GRU also returns its state,
    so each new character costs one small step instead of a pass over the whole context.
    """

    def __init__(self, model):
        embedding, gru, _, dense = model.layers
        self.embedding = embedding
        self.dense = dense
        self.gru = keras.layers.GRU(gru.units, return_sequences=True, return_state=True)
        self.gru.build((None, None, embedding.output_dim))
        self.gru.set_weights(gru.get_weights())

    @tf.function
    def _step(self, ids, state):
        outputs, state = self.gru(self.embedding(ids), initial_state=state)
        return self.dense(outputs[:, -1, :]), state

    def generate(self, vocab, seed_text, length=400, temperature=1.0, rng=None):
        rng = rng or np.random.default_rng()
        ids = vocab.encode(seed_text)
        state = tf.zeros((1, self.gru.units))
        logits, state = self._step(tf.constant([ids], dtype="int32"), state)
        for _ in range(length):
            logits = logits[0].numpy().astype("float64")
            logits[0] = -np.inf  # never generate <unk>
            logits = logits / temperature
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            next_id = int(rng.choice(len(probs), p=probs))
            ids.append(next_id)
            logits, state = self._step(tf.constant([[next_id]], dtype="int32"), state)
        return vocab.decode(ids)
