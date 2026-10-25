"""Vocabulary, word embeddings and bidirectional RNN classifiers (Keras)."""

from collections import Counter

import numpy as np

PAD_INDEX = 0
OOV_INDEX = 1


def build_vocabulary(texts, num_words):
    """
    Map the (num_words - 2) most frequent tokens to integer ids.
    Id 0 is reserved for padding and id 1 for out-of-vocabulary tokens.
    Fit on the training texts only.
    """
    counts = Counter(token for text in texts for token in text.split())
    most_common = [word for word, _ in counts.most_common(num_words - 2)]
    return {word: idx + 2 for idx, word in enumerate(most_common)}


def texts_to_padded_sequences(texts, vocabulary, maxlen):
    """Convert texts to a (n_texts, maxlen) integer matrix, zero-padded at the front."""
    sequences = np.full((len(texts), maxlen), PAD_INDEX, dtype="int32")
    for row, text in enumerate(texts):
        ids = [vocabulary.get(token, OOV_INDEX) for token in text.split()][-maxlen:]
        if ids:
            sequences[row, -len(ids):] = ids
    return sequences


def build_embedding_matrix(vocabulary, word_vectors, vocab_size):
    """
    Build an embedding matrix from pretrained word vectors (gensim KeyedVectors).
    Words without a pretrained vector keep a zero row and are learned in training.
    """
    matrix = np.zeros((vocab_size, word_vectors.vector_size))
    missing = []
    for word, idx in vocabulary.items():
        if word in word_vectors:
            matrix[idx] = word_vectors[word]
        else:
            missing.append(word)
    found = len(vocabulary) - len(missing)
    print(f"{found} of {len(vocabulary)} vocabulary words have a pretrained vector "
          f"({len(missing)} not found, initialized as zeros).")
    return matrix, missing


def build_bidirectional_rnn_classifier(
    vocab_size,
    embedding_dim,
    rnn_type="gru",
    hidden_units=16,
    learning_rate=0.001,
    embedding_weights=None,
    trainable_embedding=True,
    rnn_activation="relu",
):
    """
    Build and compile a Bidirectional GRU or LSTM sentiment classifier.

    If `embedding_weights` is given (from build_embedding_matrix), the
    embedding layer is initialized with these pretrained vectors.
    """
    import keras

    layer_by_type = {"gru": keras.layers.GRU, "lstm": keras.layers.LSTM}
    if rnn_type.lower() not in layer_by_type:
        raise ValueError("rnn_type must be either 'gru' or 'lstm'")
    rnn_layer = layer_by_type[rnn_type.lower()]

    if embedding_weights is not None:
        embedding_dim = embedding_weights.shape[1]
        initializer = keras.initializers.Constant(embedding_weights)
    else:
        initializer = "uniform"

    model = keras.Sequential([
        keras.Input(shape=(None,), dtype="int32"),
        keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=initializer,
            trainable=trainable_embedding,
        ),
        keras.layers.Bidirectional(rnn_layer(hidden_units, activation=rnn_activation)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
