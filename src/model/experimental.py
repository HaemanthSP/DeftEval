import tensorflow as tf
from tensorflow.keras import layers


def simplified_baseline(vocab_size, embedding_dim):
    """
    Reduce the capacity of baseline untill it stops overfitting.
    Since the addition of new feature on already overfitting model will be ignored.
    """
    model = tf.keras.Sequential([
      layers.Embedding(vocab_size, embedding_dim),
      layers.Bidirectional(layers.LSTM(32)),
      layers.Dense(20, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(1, activation='sigmoid')
      ])
    return model
