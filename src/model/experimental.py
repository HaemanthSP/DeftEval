import tensorflow as tf
from tensorflow.keras import layers


def simplified_baseline(vocab_size, embedding_dim):
    """
    Reduce the capacity of baseline untill it stops overfitting.
    Since the addition of new feature on already overfitting model will be ignored.
    """
    model = tf.keras.Sequential([
      layers.Embedding(vocab_size, embedding_dim),
      layers.Bidirectional(layers.LSTM(64)),
      layers.Dense(100, activation='relu'),
      layers.Dense(50, activation='relu'),
      layers.Dense(1, activation='sigmoid')
      ])
    return model
