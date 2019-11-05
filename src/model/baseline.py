import tensorflow as tf
from tensorflow.keras import layers


def create_task1_model(vocab_size, embedding_dim):
    model = tf.keras.Sequential([
      layers.Embedding(vocab_size, embedding_dim),
      layers.Bidirectional(layers.LSTM(100)),
      layers.Dense(50, activation='relu'),
      layers.Dense(1, activation='sigmoid')
      ])
    return model


def create_task2_model():
    raise NotImplementedError


def create_task3_model():
    raise NotImplementedError
