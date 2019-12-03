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


def feature_extractors(inputs, vocab_size, embedding_dim):
    """
    Extract feature from a sequential input
    """
    embedded = layers.Embedding(vocab_size, embedding_dim)(inputs)
    feature = layers.Bidirectional(layers.LSTM(64))(embedded)
    return feature


def create_multi_feature_model(input_attribs):
    """
    combine features from various inputs and build a model
    """
    inputs_list = []
    for idx, input_attrib in enumerate(input_attribs):
        inputs = tf.keras.Input(shape=(input_attrib['dim'],))
        feature = feature_extractors(inputs,
                                     input_attrib['vocab_size'],
                                     input_attrib['embedding_dim'])

        inputs_list.append(inputs)
        if idx == 0:
            concate = feature
        else:
            concate = tf.keras.layers.concatenate([concate, feature])

    Dense1 = tf.keras.layers.Dense(100, activation='relu')(concate)
    Dense2 = tf.keras.layers.Dense(50, activation='relu')(Dense1)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(Dense2)
    model = tf.keras.Model(inputs=inputs_list, outputs=outputs)

    return model
