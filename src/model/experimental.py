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
    bilstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(embedded)
    bilstm2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(bilstm1)
    dense = tf.keras.layers.Dense(100, activation='relu')(bilstm2)
    return dense


def create_multi_feature_model(input_attribs):
    """
    combine features from various inputs and build a model
    """
    inputs_list = []
    for idx, input_attrib in enumerate(input_attribs):
        inputs = tf.keras.Input(shape=(input_attrib['dim'],), name="Feature_%s" % (idx+1))
        feature = feature_extractors(inputs,
                                     input_attrib['vocab_size'],
                                     input_attrib['embedding_dim'])

        inputs_list.append(inputs)
        if idx == 0:
            concate = feature
        else:
            concate = tf.keras.layers.concatenate([concate, feature])

    bilstm = layers.Bidirectional(layers.LSTM(64))(concate)
    Dense1 = tf.keras.layers.Dense(100, activation='relu')(bilstm)
    Dense2 = tf.keras.layers.Dense(50, activation='relu')(Dense1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(Dense2)
    model = tf.keras.Model(inputs=inputs_list, outputs=outputs)

    return model
