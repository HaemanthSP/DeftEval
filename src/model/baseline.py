import sys
sys.path.append("..")
from common_imports import *


def create_task1_model(vocab_size, embedding_dim):
    model = tf.keras.Sequential([
      layers.Embedding(vocab_size, embedding_dim),
      layers.Bidirectional(layers.LSTM(64)),
      layers.Dense(100, activation='relu'),
      layers.Dense(50, activation='relu'),
      layers.Dense(1, activation='sigmoid')
      ])
    return model


def create_task2_model(input_attribs, num_tags):
    def feature_extractors(inputs, vocab_size, embedding_dim):
        embedded = layers.Embedding(vocab_size, embedding_dim, embeddings_regularizer=regularizers.l2(0.001))(inputs)
        return embedded

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
    bilstm = layers.Bidirectional(layers.LSTM(
        32, kernel_regularizer=regularizers.l2(0.001), return_sequences=True))(concate)
    dropout = layers.Dropout(0.5)(bilstm)
    outputs = layers.Dense(num_tags, activation='softmax')(dropout)
    model = tf.keras.Model(inputs=inputs_list, outputs=outputs)

    return model


def create_task3_model():
    raise NotImplementedError
