import sys
sys.path.append("..")
from common_imports import *


def simplified_baseline(vocab_size, embedding_dim):
    """
    Reduce the capacity of baseline untill it stops overfitting.
    Since the addition of new feature on already overfitting model will be ignored.
    """
    model = tf.keras.Sequential([
      layers.Embedding(vocab_size, embedding_dim),
      layers.Conv1D(50,
		3,
		padding='valid',
		activation='relu',
		strides=1),
	  layers.MaxPooling1D(pool_size=4),
    #   layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
      layers.Bidirectional(layers.LSTM(100)),
    #   layers.Dense(100, activation='relu'),
    #   layers.Dense(50, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(1, activation='sigmoid')
      ])
    return model


def feature_extractors(inputs, vocab_size, embedding_dim):
    """
    Extract feature from a sequential input
    """
    embedded = layers.Embedding(vocab_size, embedding_dim, embeddings_regularizer=regularizers.l2(0.001))(inputs)
    # bilstm1 = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(embedded)
    # bilstm2 = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(bilstm1)
    # bilstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(embedded)
    # dense = tf.keras.layers.Dense(8, activation='relu')(bilstm1)
    return embedded


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
    conv1 = layers.Conv1D(128, 5, activation='relu')(concate)
    conv2 = layers.Conv1D(64, 4, activation='relu')(conv1)
    conv3 = layers.Conv1D(64, 3, activation='relu')(conv2)
    # bilstm1 = layers.Bidirectional(layers.LSTM(32, kernel_regularizer=regularizers.l2(0.001), use_bias=False, return_sequences=True))(conv2)
    bilstm1 = layers.Bidirectional(layers.LSTM(64, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001), return_sequences=True, use_bias=False))(conv3)
    bilstm2 = layers.Bidirectional(layers.LSTM(64, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001), use_bias=False))(bilstm1)
    # bilstm2 = layers.Bidirectional(layers.LSTM(32, kernel_regularizer=regularizers.l2(0.001), use_bias=False))(bilstm1)
    # Dense1 = tf.keras.layers.Dense(100, activation='relu')(bilstm)
    # Dense2 = tf.keras.layers.Dense(50, activation='relu')(Dense1)
    Dense2 = layers.Dense(64, activation='relu')(bilstm2)
    Drop = layers.Dropout(0.5)(Dense2)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(Drop)
    model = tf.keras.Model(inputs=inputs_list, outputs=outputs)

    return model
