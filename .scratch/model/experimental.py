import sys
sys.path.append("..")
from common_imports import *
from tensorflow.keras.initializers import Constant


def feature_extractors(inputs, vocab_size, embedding_dim, embedding_initializer):
    embedded = layers.Embedding(vocab_size, embedding_dim,
                                trainable=True,
                                mask_zero=True,
                                embeddings_initializer=Constant(embedding_initializer) if embedding_initializer is not None else 'uniform',
                                embeddings_regularizer=regularizers.l2(0.001),
                                input_length=inputs.shape[1])

    return embedded(inputs)


def create_multi_feature_model(input_attribs):
    inputs_list = []
    for idx, input_attrib in enumerate(input_attribs):
        inputs = tf.keras.Input(shape=(input_attrib['dim'],), name="Feature_%s" % (idx+1))
        feature = feature_extractors(inputs,
                                     input_attrib['vocab_size'],
                                     input_attrib['embedding_dim'],
                                     input_attrib['embedding_initializer'])

        inputs_list.append(inputs)
        if idx == 0:
            concate = feature
        else:
            concate = tf.keras.layers.concatenate([concate, feature])

    # hyperparams
    kernel_size = 4
    pool_size = 2
    strides=1

    layer_accum = layers.Conv1D(256,
                        kernel_size=kernel_size,
                        padding='valid',
			                  activation='relu',
			                  strides=strides)(concate)

    layer_accum = layers.Conv1D(256,
                        kernel_size=kernel_size,
                        padding='valid',
			                  activation='relu',
			                  strides=strides)(layer_accum)

    layer_accum = layers.Conv1D(256,
                        kernel_size=kernel_size,
                        padding='valid',
			                  activation='relu',
			                  strides=strides)(layer_accum)

    layer_accum = layers.Conv1D(64,
                        kernel_size=kernel_size,
                        padding='valid',
			                  activation='relu',
			                  strides=strides)(layer_accum)

    layer_accum = layers.Conv1D(64,
                        kernel_size=kernel_size,
                        padding='valid',
			                  activation='relu',
			                  strides=strides)(layer_accum)


    layer_accum = layers.Conv1D(64,
                        kernel_size=kernel_size,
                        padding='valid',
			                  activation='relu',
			                  strides=strides)(layer_accum)

    layer_accum = layers.MaxPooling1D(pool_size=pool_size)(layer_accum)
    layer_accum = layers.Flatten()(layer_accum)
    layer_accum = layers.Dense(48, kernel_regularizer=regularizers.l2(0.001))(layer_accum)
    layer_accum = layers.Dropout(0.5)(layer_accum)
    layer_accum = layers.Dense(24, kernel_regularizer=regularizers.l2(0.001))(layer_accum)
    layer_accum = layers.Dropout(0.5)(layer_accum)
    layer_accum = layers.Dense(12, kernel_regularizer=regularizers.l2(0.001))(layer_accum)
    layer_accum = layers.Dropout(0.5)(layer_accum)
    outputs = layers.Dense(1, kernel_regularizer=regularizers.l2(0.001), activation='sigmoid')(layer_accum)
    model = tf.keras.Model(inputs=inputs_list, outputs=outputs)

    return model
