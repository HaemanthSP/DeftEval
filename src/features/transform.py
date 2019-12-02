from enum import Enum
from tqdm import tqdm
import spacy
import tensorflow as tf
import numpy as np
import pandas as pd
from util.numberer import Numberer

class InputPrimitive(Enum):
    TOKEN = 1,
    POS = 2,

spacy.prefer_gpu()
NLP = spacy.load("en_core_web_lg")

def tf_datasets_for_subtask_1(train_dataset, test_dataset, input_primitive):
    def generate_primitives_and_vocabulary(dataset, input_primitive, x, y, vocabulary_set):
        for file in tqdm(dataset.files):
            for context in file.contexts:
                for sent in context.sentences:
                    if input_primitive == InputPrimitive.TOKEN:
                        tokens = [ele.token.lower() for ele in sent.tokens]
                    elif input_primitive == InputPrimitive.POS:
                        tokens = [t.pos_ for t in NLP(' '.join([ele.token for ele in sent.tokens]), disable=['parser', 'ner'])]

                    label = 0
                    for token in sent.tokens:
                        if token.tag[2:] == 'Definition':
                            label = 1
                            break
                    x.append(tokens)
                    y.append(label)
                    vocabulary_set.update(tokens)

    def encode_primitives(x, encoder, shape):
        for row_idx, row in enumerate(tqdm(x)):
            new_feature_array = np.zeros(shape, dtype=np.int32)
            for primitive_idx, primitive in enumerate(row):
                new_feature_array[primitive_idx] = encoder.number(primitive)

            x[row_idx] = new_feature_array

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    combined_vocab = set()

    print("Generating primitives and constructing vocabulary")
    generate_primitives_and_vocabulary(train_dataset, input_primitive, x_train, y_train, combined_vocab)
    generate_primitives_and_vocabulary(test_dataset, input_primitive, x_test, y_test, combined_vocab)

    print("Encoding primitives")
    encoder = Numberer(combined_vocab)
    feature_vector_shape = [max(train_dataset.max_sent_len, test_dataset.max_sent_len)]
    encode_primitives(x_train, encoder, feature_vector_shape)
    encode_primitives(x_test, encoder, feature_vector_shape)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train, dtype=np.int8)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test, dtype=np.int8)
    return tf.data.Dataset.from_tensor_slices((x_train, y_train)), tf.data.Dataset.from_tensor_slices((x_test, y_test)), combined_vocab, encoder
