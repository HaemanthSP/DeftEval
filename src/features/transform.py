# Built in
from enum import Enum

# Third party
import spacy
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Local
from util.numberer import Numberer

spacy.prefer_gpu()
NLP = spacy.load("en_core_web_lg")
PAD_FEATURE_VECTORS = True


class InputPrimitive(Enum):
    TOKEN = 1,
    POS = 2,


def get_token(tokens):
    return [ele.token.lower() for ele in tokens]


def get_pos(tokens):
    return [t.pos_ for t in
            NLP(' '.join([ele.token for ele in tokens]),
                disable=['parser', 'ner'])]


def tf_datasets_for_subtask_1(train_dataset, test_dataset, input_primitives):

    def generate_primitives_and_vocabulary(dataset, input_primitives, x, y, vocabulary_set):
        feature_map = {'POS': get_pos, 'TOKEN': get_token}
        for file in tqdm(dataset.files):
            for context in file.contexts:
                for sent in context.sentences:
                    feature_inputs = []
                    for idx, primitive in enumerate(input_primitives):
                        feature_input = feature_map[primitive.name](sent.tokens)
                        vocabulary_set[idx].update(feature_input)
                        feature_inputs.append(feature_input)

                    label = 0
                    for token in sent.tokens:
                        if token.tag[2:] == 'Definition':
                            label = 1
                            break
                    x.append(feature_inputs)
                    y.append(label)

    def encode_primitives(x, encoders, shapes):
        # store the encoded primitives as a byte string to keep the tensor length fixed
        # individual features can be extracted using a map operation on the generated tf.data.Dataset object
        # the alternative would be to pad it to the maximum sentence length in advance
        for row_idx, row in enumerate(tqdm(x)):
            new_feature_arrays = [np.zeros(shape, dtype=np.int32)
                                  for shape in shapes]

            for idx, feature_input in enumerate(row):
                for primitive_idx, primitive in enumerate(feature_input):
                    new_feature_arrays[idx][primitive_idx] = encoders[idx].number(primitive)

            x[row_idx] = {"Feature_1": new_feature_arrays[0],
                          "Feature_2": new_feature_arrays[1]}

    # x_train, y_train, x_test, y_test = [[]] * 4
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    combined_vocabs = [set() for i in input_primitives]

    print("Generating primitives and constructing vocabulary")
    generate_primitives_and_vocabulary(train_dataset, input_primitives, x_train, y_train, combined_vocabs)
    generate_primitives_and_vocabulary(test_dataset, input_primitives, x_test, y_test, combined_vocabs)

    print("Encoding primitives")
    encoders = [Numberer(vocab) for vocab in combined_vocabs]
    # For now all features are padded with same length
    # TODO: Make custom padding length for individual features
    feature_vector_shapes = [max(train_dataset.max_sent_len, test_dataset.max_sent_len)] * len(input_primitives)
    encode_primitives(x_train, encoders, feature_vector_shapes)
    encode_primitives(x_test, encoders, feature_vector_shapes)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train, dtype=np.int8)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test, dtype=np.int8)

    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    def train_generator():
        for x, y in zip(x_train, y_train):
            yield x, y

    def test_generator():
        for x, y in zip(x_test, y_test):
            yield x, y
    types = {"Feature_"+str(i+1): tf.int32 for i, _ in enumerate(input_primitives)}, tf.int8
    shapes = {"Feature_"+str(i+1): tf.TensorShape([None,]) for i, _ in enumerate(input_primitives)}, tf.TensorShape([])
    train_dataset = tf.data.Dataset.from_generator(train_generator, types, shapes)
    test_dataset = tf.data.Dataset.from_generator(test_generator, types, shapes)

    return train_dataset, test_dataset, combined_vocabs, encoders
