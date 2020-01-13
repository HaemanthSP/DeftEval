# Local
import sys
sys.path.append("..")
from common_imports import *
from util.numberer import Numberer


class InputPrimitive(Enum):
    TOKEN = 1,
    POS = 2,
    POS_WPUNCT = 3,
    DEP = 4

def get_token(tokens, nlp_annotations):
    return [ele.token.lower() for ele in tokens]


def get_pos_with_punct(tokens, nlp_annotations):
    return [t.text if t.pos_ == 'PUNCT' else t.pos_ for t in nlp_annotations]


def get_pos(tokens, nlp_annotations):
    return [t.pos_ for t in nlp_annotations]


def get_dep(tokens, nlp_annotations):
    return [t.dep_ for t in nlp_annotations]


def tf_datasets_for_subtask_1(train_dataset, test_dataset, input_primitives, max_feature_vector_length):

    def generate_primitives_and_vocabulary(dataset, input_primitives, x, y, vocabulary_set):
        feature_map = {'POS': get_pos,
                       'TOKEN': get_token,
                       'POS_WPUNCT': get_pos_with_punct,
                       'DEP': get_dep}

        num_pos = 0
        num_neg = 0
        for file in tqdm(dataset.files):
            for context in file.contexts:
                for sent in context.sentences:
                    feature_inputs = []
                    for idx, primitive in enumerate(input_primitives):
                        feature_input = feature_map[primitive.name](sent.tokens, sent.nlp_annotations)
                        vocabulary_set[idx].update(feature_input)
                        feature_inputs.append(feature_input)

                    label = 0
                    for token in sent.tokens:
                        if token.tag[2:] == 'Definition':
                            label = 1
                            break

                    if label == 1:
                        num_pos += 1
                    else:
                        num_neg += 1

                    x.append(feature_inputs)
                    y.append(label)

        return num_pos, num_neg

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

            x[row_idx] = {}
            for idx, feat in enumerate(new_feature_arrays, 1):
                x[row_idx].update({"Feature_%s" %(idx): feat})

    # x_train, y_train, x_test, y_test = [[]] * 4
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    combined_vocabs = [set() for i in input_primitives]

    print("Generating primitives and constructing vocabulary")
    num_pos_train, num_neg_train = generate_primitives_and_vocabulary(train_dataset, input_primitives, x_train, y_train, combined_vocabs)
    num_pos_test, num_neg_test = generate_primitives_and_vocabulary(test_dataset, input_primitives, x_test, y_test, combined_vocabs)

    print("Encoding primitives")
    encoders = [Numberer(vocab) for vocab in combined_vocabs]
    # For now all features are padded with same length
    # TODO: Make custom padding length for individual features
    feature_vector_length = min(max_feature_vector_length, max(train_dataset.max_sent_len, test_dataset.max_sent_len))
    feature_vector_shapes = [feature_vector_length] * len(input_primitives)
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

    return train_dataset, test_dataset, combined_vocabs, encoders, feature_vector_length
