import os

import pandas as pd
from pathlib import Path
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras import optimizers, metrics
import tensorflow as tf

from model import baseline

BUFFER_SIZE = 20000
BATCH_SIZE = 64
TAKE_SIZE = 1000
VOCAB_SIZE = None


# Load dataset
def load_dataset(dataset_path):
    datasets = None
    for data_file in Path(dataset_path).iterdir():
        if data_file.suffix == '.deft':
            df = pd.read_table(data_file,
                               header=None,
                               names=['sentence', 'label'])
            df['sentence'] = df['sentence'].values.astype(str)
            label = df.pop('label')
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                        (df['sentence'].values, label.values))
            if datasets:
                datasets = datasets.concatenate(tf_dataset)
            else:
                datasets = tf_dataset
    return datasets


def build_vocabulary(dataset):
    """
    Build vocabulary for encoding.
    """
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()
    for text_tensor, _ in dataset:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    print("Vocabulary Size: ", len(vocabulary_set))
    return vocabulary_set


def prepare_data(dataset_path):

    raw_train_dataset = load_dataset(os.path.join(dataset_path, 'train'))
    raw_test_dataset = load_dataset(os.path.join(dataset_path, 'dev'))

    # # Shuffle
    raw_train_dataset = raw_train_dataset.shuffle(
                BUFFER_SIZE, reshuffle_each_iteration=False)
    raw_test_dataset = raw_test_dataset.shuffle(
                BUFFER_SIZE, reshuffle_each_iteration=False)
    # test_dataset

    vocabulary_set = build_vocabulary(
            raw_test_dataset.concatenate(raw_train_dataset))
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    # example_text = next(iter(raw_train_dataset))[0].numpy()
    # print(example_text)
    # encoded_example = encoder.encode(example_text)
    # print(encoded_example)

    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(
                encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    test_data = raw_test_dataset.map(encode_map_fn)
    train_data = raw_train_dataset.map(encode_map_fn)
    print(next(iter(test_data)))

    # Shuffle training data before sampling validation set
    train_data = train_data.shuffle(BUFFER_SIZE,
                                    reshuffle_each_iteration=False)

    print("Cardinality:", tf.data.experimental.cardinality(train_data))
    train_data = train_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    valid_data = train_data.take(TAKE_SIZE)

    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))
    valid_data = valid_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))
    ## valid_data = None
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

    # Additional one for padding element
    global VOCAB_SIZE
    VOCAB_SIZE = len(vocabulary_set) + 1

    return train_data, valid_data, test_data


def train(dataset_path):
    train_data, valid_data, test_data = prepare_data(dataset_path)

    print("Loading baseline model")
    model = baseline.create_task1_model(VOCAB_SIZE, 64)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.0001),
                  metrics=[metrics.Precision(), metrics.Recall()])

    model.fit(train_data, epochs=10, validation_data=valid_data)

    eval_loss, eval_acc = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
    return model
