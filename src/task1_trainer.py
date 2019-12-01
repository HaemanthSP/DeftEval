# Built in packages
import os
import datetime
from enum import Enum

# Third party packages
from tqdm import tqdm
import spacy
import pandas as pd
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pathlib import Path
import tensorflow_datasets as tfds
from tensorflow.keras import optimizers, metrics
# import tensorflow_addons as tfa    # this is not available on Windows at the moment


# Local packages
from model import baseline, experimental
from features import corpus, transform
from util.numberer import Numberer


# define globals
BUFFER_SIZE = 20000
BATCH_SIZE = 64
TAKE_SIZE = 1000
VOCAB_SIZE = None


def prepare_data(dataset_path, primitive_type):
    """
    Prepare the raw dataset into encoded train, validation and test sets.
    """
    print("Loading dataset")
    raw_train_dataset = corpus.load_files_into_dataset(os.path.join(dataset_path, 'train'))
    raw_test_dataset = corpus.load_files_into_dataset(os.path.join(dataset_path, 'dev'))

    print("Transforming dataset")
    max_sent_len = max(raw_train_dataset.max_sent_len,
                       raw_test_dataset.max_sent_len)
    raw_train_dataset, train_vocab = transform.tf_dataset_for_subtask_1(raw_train_dataset,
                                                                        primitive_type, max_sent_len)
    raw_test_dataset, test_vocab = transform.tf_dataset_for_subtask_1(raw_test_dataset,
                                                                      primitive_type, max_sent_len)
    combined_vocab = train_vocab.union(test_vocab)
    print("Train vocab Size: ", len(train_vocab))
    print("Test vocab Size: ", len(test_vocab))
    print("Vocabulary Size: ", len(combined_vocab))

    # Shuffle only the training set. Since test set doesnt need to be shuffled.
    raw_train_dataset = raw_train_dataset.shuffle(
                BUFFER_SIZE, reshuffle_each_iteration=False)

    encoder = Numberer(combined_vocab)

    def encode_map_fn(features, label):
        def inner(features, label):
            encoded_features = [encoder.number(x) for x in features.numpy()]
            return encoded_features, label

        return tf.py_function(
                inner, inp=[features, label], Tout=(tf.int32, tf.int8))

    test_data = raw_test_dataset.map(encode_map_fn)
    train_data = raw_train_dataset.map(encode_map_fn)
    print(next(iter(test_data)))

    # Shuffle training data before sampling validation set
    train_data_temp = train_data.shuffle(BUFFER_SIZE,
                                         reshuffle_each_iteration=False)

    print("Cardinality:", tf.data.experimental.cardinality(train_data))
    train_data = train_data_temp.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    valid_data = train_data_temp.take(TAKE_SIZE)

    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))
    valid_data = valid_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1], []))

    # Additional one for padding element
    global VOCAB_SIZE
    VOCAB_SIZE = len(combined_vocab) + 1

    return train_data, valid_data, test_data, encoder


def print_mispredictions(gold_dataset, predictions, encoder, filepath):
    mispredictions = []
    for idx, data in enumerate(gold_dataset):
        if (predictions[idx] > 0.5 and data[1].numpy() != 1) or (predictions[idx] < 0.5 and data[1].numpy() != 0):
            mispredictions.append(encoder.decode(data[0].numpy()))

    with open(filepath, 'w', encoding="utf-8") as f:
        for i in mispredictions:
            f.write(i + "\n")


def train(dataset_path):
    print("Preparing data")
    train_data, valid_data, test_data, encoder = prepare_data(dataset_path, transform.InputPrimitive.TOKEN)

    print("Loading model")
    model = experimental.simplified_baseline(VOCAB_SIZE, 64)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.0001),
                  metrics=[metrics.Precision(), metrics.Recall()])

    if os.name == 'nt':
        log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\"
    else:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

    print("Training model")
    model.fit(train_data,
              epochs=10,
              validation_data=valid_data,
              callbacks=[tensorboard_callback])

    eval_loss, eval_precision, eval_recall = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval precision: {:.3f}, Eval recall: {:.3f}'.format(eval_loss, eval_precision, eval_recall))

    predictions = model.predict(test_data)
    print_mispredictions(test_data.unbatch(), predictions, encoder,
                         '../deft_corpus/data/task_1_test_mispredictions.txt')

    return model


if __name__ == '__main__':
    train('../deft_corpus/data/deft_files/')
