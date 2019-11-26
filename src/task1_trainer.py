# Built in packages
import os
import datetime

# Third party packages
import pandas as pd
from pathlib import Path
import tensorflow_datasets as tfds
from tensorflow.keras import optimizers, metrics
import tensorflow as tf
#import tensorflow_addons as tfa    # this is not available on Windows at the moment

# Local packages
from model import baseline, experimental


# define globals
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FIXME: Should this be set before imports?
BUFFER_SIZE = 20000
BATCH_SIZE = 64
TAKE_SIZE = 1000
VOCAB_SIZE = None


def load_dataset(dataset_path):
    """
    Load dataset from the files into a tensorflow dataset instance.
    """
    datasets = None
    num_instances = 0
    for data_file in Path(dataset_path).iterdir():
        if data_file.suffix == '.deft':
            df = pd.read_table(data_file,
                               header=None,
                               names=['sentence', 'label'])
            df['sentence'] = df['sentence'].values.astype(str)
            label = df.pop('label')
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                        (df['sentence'].values, label.values))
            num_instances += df.shape[0]
            if datasets:
                datasets = datasets.concatenate(tf_dataset)
            else:
                datasets = tf_dataset
    return (datasets, num_instances)


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
    """
    Prepare the raw dataset into encoded train, validation and test sets.
    """
    raw_train_dataset, _ = load_dataset(os.path.join(dataset_path, 'train'))
    raw_test_dataset, _ = load_dataset(os.path.join(dataset_path, 'dev'))

    # Shuffle only the training set. Since test set doesnt need to be shuffled.
    raw_train_dataset = raw_train_dataset.shuffle(
                BUFFER_SIZE, reshuffle_each_iteration=False)

    vocabulary_set = build_vocabulary(
            raw_test_dataset.concatenate(raw_train_dataset))
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

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
    VOCAB_SIZE = len(vocabulary_set) + 1

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
    train_data, valid_data, test_data, encoder = prepare_data(dataset_path)

    print("Loading baseline model")
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

    model.fit(train_data,
              epochs=10,
              validation_data=valid_data,
              callbacks=[tensorboard_callback])

    eval_loss, eval_precision, eval_recall = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval precision: {:.3f}, Eval recall: {:.3f}'.format(eval_loss, eval_precision, eval_recall))

    predictions = model.predict(test_data)
    print_mispredictions(test_data.unbatch(), predictions, encoder,
                         '../data/task_1_deft_files/test_mispredictions.txt')

    return model


if __name__ == '__main__':
    train('../data/task_1_deft_files/')
