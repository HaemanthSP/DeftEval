import os

import pandas as pd
from pathlib import Path
import tensorflow_datasets as tfds
from tensorflow.keras import optimizers

from model import baseline


# Load dataset
def load_dataset(dataset_path):
    df = pd.DataFrame()
    for data_file in Path(dataset_path).iterdir():
        if data_file.suffix == '.deft':
            df = df.append(pd.read_table(
                    data_file, header=None, names=['sentences', 'label']))
    return df


def encode(dataset_path):
    test_df = load_dataset(os.path.join(dataset_path, 'dev'))
    train_df = load_dataset(os.path.join(dataset_path, 'train'))

    build_vocabulary(pd.concat([train_df, test_df]))
     
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


def build_vocabulary(dataframe):
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()
    for _, row in dataframe:
        some_tokens = tokenizer.tokenize(row['sentence'].numpy())
        vocabulary_set.update(some_tokens)
    return vocabulary_set


#def train():
#    model = baseline.create_task1_model(vocab_size, embedding_dim)
#    model.compile(loss='binary_crossentropy',
#                  optimizer=optimizers.Adam(0.0001),
#                  metrics=['accuracy'])
