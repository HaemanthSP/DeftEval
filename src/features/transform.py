from enum import Enum
from tqdm import tqdm
import spacy
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class InputPrimitive(Enum):
    TOKEN = 1,
    POS = 2,


spacy.prefer_gpu()
NLP = spacy.load("en_core_web_lg")

def tf_dataset_for_subtask_1(dataset, input_primitive, max_sent_len):
    x = []
    y = []
    vocabulary_set = set()

    for file in tqdm(dataset.files):
        for context in file.contexts:
            for sent in context.sentences:
                if input_primitive == InputPrimitive.TOKEN:
                    tokens = [x.token.lower() for x in sent.tokens]
                elif input_primitive == InputPrimitive.POS:
                    tokens = [x.tag for x in NLP(' '.join([x.token for x in sent.tokens]), disable=['parser', 'ner'])]

                label = 0
                for token in sent.tokens:
                    if token.tag[2:] == 'Definition':
                        label = 1
                        break

                np_arr = np.pad(np.asarray(tokens), [(0, max_sent_len - len(tokens))], constant_values='')
                x.append(np_arr)
                y.append(label)
                vocabulary_set.update(tokens)

    x = np.asarray(x)
    y = np.asarray(y)
    return tf.data.Dataset.from_tensor_slices((x, y)), vocabulary_set
