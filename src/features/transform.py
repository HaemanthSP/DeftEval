from enum import Enum
from tqdm import tqdm
import spacy
import tensorflow as tf
import numpy as np
import pandas as pd

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
                    tokens = [ele.token.lower() for ele in sent.tokens]
                elif input_primitive == InputPrimitive.POS:
                    tokens = [t.tag for t in NLP(' '.join([ele.token for ele in sent.tokens]), disable=['parser', 'ner'])]

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
