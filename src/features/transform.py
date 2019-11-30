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

def tf_dataset_for_subtask_1(dataset, input_primitive):
    x = []
    y = []
    vocabulary_set = set()

    for file in tqdm(dataset.files):
        for context in file.contexts:
            for sent in context.sentences:
                if input_primitive == InputPrimitive.TOKEN:
                    tokens = [x.token.lower() for x in sent.tokens]
                elif input_primitive == InputPrimitive.POS:
                    tokens = [x.tag for x in NLP(' '.join(sent.tokens), disable=['parser', 'ner'])]

                label = 0
                for token in sent.tokens:
                    if token.tag[3:] == 'Definition':
                        label = 1
                        break

                x.append(np.array(tokens))
                y.append(label)
                vocabulary_set.update(tokens)

    return tf.data.Dataset.from_tensor_slices((np.array(x), np.array(y))), vocabulary_set
