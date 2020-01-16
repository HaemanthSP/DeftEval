import pickle

# Local
import sys
sys.path.append("..")
from common_imports import *


def save_numberer(numberer, path):
    with open(path, mode='wb') as file:
        pickle.dump(numberer, file)

def load_numberer(path):
    with open(path, mode='rb') as file:
        return pickle.load(file)


def save_tf_model(model, path):
    # path must have a '.h5' extension
    model.save(path)

def load_tf_model(path):
    return tf.keras.models.load_model(path)
