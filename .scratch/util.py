from common_imports import *
import pickle

class Serde:
    @staticmethod
    def save_metadata(metadata, path):
        with open(path, mode='wb') as file:
            pickle.dump(metadata, file)


    @staticmethod
    def load_metadata(path):
        with open(path, mode='rb') as file:
            return pickle.load(file)


    @staticmethod
    def save_tf_model(model, path):
        assert path.endswith('.h5') == True
        model.save(path)


    @staticmethod
    def load_tf_model(path):
        return tf.keras.models.load_model(path)


class Numberer:
    def __init__(self, vocabulary):
        self.v2n = dict()
        self.n2v = list()
        self.INVALID_NUMBER = 0

        for item in vocabulary:
            _ = self.number(item, add_if_absent=True)


    def number(self, value, add_if_absent=False):
        n = self.v2n.get(value)

        if n is None:
            if add_if_absent:
                n = len(self.n2v) + 1
                self.v2n[value] = n
                self.n2v.append(value)
            else:
                n = self.INVALID_NUMBER

        return n


    def value(self, number):
        assert number > self.INVALID_NUMBER
        return self.n2v[number - 1]


    def max_number(self):
        return len(self.n2v)


class Preprocessor:
    @staticmethod
    def replace_urls(text, replacement=' url '):
        return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+ ', replacement, text)


    @staticmethod
    def add_space_around(text, elements=r'([+\-\{\}\[\]\(\)=–\"\'])'):
        return re.sub(elements, r' \1 ', text)


    @staticmethod
    def remove_quotes(text):
        return re.sub(r'^\"(.+)\"$', r'\1', text)
