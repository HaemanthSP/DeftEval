from common_imports import *

class Token:
    """
    Represents an individual token in a sentence
    """
    def __init__(self, token, start_char=None, end_char=None, tag=None, tag_id=None, root_id=None, relation=None):
        self.token = token
        self.start_char = start_char
        self.end_char = end_char
        self.tag = tag
        self.tag_id = tag_id
        self.root_id = root_id
        self.relation = relation


class Sentence:
    """
    Represents a sentence in a context window
    """
    def __init__(self, sent_id, line_num,raw_sent=None):
        self.sent_id = sent_id
        self.tokens = []
        self.line_no = line_num
        self.nlp_annotations = None
        self.raw_sent = raw_sent

    def add_token(self, token, start_char=None, end_char=None, tag=None, tag_id=None, root_id=None, relation=None):
        new_token = Token(token, start_char, end_char, tag, tag_id, root_id, relation)
        self.tokens.append(new_token)

    def len(self):
        return len(self.tokens)


class Context:
    """
    Represents a 3-sentence context window in which a definition may or may not occur
    """
    def __init__(self):
        self.sentences = []

    def add_sentence(self, sent):
        self.sentences.append(sent)

    def len(self):
        return len(self.sentences)


class File:
    """
    Represents a collection of definition contexts and their features
    """
    def __init__(self, filename):
        self.filename = filename
        self.contexts = []

    def add_context(self, context):
        self.contexts.append(context)


class Dataset:
    """
    Represents a collection of files in the corpus
    """
    def __init__(self):
        self.files = []
        self.max_sent_len = 0

    def add_file(self, file):
        self.files.append(file)
