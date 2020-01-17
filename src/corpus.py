from common_imports import *

class Preprocessor:
    @staticmethod
    def replace_urls(text, replacement=' url '):
        return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+ ', replacement, text)

    @staticmethod
    def add_space_around(text, elements=r'([+\-\{\}\[\]\(\)=–])'):
        return re.sub(elements, r' \1 ', text)


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
        # Preprocessing text to limit the exploding vocabulary
        # clean_token = clean.replace_urls(new_token)  # Lots of URLs
        # clean_tokens = clean.add_space_around(clean_token).split()  # Replace improperly parsed words such as 2003).since link],consist 4-5
        # self.tokens.extend(clean_tokens)
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

def perform_nlp(dataset, dummy_data=False):
    for file in tqdm(dataset.files):
        for context in file.contexts:
            for sent in context.sentences:
                if dummy_data:
                    sent.nlp_annotations = []
                else:
                    raw_sent = ' '.join([ele.token for ele in sent.tokens])

                    # Preprocessing text to limit the exploding vocabulary
                    # clean_sent = clean.replace_urls(raw_sent)  # Lots of URLs
                    # clean_sent = clean.add_space_around(clean_sent)  # Replace improperly parsed words such as 2003).since link],consist 4-5
                    # sent.nlp_annotations = NLP(clean_sent, disable=['ner'])
                    sent.nlp_annotations = NLP(raw_sent, disable=['ner'])

    return dataset
