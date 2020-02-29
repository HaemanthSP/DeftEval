from common_imports import *
import functools
from nltk.corpus import wordnet


class InputPrimitive(Enum):
    TOKEN = 1,
    POS = 2,
    POS_WPUNCT = 3,
    DEP = 4,
    HEAD = 5,
    SYNSET = 6


LOWERCASE_TOKENS = True
MIN_FREQ_COUNT = 0
USE_FIRST_SYNSET = False


class Task1:
    @staticmethod
    def get_token(tokens, nlp_annotations, term_frequencies):
        out = []
        for t in nlp_annotations:
            to_add = t.text.lower() if LOWERCASE_TOKENS else t.text

            if t.pos_ == 'SPACE':
                continue

            if t.pos_ == 'NUM' or (to_add in term_frequencies and term_frequencies[to_add] < MIN_FREQ_COUNT):
                to_add = t.pos_      # more informative than a placeholder

            out.append(to_add)

        return out


    @staticmethod
    def get_pos_with_punct(tokens, nlp_annotations, term_frequencies):
        return [t.text if t.pos_ == 'PUNCT' else t.pos_ for t in nlp_annotations if t.pos_ != 'SPACE']


    @staticmethod
    def get_pos(tokens, nlp_annotations, term_frequencies):
        return [t.pos_ for t in nlp_annotations if t.pos_ != 'SPACE']


    @staticmethod
    def get_dep(tokens, nlp_annotations, term_frequencies):
        return [t.dep_ for t in nlp_annotations if t.pos_ != 'SPACE']


    @staticmethod
    def get_head(tokens, nlp_annotations, term_frequencies):
        out = []
        for t in nlp_annotations:
            ancestors = list(t.ancestors)
            if ancestors:
                head = ancestors[0] if LOWERCASE_TOKENS else ancestors[0]
                if head.pos_ == 'NUM' or (head in term_frequencies and term_frequencies[head] < MIN_FREQ_COUNT):
                    to_add = head.pos_      # more informative than a placeholder
                else:
                    to_add = str(head).lower()
            else:
                to_add = 'ROOT'
            out.append(to_add)
        return out


    @staticmethod
    def get_synset(tokens, nlp_annotations, term_frequencies):
        out = []
        for tok in nlp_annotations:
            if tok.pos_ == 'VERB':
                pos = wordnet.VERB
            elif tok.pos_ == 'NOUN':
                pos = wordnet.NOUN
            elif tok.pos_ == 'ADJ':
                pos = wordnet.ADJ
            elif tok.pos_ == 'ADV':
                pos = wordnet.ADV
            else:
                pos = None

            if pos is not None:
                synsets = wordnet.synsets(tok.text.lower(), pos=pos)
            else:
                synsets = wordnet.synsets(tok.text.lower())

            if len(synsets) > 1 and not USE_FIRST_SYNSET:
                out.append('<MULTIPLE_SYNSETS_' + str(len(synsets)) + '>')
            elif (len(synsets) >= 1 and USE_FIRST_SYNSET) or (len(synsets) == 1):
                out.append(synsets[0].name())
            else:
                out.append('<NO_SYNSET>')

        return out


    @staticmethod
    def get_feature_input(sentence, primitive, term_frequencies):
        feature_map = {
            'POS': Task1.get_pos,
            'TOKEN': Task1.get_token,
            'POS_WPUNCT': Task1.get_pos_with_punct,
            'DEP': Task1.get_dep,
            'HEAD': Task1.get_head,
            'SYNSET': Task1.get_synset
        }

        return feature_map[primitive.name](sentence.tokens, sentence.nlp_annotations, term_frequencies)


def get_oov_placeholder(primitive):
    placeholder_map = {
            'POS': '<UNK_POS>',
            'TOKEN': '<UNK_WORD_TOKEN>',
            'POS_WPUNCT': '<UNK_POS_WPUNCT>',
            'DEP': '<UNK_DEP>',
            'HEAD': '<UNK_HEAD>',
            'SYNSET': '<UNK_SYNSET>'
        }

    return placeholder_map[primitive.name]



