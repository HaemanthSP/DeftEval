from common_imports import *
import functools

class InputPrimitive(Enum):
    TOKEN = 1,
    POS = 2,
    POS_WPUNCT = 3,
    DEP = 4,
    HEAD = 5


LOWERCASE_TOKENS = True
MIN_FREQ_COUNT = 2


class Task1:
    @staticmethod
    def get_token(tokens, nlp_annotations, term_frequencies):
        out = []
        for t in nlp_annotations:
            to_add = t.text.lower() if LOWERCASE_TOKENS else t.text

            if t.pos_ == 'SPACE':
                continue

            if t.pos_ == 'NUM' or term_frequencies[to_add] < MIN_FREQ_COUNT:
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
                if head.pos_ == 'NUM' or term_frequencies[head] < MIN_FREQ_COUNT:
                    to_add = head.pos_      # more informative than a placeholder
                else:
                    to_add = str(head).lower()
            else:
                to_add = 'ROOT'
            out.append(to_add)
        return out


    @staticmethod
    def get_feature_input(sentence, primitive, term_frequencies):
        feature_map = {
            'POS': Task1.get_pos,
            'TOKEN': Task1.get_token,
            'POS_WPUNCT': Task1.get_pos_with_punct,
            'DEP': Task1.get_dep,
            'HEAD': Task1.get_head
        }

        return feature_map[primitive.name](sentence.tokens, sentence.nlp_annotations, term_frequencies)



