from common_imports import *

class InputPrimitive(Enum):
    TOKEN = 1,
    POS = 2,
    POS_WPUNCT = 3,
    DEP = 4


LOWERCASE_TOKENS = True
MIN_FREQ_COUNT = 2


# FIXME: We cannot assume that there will be 1-1 correspondence between tokens
# and NLP annotations, especially in the case of badly tokenized sentences; e.g:
# the number of tokens as read from corpus will not be equal to the number of tokens
# in the annotation object as the NLP is performed on a reconstructed sentence. If
# the individual tokens have sentence markers in them (in many cases, they do indeed),
# the number of tokens parsed by the NLP pipeline will be greater than the former.


def get_token(tokens, nlp_annotations, term_frequencies):
    out = []
    for t in nlp_annotations:
        to_add = t.text.lower() if LOWERCASE_TOKENS else t.text

        if t.pos_ == 'NUM':
            to_add = t.pos_
        elif term_frequencies[to_add] < MIN_FREQ_COUNT:
            to_add = t.pos_      # more informative than a placeholder

        out.append(to_add)

    return out

def get_pos_with_punct(tokens, nlp_annotations, term_frequencies):
    return [t.text if t.pos_ == 'PUNCT' else t.pos_ for t in nlp_annotations]


def get_pos(tokens, nlp_annotations, term_frequencies):
    return [t.pos_ for t in nlp_annotations]


def get_dep(tokens, nlp_annotations, term_frequencies):
    return [t.dep_ for t in nlp_annotations]


class Task1:
    @staticmethod
    def get_feature_map():
        return {'POS': get_pos,
                'TOKEN': get_token,
                'POS_WPUNCT': get_pos_with_punct,
                'DEP': get_dep}
