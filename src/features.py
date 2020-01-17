from common_imports import *

class InputPrimitive(Enum):
    TOKEN = 1,
    POS = 2,
    POS_WPUNCT = 3,
    DEP = 4

def get_token(tokens, nlp_annotations):
    return [ele.token.lower() for ele in tokens]


def get_pos_with_punct(tokens, nlp_annotations):
    return [t.text if t.pos_ == 'PUNCT' else t.pos_ for t in nlp_annotations]


def get_pos(tokens, nlp_annotations):
    return [t.pos_ for t in nlp_annotations]


def get_dep(tokens, nlp_annotations):
    return [t.dep_ for t in nlp_annotations]


class Task1:
    @staticmethod
    def get_feature_map():
        return {'POS': get_pos,
                'TOKEN': get_token,
                'POS_WPUNCT': get_pos_with_punct,
                'DEP': get_dep}
