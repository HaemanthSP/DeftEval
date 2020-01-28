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


# FIXME: We cannot assume that there will be 1-1 correspondence between tokens
# and NLP annotations, especially in the case of badly tokenized sentences; e.g:
# the number of tokens as read from corpus will not be equal to the number of tokens
# in the annotation object as the NLP is performed on a reconstructed sentence. If
# the individual tokens have sentence markers in them (in many cases, they do indeed),
# the number of tokens parsed by the NLP pipeline will be greater than the former.
# This will be problematic during the sequence labelling tasks.


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


class Task2:
    SENTENCE_BEGIN_MARKER = "<sent_begin>"
    SENTENCE_END_MARKER = "<sent_end>"
    SENTENCE_MARKER_LABEL = ""


    @staticmethod
    def wrap_sentences(context, sent_functor, sent_begin_marker, sent_end_marker):
        out = []
        for sent in context.sentences:
            out.append(sent_begin_marker)
            out += sent_functor(sent)
            out.append(sent_end_marker)
        return out


    @staticmethod
    def __get_token(tokens, nlp_annotations, term_frequencies):
        """This be the primary feature untouched"""
        out = []
        for t in tokens:
            to_add = t.token.lower() if LOWERCASE_TOKENS else t.text

            # if t.pos_ == 'NUM' or term_frequencies[to_add] < MIN_FREQ_COUNT:
                # to_add = t.pos_      # more informative than a placeholder

            out.append(to_add)

        return out

    @staticmethod
    def __get_pos_with_punct(tokens, nlp_annotations, term_frequencies):
        return [t.text if t.pos_ == 'PUNCT' else t.pos_ for t in nlp_annotations if t.pos_ != 'SPACE']


    @staticmethod
    def __get_pos(tokens, nlp_annotations, term_frequencies):
        return [t.pos_ for t in nlp_annotations if t.pos_ != 'SPACE']


    @staticmethod
    def __get_dep(tokens, nlp_annotations, term_frequencies):
        return [t.dep_ for t in tokens]


    @staticmethod
    def __get_head(tokens, nlp_annotations, term_frequencies):
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
    def get_token(context, term_frequencies):
        return Task2.wrap_sentences(context,
            lambda sent, tf=term_frequencies: Task2.__get_token(sent.tokens, sent.nlp_annotations, tf),
            sent_begin_marker=Task2.SENTENCE_BEGIN_MARKER, sent_end_marker=Task2.SENTENCE_END_MARKER)


    @staticmethod
    def get_pos_with_punct(context, term_frequencies):
        return Task2.wrap_sentences(context,
            lambda sent, tf=term_frequencies: Task2.__get_pos_with_punct(sent.tokens, sent.nlp_annotations, tf),
            sent_begin_marker=Task2.SENTENCE_BEGIN_MARKER, sent_end_marker=Task2.SENTENCE_END_MARKER)


    @staticmethod
    def get_pos(context, term_frequencies):
        return Task2.wrap_sentences(context,
            lambda sent, tf=term_frequencies: Task2.__get_pos(sent.tokens, sent.nlp_annotations, tf),
            sent_begin_marker=Task2.SENTENCE_BEGIN_MARKER, sent_end_marker=Task2.SENTENCE_END_MARKER)


    @staticmethod
    def get_dep(context, term_frequencies):
        return Task2.wrap_sentences(context,
            lambda sent, tf=term_frequencies: Task2.__get_dep(sent.tokens, sent.nlp_annotations, tf),
            sent_begin_marker=Task2.SENTENCE_BEGIN_MARKER, sent_end_marker=Task2.SENTENCE_END_MARKER)


    @staticmethod
    def get_feature_input(context, primitive, term_frequencies):
        feature_map = {
            'POS': Task2.get_pos,
            'TOKEN': Task2.get_token,
            'POS_WPUNCT': Task2.get_pos_with_punct,
            'DEP': Task2.get_dep
        }

        return feature_map[primitive.name](context, term_frequencies)


    @staticmethod
    def get_labels(context):
        return Task2.wrap_sentences(context, lambda sent: [t.tag for t in sent.tokens], sent_begin_marker='O', sent_end_marker='O')




