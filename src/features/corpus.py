from pathlib import Path
from tqdm import tqdm

LOG_WARNINGS = False

# FIXME: Are tag and root IDs local to a specific context window?
# The training data seems to imply that since there _appear_ to be no
# sentences that refer to IDs outside of their context windows
# TODO: do a quick validation check after loading the corpus
# If so, would it make sense to encode it as a local feature?
# Having a global counter for each file would alternatively allow for
# greater flexibility, and the model should ultimately learn that the
# references are always local (if that does indeed turn out to be the case)
class Token:
    """
    Represents an individual token in a sentence
    """
    def __init__(self, features):
        assert len(features) == 8

        self.token = features[0]
        self.start_char = features[2]
        self.end_char = features[3]
        self.tag = features[4]
        self.tag_id = features[5]
        self.root_id = features[6]
        self.relation = features[7]


class Sentence:
    """
    Represents a sentence in a context window
    """
    def __init__(self, sent_id, line_num):
        self.sent_id = sent_id
        self.tokens = []
        self.line_no = line_num

    def add_token(self, features):
        new_token = Token(features)
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

    def add_file(self, file):
        self.files.append(file)


def load_files_into_dataset(dataset_path):
    dataset = Dataset()
    for data_file in tqdm(Path(dataset_path).iterdir()):
        if data_file.suffix == '.deft':
            with open(data_file, 'r', encoding='utf-8') as file:
                current_file = None
                current_sentence = None
                current_context = None
                next_sent_id = 0
                line_break_count = 0
                num_lines_to_skip = 0
                new_context = False
                new_sentence = False

                lines = file.readlines()
                for i in range(0, len(lines)):
                    if num_lines_to_skip > 0:
                        num_lines_to_skip -= 1
                        continue

                    first_previous_line = lines[i - 1] if i - 1 >= 0 else None
                    second_previous_line = lines[i - 2] if i - 2 >= 0 else None
                    current_line = lines[i]
                    current_line_no = i + 1
                    first_next_line = lines[i + 1] if i + 1 < len(lines) else None
                    second_next_line = lines[i + 2] if i + 2 < len(lines) else None

                    splits = [x.strip() for x in current_line.split('\t')]

                    # attempt to detect and recover from malformed lines
                    if len(splits) > 0:
                        try:
                            dummy = int(splits[0])
                            first_token_is_int = True
                        except ValueError:
                            first_token_is_int = False

                        # check if potential context windows are correctly delimited
                        # 1: .  data/source_txt/train/t1_biology_0_101.txt	 24472	 24473	 O	 -1	 -1	 0
                        # 2:
                        # 3: 509	data/source_txt/train/t1_biology_0_101.txt	 24469	 24472	 O	 -1	 -1	 0
                        # 4: .	data/source_txt/train/t1_biology_0_101.txt	 24472	 24473	 O	 -1	 -1	 0
                        # 5:
                        # 6: ...

                        if first_token_is_int:
                            first_next_line_splits = [x.strip() for x in first_next_line.split('\t')] if first_next_line != None else []
                            first_token_first_next_line = first_next_line_splits[0] if len(first_next_line_splits) > 0 else None

                            if first_token_first_next_line == '.' and second_next_line == '\n':
                                if first_previous_line == '\n' and second_previous_line != '\n':
                                    if LOG_WARNINGS:
                                        print("Potential missing line-break on line %d in file %s" % (current_line_no, str(data_file)))

                                    new_context = True

                    if new_context or current_line_no == 1:
                        if current_file == None:
                            current_file = File(str(data_file))

                        if current_context != None:
                            assert current_file != None
                            current_file.add_context(current_context)

                        current_context = Context()

                        if second_next_line != '\n':
                            if LOG_WARNINGS:
                                print("Malformed context window separator on line %d in file %s" % (current_line_no + 2, str(data_file)))

                            # skip until the next line break
                            num_lines_to_skip = 0
                            for k in range(i + 1, len(lines)):
                                if lines[k] != '\n':
                                    num_lines_to_skip += 1
                                else:
                                    break
                        else:
                            num_lines_to_skip = 1

                        line_break_count = 0
                        next_sent_id = int(splits[0]) - 1

                        new_context = False
                        continue
                    elif new_sentence:
                        if current_sentence != None:
                            assert current_context != None

                            if current_context.len() >= 3 and LOG_WARNINGS:
                                print("Extra sentence on line %d in file %s" % (current_line_no, str(data_file)))

                            if current_sentence.len() != 0:
                                current_context.add_sentence(current_sentence)

                                if current_sentence.len() < 3 and LOG_WARNINGS:
                                    print("Suspiciously short sentence on line %d in file %s" % (current_sentence.line_no, str(data_file)))

                        next_sent_id += 1
                        current_sentence = Sentence(next_sent_id, current_line_no)

                        new_sentence = False

                    if current_line == '\n':
                        line_break_count += 1
                    else:
                        line_break_count = 0

                    if line_break_count == 1:
                        new_sentence = True
                    elif line_break_count == 2:
                        new_context = True
                    else:
                        assert current_sentence != None
                        current_sentence.add_token(splits)

                if current_sentence.len() > 0:
                    current_context.add_sentence(current_sentence)

                current_file.add_context(current_context)
                dataset.add_file(current_file)

    return dataset

