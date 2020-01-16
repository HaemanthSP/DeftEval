from common_imports import *
from corpus import *
import features
from util import Numberer

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


class Common:
    LOG_WARNINGS = False

    @staticmethod
    def load_training_data(dataset_path):
        dataset = Dataset()
        data_files = []
        for file in Path(dataset_path).iterdir():
            if file.suffix == '.deft':
                data_files.append(file)

        for data_file in tqdm(data_files):
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
                                        if Common.LOG_WARNINGS:
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
                                if Common.LOG_WARNINGS:
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

                                if current_context.len() >= 3 and Common.LOG_WARNINGS:
                                    print("Extra sentence on line %d in file %s" % (current_line_no, str(data_file)))

                                if current_sentence.len() != 0:
                                    dataset.max_sent_len = max(dataset.max_sent_len, current_sentence.len())
                                    current_context.add_sentence(current_sentence)

                                    if current_sentence.len() < 3 and Common.LOG_WARNINGS:
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
                            current_sentence.add_token(
                                splits[0], splits[2], splits[3], splits[4], splits[5], splits[6], splits[7])

                    if current_sentence.len() > 0:
                        dataset.max_sent_len = max(dataset.max_sent_len, current_sentence.len())
                        current_context.add_sentence(current_sentence)

                    current_file.add_context(current_context)
                    dataset.add_file(current_file)

        return dataset


    @staticmethod
    def perform_nlp(dataset, dummy_data=False):
        for file in tqdm(dataset.files):
            for context in file.contexts:
                for sent in context.sentences:
                    if dummy_data:
                        sent.nlp_annotations = []
                    else:
                        raw_sent = ' '.join([ele.token for ele in sent.tokens])

                        # TODO: Preprocessing text to limit the exploding vocabulary
                        # clean_sent = clean.replace_urls(raw_sent)  # Lots of URLs
                        # clean_sent = clean.add_space_around(clean_sent)  # Replace improperly parsed words such as 2003).since link],consist 4-5
                        # sent.nlp_annotations = NLP(clean_sent, disable=['ner'])
                        sent.nlp_annotations = NLP(raw_sent, disable=['ner'])


# Evaluation data format:
# https://groups.google.com/forum/#!topic/semeval-2020-task-6-all/JsmVmPrycfQ

class Task1:
    @staticmethod
    def load_evaluation_data(dataset_path):
        dataset = Dataset()
        labels = []

        data_files = []
        for file in Path(dataset_path).iterdir():
            if file.suffix == '.deft':
                data_files.append(file)

        # Use the default spacy tokenizer
        # TODO: Check if the default tokenizer rules roughly correspond to the ones used in the training set
        nlp = English()
        tokenizer = nlp.Defaults.create_tokenizer(nlp)

        for data_file in tqdm(data_files):
            if data_file.suffix == '.deft':
                file = File(str(data_file))
                context = Context()

                with open(data_file, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    for i in range(0, len(lines)):
                        current_line = lines[i]
                        splits = current_line.split('\t')
                        assert len(splits) == 2

                        sentence = re.sub('\"(.+)\"', r'\1', splits[0])
                        label = 1 if splits[1] == '"1"' else 0

                        sent_wrapper = Sentence(sent_id=i,line_num=i)
                        for token in tokenizer(sentence):
                            # Flag all tokens as being inside a definition when the sentence has a positive label
                            sent_wrapper.add_token(token=token.text, tag='I-Definition' if label == 1 else 'B-Random')

                        context.add_sentence(sent_wrapper)
                        labels.append(label)

                file.add_context(context)
                dataset.add_file(file)

        return dataset


    @staticmethod
    def generate_primitives_and_vocabulary(dataset, input_primitives, x, y, vocabulary_set):
        feature_map = features.Task1.get_feature_map()

        for file in tqdm(dataset.files):
            for context in file.contexts:
                for sent in context.sentences:
                    feature_inputs = []
                    for idx, primitive in enumerate(input_primitives):
                        feature_input = feature_map[primitive.name](sent.tokens, sent.nlp_annotations)
                        vocabulary_set[idx].update(feature_input)
                        feature_inputs.append(feature_input)

                    label = 0
                    for token in sent.tokens:
                        if token.tag[2:] == 'Definition':
                            label = 1
                            break

                    x.append(feature_inputs)
                    y.append(label)


    @staticmethod
    def encode_primitives(x, encoders, shapes, add_if_absent):
        # store the encoded primitives as a byte string to keep the tensor length fixed
        # individual features can be extracted using a map operation on the generated tf.data.Dataset object
        # the alternative would be to pad it to the maximum sentence length in advance
        for row_idx, row in enumerate(tqdm(x)):
            new_feature_arrays = [np.zeros(shape, dtype=np.int32)
                                  for shape in shapes]

            for idx, feature_input in enumerate(row):
                for primitive_idx, primitive in enumerate(feature_input):
                    new_feature_arrays[idx][primitive_idx] = encoders[idx].number(primitive, add_if_absent)

            x[row_idx] = {}
            for idx, feat in enumerate(new_feature_arrays, 1):
                x[row_idx].update({"Feature_%s" %(idx): feat})


    @staticmethod
    def generate_model_train_inputs(train_dataset, input_primitives, feature_vector_length):
        x_train = []
        y_train = []
        combined_vocabs = [set() for i in input_primitives]

        print("[TRAIN] Generating primitives and constructing vocabulary")
        Task1.generate_primitives_and_vocabulary(train_dataset, input_primitives, x_train, y_train, combined_vocabs)

        print("[TRAIN] Encoding primitives")
        encoders = [Numberer(vocab) for vocab in combined_vocabs]
        # For now all features are padded with same length
        # TODO: Make custom padding length for individual features, account for OOV primitives in the test set
        feature_vector_shapes = [feature_vector_length] * len(input_primitives)
        Task1.encode_primitives(x_train, encoders, feature_vector_shapes, add_if_absent=False)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train, dtype=np.int8)

        def train_generator():
            for x, y in zip(x_train, y_train):
                yield x, y

        types = {"Feature_"+str(i+1): tf.int32 for i, _ in enumerate(input_primitives)}, tf.int8
        shapes = {"Feature_"+str(i+1): tf.TensorShape([None,]) for i, _ in enumerate(input_primitives)}, tf.TensorShape([])
        train_dataset = tf.data.Dataset.from_generator(train_generator, types, shapes)

        return train_dataset, combined_vocabs, encoders


    @staticmethod
    def generate_model_test_inputs(test_dataset, input_primitives, encoders, combined_vocabs, feature_vector_length):
        x_test = []
        y_test = []

        print("Generating primitives and constructing vocabulary")
        # FIXME: Should we update the encoder/vocab here with new IDs for test set primitives that are OOV in the train set?
        Task1.generate_primitives_and_vocabulary(test_dataset, input_primitives, x_test, y_test, combined_vocabs)

        print("Encoding primitives")
        # For now all features are padded with same length
        # TODO: Make custom padding length for individual features, account for OOV primitives in the test set
        feature_vector_shapes = [feature_vector_length] * len(input_primitives)
        Task1.encode_primitives(x_test, encoders, feature_vector_shapes, add_if_absent=False)

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test, dtype=np.int8)

        def test_generator():
            for x, y in zip(x_test, y_test):
                yield x, y

        types = {"Feature_"+str(i+1): tf.int32 for i, _ in enumerate(input_primitives)}, tf.int8
        shapes = {"Feature_"+str(i+1): tf.TensorShape([None,]) for i, _ in enumerate(input_primitives)}, tf.TensorShape([])
        test_dataset = tf.data.Dataset.from_generator(test_generator, types, shapes)

        return test_dataset












