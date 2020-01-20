from common_imports import *
import corpus, features
from util import Numberer, Preprocessor

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


class Common:
    LOG_WARNINGS = False
    SKIP_NLP = False

    # Initialized after dependencies are resolved
    TASK_REGISTRY = {}


    @staticmethod
    def load_deft_data(task, dataset_path, evaluation_data=False):
        dataset = corpus.Dataset()
        data_files = []
        for file in Path(dataset_path).iterdir():
            if file.suffix == '.deft':
                data_files.append(file)

        # Use the default spacy tokenizer
        # TODO: Check if the default tokenizer rules roughly correspond to the ones used in the training set
        if evaluation_data and task == Task.TASK_1:
            nlp = English()
            tokenizer = nlp.Defaults.create_tokenizer(nlp)

        for data_file in tqdm(data_files):
            if data_file.suffix == '.deft':
                with open(data_file, 'r', encoding='utf-8') as file:
                    current_file = corpus.File(str(data_file))
                    current_sentence = corpus.Sentence(sent_id=1, line_num=1) if evaluation_data and task == Task.TASK_2 else None
                    current_context = corpus.Context() if evaluation_data and task != Task.TASK_3 else None
                    next_sent_id = 0
                    line_break_count = 0
                    num_lines_to_skip = 0
                    new_context = True if not evaluation_data or task == Task.TASK_3 else False
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

                        if evaluation_data:
                            """
                            Evaluation data format:
                            https://groups.google.com/forum/#!topic/semeval-2020-task-6-all/JsmVmPrycfQ

                            Input:
                            Subtask 1: Sentences
                            Subtask 2: [TOKEN] [SOURCE] [START_CHAR] [END_CHAR]
                            Subtask 3: [TOKEN] [SOURCE] [START_CHAR] [END_CHAR] [TAG] [TAG_ID]

                            Output:
                            Subtask 1: Sentence classification label (0 or 1)
                            Subtask 2: [TAG]
                            Subtask 3: [ROOT_ID] [RELATION]
                            """

                            # Task 1 needs special casing, the rest share the same parsing routine as the training data
                            if task == Task.TASK_1:
                                # FIXME: The first token is assigned as sentence. This seems to be the cause for the issue.
                                sentence = Preprocessor.remove_quotes(splits[0])
                                sent_wrapper = corpus.Sentence(sent_id=i,line_num=i,raw_sent=sentence)
                                for token in tokenizer(sentence):
                                    sent_wrapper.add_token(token=token.text)

                                current_context.add_sentence(sent_wrapper)
                                continue

                        if len(splits) > 0:
                            try:
                                dummy = int(splits[0])
                                first_token_is_int = True
                            except ValueError:
                                first_token_is_int = False

                            if first_token_is_int:
                                # check if potential context windows are correctly delimited
                                first_next_line_splits = [x.strip() for x in first_next_line.split('\t')] if first_next_line != None else []
                                first_token_first_next_line = first_next_line_splits[0] if len(first_next_line_splits) > 0 else None

                                if first_token_first_next_line == '.' and second_next_line == '\n':
                                    if first_previous_line == '\n' and second_previous_line != '\n':
                                        if Common.LOG_WARNINGS:
                                            print("Potential missing line-break on line %d in file %s" % (current_line_no, str(data_file)))

                                        new_context = True

                        if new_context:
                            if current_context != None:
                                current_file.add_context(current_context)

                            current_context = corpus.Context()

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

                                if current_context.len() >= 3 and not evaluation_data and Common.LOG_WARNINGS:
                                    print("Extra sentence on line %d in file %s" % (current_line_no, str(data_file)))

                                if current_sentence.len() != 0:
                                    dataset.max_sent_len = max(dataset.max_sent_len, current_sentence.len())
                                    current_context.add_sentence(current_sentence)

                                    if current_sentence.len() < 3 and Common.LOG_WARNINGS:
                                        print("Suspiciously short sentence on line %d in file %s" % (current_sentence.line_no, str(data_file)))

                            next_sent_id += 1
                            current_sentence = corpus.Sentence(next_sent_id, current_line_no)

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

                            # the latter annotations are only available in the train/dev sets
                            current_sentence.add_token(
                                token=splits[0],
                                start_char=splits[2],
                                end_char=splits[3],
                                tag=splits[4] if len(splits) > 4 else None,
                                tag_id=splits[5] if len(splits) > 5 else None,
                                root_id=splits[6] if len(splits) > 6 else None,
                                relation=splits[7] if len(splits) > 7 else None)

                    if current_sentence != None and current_sentence.len() > 0:
                        dataset.max_sent_len = max(dataset.max_sent_len, current_sentence.len())
                        current_context.add_sentence(current_sentence)

                    current_file.add_context(current_context)
                    dataset.add_file(current_file)

        return dataset


    @staticmethod
    def do_basic_preprocessing(sentence_str):
        preprocessed_sent = Preprocessor.replace_urls(sentence_str)

        return preprocessed_sent


    @staticmethod
    def preprocess_dataset(task, dataset, dummy_data=False):
        for file in tqdm(dataset.files):
            for context in file.contexts:
                for sent in context.sentences:
                    preprocessed_sent = Common.TASK_REGISTRY[task].preprocess_sentence(sent)
                    if Common.SKIP_NLP:
                        sent.nlp_annotations = []
                    else:
                        sent.nlp_annotations = NLP(preprocessed_sent, disable=['ner'])

                    for token in sent.nlp_annotations:
                        if features.LOWERCASE_TOKENS:
                            dataset.term_frequencies[token.text.lower()] += 1
                        else:
                            dataset.term_frequencies[token.text] += 1


class Task1:
    @staticmethod
    def load_evaluation_data(dataset_path):
        return Common.load_deft_data(Task.TASK_1, dataset_path, evaluation_data=True)


    @staticmethod
    def preprocess_sentence(sentence):
        raw_sent = ' '.join([ele.token for ele in sentence.tokens])
        preprocessed_sent = Common.do_basic_preprocessing(raw_sent)
        return Preprocessor.add_space_around(preprocessed_sent)  # Replace improperly parsed words such as 2003)


    @staticmethod
    def generate_primitives_and_vocabulary(dataset, input_primitives, x, y, vocabulary_set, metadata=[]):
        for file in tqdm(dataset.files):
            for context in file.contexts:
                for sent in context.sentences:
                    feature_inputs = []
                    for idx, primitive in enumerate(input_primitives):
                        feature_input = features.Task1.get_feature_input(
                            sent, primitive, dataset.term_frequencies)
                        vocabulary_set[idx].update(feature_input)
                        feature_inputs.append(feature_input)

                    label = 0
                    for token in sent.tokens:
                        if token.tag != None and token.tag[2:] == 'Definition':
                            label = 1
                            break

                    x.append(feature_inputs)
                    y.append(label)

                    # add metadata specific to each instance for use during evaluation
                    metadata.append((file, context, sent, label))


    @staticmethod
    def encode_primitives(x, encoders, shapes, add_if_absent):
        for row_idx, row in enumerate(tqdm(x)):
            new_feature_arrays = [np.zeros(shape, dtype=np.int32)
                                  for shape in shapes]

            for idx, feature_input in enumerate(row):
                for primitive_idx, primitive in enumerate(feature_input):
                    if primitive_idx >= shapes[0]:
                        # Can occur if the NLP pipeline identifies more
                        # tokens in the reconstructed sentence
                        break

                    new_feature_arrays[idx][primitive_idx] = encoders[idx].number(primitive, add_if_absent)

            x[row_idx] = {}
            for idx, feat in enumerate(new_feature_arrays, 1):
                x[row_idx].update({"Feature_%s" %(idx): feat})


    @staticmethod
    def generate_model_train_inputs(train_dataset, input_primitives, feature_vector_length):
        x_train = []
        y_train = []
        combined_vocabs = [set() for i in input_primitives]

        print("Generating primitives and constructing vocabulary")
        Task1.generate_primitives_and_vocabulary(train_dataset, input_primitives, x_train, y_train, combined_vocabs)

        print("Encoding primitives")
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
        test_metadata = []

        print("Generating primitives and constructing vocabulary")
        # FIXME: Should we update the encoder/vocab here with new IDs for test set primitives that are OOV in the train set?
        Task1.generate_primitives_and_vocabulary(test_dataset, input_primitives, x_test, y_test, combined_vocabs, test_metadata)

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

        return test_dataset, test_metadata


class Task2:
    @staticmethod
    def load_evaluation_data(dataset_path):
        dataset = Common.load_deft_data(Task.TASK_2, dataset_path, evaluation_data=True)
        # move each sentence into its own context to prevent them being
        # converted into a single prediction instance down the line
        # FIXME: ensure that the actual evaluation data has no context information for this task
        for file in dataset.files:
            assert len(file.contexts) == 1
            default_context = file.contexts[0]

            file.contexts = []
            for sent in default_context.sentences:
                new_context = corpus.Context()
                new_context.add_sentence(sent)
                file.add_context(new_context)

        return dataset


    @staticmethod
    def preprocess_sentence(sentence):
        raw_sent = ' '.join([ele.token for ele in sentence.tokens])
        return Common.do_basic_preprocessing(raw_sent)


    @staticmethod
    def generate_primitives_and_vocabulary(dataset, input_primitives, x, y, vocab_x, vocab_y, metadata=[]):
        longest_input_size = 0
        for file in tqdm(dataset.files):
            # concatenate all sentences in single context into a single data point
            for context in file.contexts:
                feature_inputs = []
                labels = features.Task2.get_labels(context)
                vocab_y.update(labels)

                for idx, primitive in enumerate(input_primitives):
                    feature_input = features.Task2.get_feature_input(context, primitive, dataset.term_frequencies)
                    if len(feature_input) != len(labels):
                        print("Length: %s, %s" % (len(feature_input), len(labels)))
                        print(feature_input)
                        print([t.token for sent in context.sentences for t in sent.tokens])
                    assert len(feature_input) == len(labels)

                    vocab_x[idx].update(feature_input)
                    feature_inputs.append(feature_input)

                    if longest_input_size < len(feature_input):
                        longest_input_size = len(feature_input)

                x.append(feature_inputs)
                y.append(labels)

                metadata.append((file, context, labels))

        return longest_input_size


    @staticmethod
    def encode_primitives(x, y, encoder_x, encoder_y, shapes, add_if_absent):
        for row_idx, row in enumerate(tqdm(x)):
            new_feature_arrays = [np.zeros(shape, dtype=np.int32)
                                  for shape in shapes]

            for idx, feature_input in enumerate(row):
                for primitive_idx, primitive in enumerate(feature_input):
                    new_feature_arrays[idx][primitive_idx] = encoder_x[idx].number(primitive, add_if_absent)

            x[row_idx] = {}
            for idx, feat in enumerate(new_feature_arrays, 1):
                x[row_idx].update({"Feature_%s" %(idx): feat})

        for row_idx, row in enumerate(tqdm(y)):
            for idx, label in enumerate(row):
                y[idx] = encoder_y.number(label, add_if_absent)


    @staticmethod
    def generate_model_train_inputs(train_dataset, input_primitives, feature_vector_length=-1):
        x_train = []
        y_train = []
        vocab_x = [set() for i in input_primitives]
        vocab_y = set()

        print("Generating primitives and constructing vocabulary")
        max_feature_vector_length = Task2.generate_primitives_and_vocabulary(train_dataset, input_primitives, x_train, y_train, vocab_x, vocab_y)

        print("Encoding primitives")
        encoder_x = [Numberer(vocab) for vocab in vocab_x]
        encoder_y = Numberer(vocab_y)
        if feature_vector_length == -1:
            feature_vector_length = max_feature_vector_length

        feature_vector_shapes = [feature_vector_length] * len(input_primitives)
        Task2.encode_primitives(x_train, y_train, encoder_x, encoder_y, feature_vector_shapes, add_if_absent=False)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train, dtype=np.int8)

        def train_generator():
            for x, y in zip(x_train, y_train):
                yield x, y

        types = {"Feature_"+str(i+1): tf.int32 for i, _ in enumerate(input_primitives)}, tf.int8
        shapes = {"Feature_"+str(i+1): tf.TensorShape([None,]) for i, _ in enumerate(input_primitives)}, tf.TensorShape([])
        train_dataset = tf.data.Dataset.from_generator(train_generator, types, shapes)

        return train_dataset, vocab_x, vocab_y, encoder_x, encoder_y, feature_vector_length


    @staticmethod
    def generate_model_test_inputs(test_dataset, input_primitives, vocab_x, vocab_y, encoder_x, encoder_y, feature_vector_length):
        x_test = []
        y_test = []
        test_metadata = []

        print("Generating primitives and constructing vocabulary")
        max_feature_vector_length = Task2.generate_primitives_and_vocabulary(test_dataset, input_primitives, x_test, y_test, vocab_x, vocab_y, test_metadata)
        # FIXME: what happens when the test set has a sentence longer than all the sents in the train set,
        # i.e., when feature_vector_length=max_feature_vector_length in train set?
        # it'll get truncated presumably, and so will the output tag sequence
        if max_feature_vector_length > feature_vector_length:
            print("WARNING: Evaluation data set has sentences that exceed the max. sentence length (%d > %d)" % (max_feature_vector_length, feature_vector_length))

        print("Encoding primitives")
        feature_vector_shapes = [feature_vector_length] * len(input_primitives)
        Task2.encode_primitives(x_test, y_test, encoder_x, encoder_y, feature_vector_shapes, add_if_absent=False)

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test, dtype=np.int8)

        def test_generator():
            for x, y in zip(x_test, y_test):
                yield x, y

        types = {"Feature_"+str(i+1): tf.int32 for i, _ in enumerate(input_primitives)}, tf.int8
        shapes = {"Feature_"+str(i+1): tf.TensorShape([None,]) for i, _ in enumerate(input_primitives)}, tf.TensorShape([])
        test_dataset = tf.data.Dataset.from_generator(test_generator, types, shapes)

        return test_dataset, test_metadata



class Task3:
    @staticmethod
    def load_evaluation_data(dataset_path):
        return Common.load_deft_data(Task.TASK_3, dataset_path, evaluation_data=True)


    @staticmethod
    def preprocess_sentence(sentence):
        raw_sent = ' '.join([ele.token for ele in sentence.tokens])
        return Common.do_basic_preprocessing(raw_sent)



# Deferred init.
Common.TASK_REGISTRY = {
    Task.TASK_1 : Task1,
    Task.TASK_2 : Task2,
    Task.TASK_3 : Task3,
}
