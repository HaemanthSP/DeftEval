from common_imports import *
import corpus, features
from util import Numberer, Preprocessor
import random
import functools
from collections import Counter

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


spacy.prefer_gpu()
NLP = spacy.load("en_core_web_lg")


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
                    lines_start_idx = 0
                    if len(lines) > 0 and lines[0] == '\n':
                        lines_start_idx = 1

                    for i in range(lines_start_idx, len(lines)):
                        if num_lines_to_skip > 0:
                            num_lines_to_skip -= 1
                            continue

                        first_previous_line = lines[i - 1] if i - 1 >= 0 else None
                        second_previous_line = lines[i - 2] if i - 2 >= 0 else None
                        current_line = lines[i]
                        current_line_no = i + 1
                        first_next_line = lines[i + 1] if i + 1 < len(lines) else None
                        second_next_line = lines[i + 2] if i + 2 < len(lines) else None

                        if current_line_no == 1 and current_line == '\n':
                            continue

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
                                filename=splits[1],
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
    def preprocess_dataset(task, dataset, use_dummy_nlp_annotations=False):
        def perform_nlp(source_sent, text):
            def default_tokenizer(source_sent, text):
                # just use the data pre-tokenized data
                return spacy.tokens.Doc(NLP.vocab, [t.token for t in source_sent.tokens])

            tokenizer_fn_wrapper = functools.partial(default_tokenizer, sent)
            NLP.tokenizer = tokenizer_fn_wrapper
            return NLP(text, disable=['ner'])


        for file in tqdm(dataset.files):
            for context in file.contexts:
                for sent in context.sentences:
                    preprocessed_sent = Common.TASK_REGISTRY[task].preprocess_sentence(sent)
                    if Common.SKIP_NLP or use_dummy_nlp_annotations:
                        sent.nlp_annotations = []
                    else:
                        sent.nlp_annotations = perform_nlp(sent, preprocessed_sent)

                    for token in sent.nlp_annotations:
                        if features.LOWERCASE_TOKENS:
                            dataset.term_frequencies[token.text.lower()] += 1
                        else:
                            dataset.term_frequencies[token.text] += 1


    @staticmethod
    def train_val_split(x_data, y_data, val_take_size):
        combined = list(zip(x_data, y_data))
        random.shuffle(combined)
        assert len(combined) > val_take_size

        train_combined = combined[:-val_take_size]
        val_combined = combined[-val_take_size:]

        train_unzipped = list(zip(*train_combined))
        train_x, train_y = train_unzipped[0], train_unzipped[1]
        val_unzipped = list(zip(*val_combined))
        val_x, val_y = val_unzipped[0], val_unzipped[1]

        return train_x, train_y, val_x, val_y


# All members are tuples that hold the values that correspond to the input and the output, i.e., x & y
# Test metadata is just a list of tuples with extra data for each test instance/data point.
class TrainMetadata:
    def __init__(self, encoders, vocabs, vocab_sizes, train_data_class_dist):
        self.encoders = encoders
        self.vocabs = vocabs
        self.vocab_sizes = vocab_sizes
        self.train_data_class_dist = train_data_class_dist


class ClassDistribution:
    def __init__(self, dist_map):
        self.class_dists = dist_map
        self.total_instances = functools.reduce(lambda a, b: a + b, self.class_dists.values())

    def calculate_class_weights(self, weight_fn):
        return { k: weight_fn(v, self.total_instances) for k,v in self.class_dists.items() }



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
    def calculate_class_distribution(y_data):
        class_dists = Counter()
        for y in y_data:
            class_dists[y] += 1
        return ClassDistribution(dict(class_dists))


    @staticmethod
    def create_tf_dataset(x_data, y_data, input_primitives):
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data, dtype=np.int8)

        def generator():
            for x, y in zip(x_data, y_data):
                yield x, y

        types = {"Feature_"+str(i+1): tf.int32 for i, _ in enumerate(input_primitives)}, tf.int8
        shapes = {"Feature_"+str(i+1): tf.TensorShape([None,]) for i, _ in enumerate(input_primitives)}, tf.TensorShape([])
        return tf.data.Dataset.from_generator(generator, types, shapes)


    @staticmethod
    def generate_model_train_inputs(train_dataset, input_primitives, feature_vector_length, valid_dataset_take_size):
        x_train = []
        y_train = []
        combined_vocabs = [set() for i in input_primitives]

        print("Generating primitives and constructing vocabulary")
        Task1.generate_primitives_and_vocabulary(train_dataset, input_primitives, x_train, y_train, combined_vocabs)

        print("Encoding primitives")
        encoders = [Numberer(vocab) for vocab in combined_vocabs]
        feature_vector_shapes = [feature_vector_length] * len(input_primitives)
        Task1.encode_primitives(x_train, encoders, feature_vector_shapes, add_if_absent=False)

        x_train, y_train, x_val, y_val = Common.train_val_split(x_train, y_train, valid_dataset_take_size)

        train_dataset = Task1.create_tf_dataset(x_train, y_train, input_primitives)
        val_dataset = Task1.create_tf_dataset(x_val, y_val, input_primitives)
        train_class_dist = Task1.calculate_class_distribution(y_train)

        # pack the vocabs and encoders in a tuple to keep the format the same between tasks
        return train_dataset, val_dataset, (combined_vocabs, ''), (encoders, ''), train_class_dist


    @staticmethod
    def generate_model_test_inputs(test_dataset, input_primitives, encoders, combined_vocabs, feature_vector_length):
        x_test = []
        y_test = []
        test_metadata = []

        # unpack tuples
        encoders = encoders[0]
        combined_vocabs = combined_vocabs[0]

        print("Generating primitives and constructing vocabulary")
        Task1.generate_primitives_and_vocabulary(test_dataset, input_primitives, x_test, y_test, combined_vocabs, test_metadata)

        print("Encoding primitives")
        feature_vector_shapes = [feature_vector_length] * len(input_primitives)
        Task1.encode_primitives(x_test, encoders, feature_vector_shapes, add_if_absent=False)

        test_dataset = Task1.create_tf_dataset(x_test, y_test, input_primitives)
        for idx, vocab in enumerate(combined_vocabs):
            print("Vocabulary Size of feat %s: %s" % (idx, len(vocab)))
            print("Random sample: %s" % (str(random.sample(vocab, min(len(vocab), 150)))))

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
    def encode_primitives(x, y, encoder_x, encoder_y, shapes, num_tags, add_if_absent):
        for row_idx, row in enumerate(tqdm(x)):
            new_feature_arrays = [np.zeros(shape, dtype=np.int32)
                                  for shape in shapes]

            for idx, feature_input in enumerate(row):
                for primitive_idx, primitive in enumerate(feature_input):
                    if primitive_idx >= shapes[0]:
                        break

                    new_feature_arrays[idx][primitive_idx] = encoder_x[idx].number(primitive, add_if_absent)

            x[row_idx] = {}
            for idx, feat in enumerate(new_feature_arrays, 1):
                x[row_idx].update({"Feature_%s" %(idx): feat})

        for row_idx, row in enumerate(tqdm(y)):
            new_label_array = np.zeros([shapes[0], num_tags + 1], dtype=np.int8)
            for idx, label in enumerate(row):
                tag_id = encoder_y.number(label, add_if_absent)
                assert tag_id != 0
                new_label_array[idx][tag_id] = 1

            y[row_idx] = new_label_array


    @staticmethod
    def calculate_class_distribution(y_data):
        class_dists = Counter()
        for y in y_data:
            for timestep in y:
                class_id = int(np.argmax(timestep))
                if class_id != 0:
                    class_dists[class_id] += 1
        return ClassDistribution(dict(class_dists))


    @staticmethod
    def create_tf_dataset(x_data, y_data, input_primitives):
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        def generator():
            for x, y in zip(x_data, y_data):
                yield x, y

        types = {"Feature_"+str(i+1): tf.int32 for i, _ in enumerate(input_primitives)}, tf.float32
        shapes = {"Feature_"+str(i+1): tf.TensorShape([None,]) for i, _ in enumerate(input_primitives)}, tf.TensorShape([None,None])
        return tf.data.Dataset.from_generator(generator, types, shapes)


    @staticmethod
    def create_ner_model_dataset(x, y, out_folder, filename):
        vocab_x = set()
        vocab_y = set()

        with open(out_folder + '/' + filename + '.words.txt', 'w') as words_file:
            with open(out_folder + '/' + filename + '.tags.txt', 'w') as tags_file:
                for i in range(0, len(x)):
                    seq = x[i][0]
                    tags = y[i]

                    vocab_x.update(set(seq))
                    vocab_y.update(set(tags))

                    words_file.write(' '.join(seq).strip() + '\n')
                    tags_file.write(' '.join(tags).strip() + '\n')

        with open(out_folder + '/' + filename + '.vocab.words.txt', 'w') as f:
            for i in vocab_x:
                f.write(str(i) + '\n')

        with open(out_folder + '/' + filename + '.vocab.tags.txt', 'w') as f:
            for i in vocab_y:
                f.write(str(i) + '\n')


    @staticmethod
    def generate_model_train_inputs(train_dataset, input_primitives, feature_vector_length, valid_dataset_take_size):
        x_train = []
        y_train = []
        vocab_x = [set() for i in input_primitives]
        vocab_y = set()

        print("Generating primitives and constructing vocabulary")
        max_feature_vector_length = Task2.generate_primitives_and_vocabulary(train_dataset, input_primitives, x_train, y_train, vocab_x, vocab_y)
        Task2.create_ner_model_dataset(x_train, y_train, '../ner_dataset/', 'train')

        print("Encoding primitives")
        encoder_x = [Numberer(vocab) for vocab in vocab_x]
        encoder_y = Numberer(vocab_y)
        if max_feature_vector_length > feature_vector_length:
            print("WARNING: Training data set has sentences that exceed the max. feature vec length (%d > %d)" % (max_feature_vector_length, feature_vector_length))

        feature_vector_shapes = [feature_vector_length] * len(input_primitives)
        Task2.encode_primitives(x_train, y_train, encoder_x, encoder_y,
                                feature_vector_shapes, len(vocab_y), add_if_absent=False)

        x_train, y_train, x_val, y_val = Common.train_val_split(x_train, y_train, valid_dataset_take_size)

        train_dataset = Task2.create_tf_dataset(x_train, y_train, input_primitives)
        val_dataset = Task2.create_tf_dataset(x_val, y_val, input_primitives)
        train_class_dist = Task2.calculate_class_distribution(y_train)

        return train_dataset, val_dataset, (vocab_x, vocab_y), (encoder_x, encoder_y), train_class_dist


    @staticmethod
    def generate_model_test_inputs(test_dataset, input_primitives, encoders, combined_vocabs, feature_vector_length):
        x_test = []
        y_test = []
        test_metadata = []

        vocab_x = combined_vocabs[0]
        vocab_y = combined_vocabs[1]
        encoder_x = encoders[0]
        encoder_y = encoders[1]

        print("Generating primitives and constructing vocabulary")
        max_feature_vector_length = Task2.generate_primitives_and_vocabulary(test_dataset, input_primitives, x_test, y_test, vocab_x, vocab_y, test_metadata)
        Task2.create_ner_model_dataset(x_test, y_test, '../ner_dataset/', 'test')

        # FIXME: what happens when the test set has a sentence longer than all the sents in the train set,
        # i.e., when feature_vector_length=max_feature_vector_length in train set?
        # it'll get truncated presumably, and so will the output tag sequence
        if max_feature_vector_length > feature_vector_length:
            print("WARNING: Evaluation data set has sentences that exceed the max. feature vec length (%d > %d)" % (max_feature_vector_length, feature_vector_length))

        print("Encoding primitives")
        feature_vector_shapes = [feature_vector_length] * len(input_primitives)
        Task2.encode_primitives(x_test, y_test, encoder_x, encoder_y,
                                feature_vector_shapes, len(vocab_y), add_if_absent=False)

        test_dataset = Task2.create_tf_dataset(x_test, y_test, input_primitives)
        for idx, vocab in enumerate(vocab_x):
            print("Vocabulary Size of feat %s: %s" % (idx, len(vocab)))
            print("Random sample: %s" % (str(random.sample(vocab, min(len(vocab), 150)))))

        print("Vocabulary Size of tags: %s" % (len(vocab_y)))
        print("Tags: %s" % (' '.join(vocab_y)))

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