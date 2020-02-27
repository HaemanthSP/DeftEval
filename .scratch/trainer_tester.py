from common_imports import *
import loader
from loader import TrainMetadata
from model import experimental, baseline
from features import InputPrimitive
import tensorflow.keras.backend as K
import pickle
import ntpath
from tensorflow.python.keras.utils import metrics_utils
from tensorflow import math as math_ops
from tensorflow.python.keras.utils.generic_utils import to_list


class Common:
    BATCH_SIZE = 64
    VALIDATION_TAKE_SIZE = 1000

    LOADER_TASK_REGISTRY = {
        Task.TASK_1 : loader.Task1,
    }

    # Initialized after dependencies are resolved
    TASK_REGISTRY = {}


    @staticmethod
    def prepare_training_data(task, dataset_path, input_primitives):
        print("Loading dataset")
        raw_train_dataset = loader.Common.load_deft_data(task, os.path.join(dataset_path, 'train'))
        raw_test_dataset = loader.Common.load_deft_data(task, os.path.join(dataset_path, 'dev'))

        print("Preprocessing dataset")
        use_dummy_nlp_annotations = False
        loader.Common.preprocess_dataset(task, raw_train_dataset, use_dummy_nlp_annotations)
        loader.Common.preprocess_dataset(task, raw_test_dataset, use_dummy_nlp_annotations)

        print("Transforming dataset")
        train_data, valid_data, vocabs, encoders, train_class_dist = Common.LOADER_TASK_REGISTRY[task].generate_model_train_inputs(
            raw_train_dataset, input_primitives, Common.TASK_REGISTRY[task].FEATURE_VECTOR_LENGTH, Common.VALIDATION_TAKE_SIZE)
        test_data, test_metadata = Common.LOADER_TASK_REGISTRY[task].generate_model_test_inputs(
            raw_test_dataset, input_primitives, encoders, vocabs, Common.TASK_REGISTRY[task].FEATURE_VECTOR_LENGTH)

        #print("Train data sample:")
        #print(next(iter(train_data)))
        #print("Test data sample:")
        #print(next(iter(test_data)))

        print("Cardinality:", tf.data.experimental.cardinality(train_data))

        train_data = train_data.batch(Common.BATCH_SIZE)
        valid_data = valid_data.batch(Common.BATCH_SIZE)
        test_data = test_data.batch(Common.BATCH_SIZE)

        print("Sample after tf padding")
        print("Train data sample:")
        print(next(iter(train_data)))
        print("Test data sample:")
        print(next(iter(test_data)))

        vocab_size_x = None
        vocab_size_y = None

        if task == Task.TASK_1:
            # Additional one for padding element
            vocab_size_x = [len(vocab) + 1 for vocab in vocabs[0]]
        else:
            vocab_size_x = [len(vocab) + 1 for vocab in vocabs[0]]
            vocab_size_y = len(vocabs[1]) + 1

        train_metadata = TrainMetadata(encoders, vocabs, (vocab_size_x, vocab_size_y), train_class_dist)

        return train_data, valid_data, test_data, train_metadata, test_metadata


    @staticmethod
    def prepare_evaluation_data(task, dataset_path, input_primitives, train_metadata):
        print("Loading dataset")
        raw_test_dataset = Common.LOADER_TASK_REGISTRY[task].load_evaluation_data(dataset_path)

        print("Preprocessing dataset")
        loader.Common.preprocess_dataset(task, raw_test_dataset)

        print("Transforming dataset")
        test_data, test_metadata = Common.LOADER_TASK_REGISTRY[task].generate_model_test_inputs(raw_test_dataset, input_primitives, encoders=train_metadata.encoders, combined_vocabs=train_metadata.vocabs, feature_vector_length=Common.TASK_REGISTRY[task].FEATURE_VECTOR_LENGTH)

        #print("Test data sample:")
        #print(next(iter(test_data)))

        test_data = test_data.batch(Common.BATCH_SIZE)

        print("Sample after tf padding")
        print("Test data sample:")
        print(next(iter(test_data)))

        return test_data, test_metadata


    @staticmethod
    def get_tensorboard_callback():
        if os.name == 'nt':
            log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\"
        else:
            log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        return tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)


    @staticmethod
    def calculate_class_weights(train_class_dist):
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
        class_weights = train_class_dist.calculate_class_weights(lambda c,t: (1/c) * (t/2.0))

        print("Class distribution: " + str(train_class_dist.class_dists))
        print("Class weights: " + str(class_weights))

        return class_weights


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='F1Score', class_id=None, dtype=None):
        super(F1Score, self).__init__(name=name, dtype=dtype)

        self.class_id = class_id
        default_threshold = 0.5
        self.thresholds = metrics_utils.parse_init_thresholds(
            None, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer='zeros')


    def update_state(self, y_true, y_pred, sample_weight=None):
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        precision = math_ops.divide_no_nan(self.true_positives,
                                    self.true_positives + self.false_positives)
        recall = math_ops.divide_no_nan(self.true_positives,
                                    self.true_positives + self.false_negatives)

        precision = precision[0] if len(self.thresholds) == 1 else precision
        recall = recall[0] if len(self.thresholds) == 1 else recall

        mul_val = precision * recall
        add_val = precision + recall

        f1_score = 2 * math_ops.divide_no_nan(mul_val, add_val)
        return f1_score

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'class_id': self.class_id
        }
        base_config = super(F1Score, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Task1:
    FEATURE_VECTOR_LENGTH = 150     # Doubles as the maximum sentence length
    EPOCHS = 50
    INPUT_PRIMITIVES = [InputPrimitive.TOKEN,
                        InputPrimitive.POS_WPUNCT,
                        InputPrimitive.DEP,
                        InputPrimitive.HEAD]
    EMBEDDING_DIM = 128
    LEARNING_RATE = 0.001
    ES_MIN_DELTA = 0.001
    ES_PATIENCE = 5
    PRETRAINED_EMBEDDING_PATH = '..\\resources\\glove.840B.300d.metadata'


    @staticmethod
    def load_pretrained_embeddings(token_encoder, token_vocab_size, embedding_path):
        embeddings, vocab, dims = pickle.load(open(embedding_path, mode='rb'))

        assert token_vocab_size == token_encoder.max_number() + 1
        out_matrix = np.zeros((token_vocab_size, dims), dtype='float32')
        oov_count = 0
        for i in range(1, token_vocab_size):
            token = token_encoder.value(i)
            if token in vocab:
                out_matrix[i] = embeddings[token]
            else:
                out_matrix[i] = np.random.uniform(-1.0, 1.0, (1, dims))
                oov_count += 1

        print("\t%d words were out-of-vocabulary" % (oov_count))

        return out_matrix, dims


    @staticmethod
    def prepare_training_data(dataset_path):
        return Common.prepare_training_data(Task.TASK_1, dataset_path, Task1.INPUT_PRIMITIVES)


    @staticmethod
    def train(train_data, valid_data, train_metadata):
        vocab_size_x = train_metadata.vocab_sizes[0]
        model_gen_params = [{
                'dim': Task1.FEATURE_VECTOR_LENGTH,
                'vocab_size': i,
                'embedding_dim': Task1.EMBEDDING_DIM,
                'embedding_initializer': None,
                'trainable': True,
            } for i in vocab_size_x]

        token_primitive_feature_idx = Task1.INPUT_PRIMITIVES.index(InputPrimitive.TOKEN)
        print("Loading pretrained embeddings for word tokens")
        pretrained_embeds, pretained_dims = Task1.load_pretrained_embeddings(train_metadata.encoders[0][token_primitive_feature_idx],
                                                                            vocab_size_x[token_primitive_feature_idx],
                                                                            Task1.PRETRAINED_EMBEDDING_PATH)
        model_gen_params[token_primitive_feature_idx]['embedding_initializer'] = pretrained_embeds
        model_gen_params[token_primitive_feature_idx]['embedding_dim'] = pretained_dims
        model_gen_params[token_primitive_feature_idx]['trainable'] = False

        model = experimental.create_multi_feature_model(model_gen_params)
        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.Adam(Task1.LEARNING_RATE),
                    metrics=[metrics.Precision(), metrics.Recall(), F1Score()])
        model.summary()

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_F1Score', min_delta=Task1.ES_MIN_DELTA, patience=Task1.ES_PATIENCE, restore_best_weights=True)

        model.fit(train_data,
                epochs=Task1.EPOCHS,
                validation_data=valid_data,
                callbacks=[early_stopping_callback],
                class_weight=Common.calculate_class_weights(train_metadata.train_data_class_dist))

        return model


    @staticmethod
    def prepare_evaluation_data(dataset_path, train_metadata):
        return Common.prepare_evaluation_data(Task.TASK_1, dataset_path, Task1.INPUT_PRIMITIVES, train_metadata)


    @staticmethod
    def evaluate(model, test_data, test_metadata, train_metadata, results_path):
        # test_metadata = [(file, context, sent, label)]
        predictions = model.predict(test_data)
        assert predictions.shape[0] == len(test_metadata)

        file_handles = {}

        for i in range(0, len(predictions)):
            prediction = 1 if predictions[i] > 0.5 else 0
            sentence = test_metadata[i][2]
            filename = ntpath.basename(test_metadata[i][0].filename)
            output_filepath = os.path.join(results_path, filename)

            if output_filepath in file_handles:
                output_filehandle = file_handles[output_filepath]
            else:
                output_filehandle = open(output_filepath, mode='w', encoding='utf-8')
                file_handles[output_filepath] = output_filehandle

            output_filehandle.write('%s\t%d\n' % (sentence.raw_sent, prediction))

        for handle in file_handles.values():
            handle.close()


# Deferred init.
Common.TASK_REGISTRY = {
    Task.TASK_1 : Task1
}



