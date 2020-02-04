from common_imports import *
import loader
from loader import TrainMetadata
from model import experimental, baseline
from features import InputPrimitive
import tensorflow.keras.backend as K


class Common:
    BATCH_SIZE = 64
    VALIDATION_TAKE_SIZE = 1000

    LOADER_TASK_REGISTRY = {
        Task.TASK_1 : loader.Task1,
        Task.TASK_2 : loader.Task2,
        Task.TASK_3 : loader.Task3,
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
            vocab_size_x = [len(vocab) + 1 for vocab in vocabs]
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
        test_data, test_metadata = Common.LOADER_TASK_REGISTRY[task].generate_model_test_inputs(raw_test_dataset, input_primitives, encoders=train_metadata[0], combined_vocabs=train_metadata[1], feature_vector_length=Common.TASK_REGISTRY[task].FEATURE_VECTOR_LENGTH)

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
            log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\"
        else:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        return tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)


    @staticmethod
    def calculate_class_weights(train_class_dist):
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
        class_weights = train_class_dist.calculate_class_weights(lambda c,t: (1/c) * (t/2.0))

        print("Class distribution: " + str(train_class_dist.class_dists))
        print("Class weights: " + str(class_weights))

        return class_weights


class Task1:
    FEATURE_VECTOR_LENGTH = 150     # Doubles as the maximum sentence length
    EPOCHS = 100
    INPUT_PRIMITIVES = [InputPrimitive.TOKEN,
                        InputPrimitive.DEP,
                        InputPrimitive.HEAD]
    EMBEDDING_DIM = 128
    LEARNING_RATE = 0.001
    ES_MIN_DELTA = 0.001
    ES_PATIENCE = 5


    @staticmethod
    def prepare_training_data(dataset_path):
        return Common.prepare_training_data(Task.TASK_1, dataset_path, Task1.INPUT_PRIMITIVES)


    @staticmethod
    def train(train_data, valid_data, train_metadata):
        vocab_size_x = train_metadata.vocab_sizes[0]
        model_gen_params = [
            { 'dim': Task1.FEATURE_VECTOR_LENGTH, 'vocab_size': i, 'embedding_dim': Task1.EMBEDDING_DIM } for i in vocab_size_x]
        model = experimental.create_multi_feature_model(model_gen_params)
        model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.Adam(Task1.LEARNING_RATE),
                    metrics=[metrics.Precision(), metrics.Recall()])
        model.summary()

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=Task1.ES_MIN_DELTA, patience=Task1.ES_PATIENCE, restore_best_weights=True)

        model.fit(train_data,
                epochs=Task1.EPOCHS,
                validation_data=valid_data,
                callbacks=[Common.get_tensorboard_callback(), early_stopping_callback],
                class_weight=Common.calculate_class_weights(train_metadata.train_data_class_dist))

        return model


    @staticmethod
    def prepare_evaluation_data(dataset_path, train_metadata):
        return Common.prepare_evaluation_data(Task.TASK_1, dataset_path, Task1.INPUT_PRIMITIVES, train_metadata)


    @staticmethod
    def evaluate(model, test_data, test_metdata, train_metadata, results_path):
        # test_metadata = [(file, context, sent, label)]
        predictions = model.predict(test_data)
        assert len(predictions) == len(test_metdata)

        with open(results_path, mode='w', encoding='UTF-8') as file:
            for i in range(0, len(predictions)):
                prediction = predictions[i]
                sentence = test_metdata[i][2]
                file.write('"%s"\t"%d"\n' % (sentence.raw_sent, prediction))


class Task2:
    FEATURE_VECTOR_LENGTH = 350
    EPOCHS = 50
    INPUT_PRIMITIVES = [InputPrimitive.DEP]
    EMBEDDING_DIM = 32
    LEARNING_RATE = 0.05
    ES_MIN_DELTA = 0.001
    ES_PATIENCE = 5


    @staticmethod
    def prepare_training_data(dataset_path):
        return Common.prepare_training_data(Task.TASK_2, dataset_path, Task2.INPUT_PRIMITIVES)

    @staticmethod
    def train(train_data, valid_data, train_metadata):
        # https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
        def weighted_categorical_crossentropy(weights):
            """
            A weighted version of keras.objectives.categorical_crossentropy

            Variables:
                weights: numpy array of shape (C,) where C is the number of classes

            Usage:
                weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
                loss = weighted_categorical_crossentropy(weights)
                model.compile(loss=loss,optimizer='adam')
            """

            weights = K.variable(weights)

            def loss(y_true, y_pred):
                # scale predictions so that the class probas of each sample sum to 1
                y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
                # clip to prevent NaN's and Inf's
                y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
                # calc
                loss = y_true * K.log(y_pred) * weights
                loss = -K.sum(loss, -1)
                return loss

            return loss


        vocab_size_x = train_metadata.vocab_sizes[0]
        vocab_size_y = train_metadata.vocab_sizes[1]
        model_gen_params = [
            { 'dim': Task2.FEATURE_VECTOR_LENGTH, 'vocab_size': i, 'embedding_dim': Task2.EMBEDDING_DIM } for i in vocab_size_x]
        model = baseline.create_task2_model(model_gen_params, vocab_size_y)

        class_weights_array = np.ones(shape=[vocab_size_y])
        for k,v in Common.calculate_class_weights(train_metadata.train_data_class_dist).items():
            class_weights_array[k] = v

        model.compile(loss=weighted_categorical_crossentropy(class_weights_array),
                    optimizer=optimizers.Adam(Task2.LEARNING_RATE),
                    metrics=[metrics.CategoricalAccuracy()])
        model.summary()

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=Task2.ES_MIN_DELTA, patience=Task2.ES_PATIENCE, restore_best_weights=True)

        history = model.fit(train_data,
                epochs=Task2.EPOCHS,
                validation_data=valid_data,
                callbacks=[Common.get_tensorboard_callback(), early_stopping_callback])

        return model


    @staticmethod
    def prepare_evaluation_data(dataset_path, train_metadata):
        return Common.prepare_evaluation_data(Task.TASK_2, dataset_path, Task2.INPUT_PRIMITIVES, train_metadata)


    @staticmethod
    def evaluate(model, test_data, test_metdata, train_metadata, results_path):
        # test_metadata = [(file, context, labels)]
        predictions = model.predict(test_data)
        assert len(predictions) == len(test_metdata)
        tag_encoder = train_metadata.encoders[1]

        with open(results_path, mode='w', encoding='UTF-8') as file:
            for i in range(0, len(predictions)):
                prediction = predictions[i]
                context = test_metdata[i][1]
                assert context.len() == 1
                sentence = context.sentences[0]

                for j in range(0, len(sentence.tokens)):
                    token = sentence.tokens[j]
                    logits = prediction[j]
                    tag_id = tf.argmax(logits).numpy()
                    tag = tag_encoder.value(tag_id)

                    file.write('%s\t%s\t%d\t%d\t%s\n' %
                                (token.token, token.filename, token.start_char, token.end_char, tag))

                file.write('\n')


class Task3:
    @staticmethod
    def prepare_evaluation_data(dataset_path, train_metadata):
        return Common.prepare_evaluation_data(Task.TASK_3, dataset_path, Task3.INPUT_PRIMITIVES, train_metadata)



# Deferred init.
Common.TASK_REGISTRY = {
    Task.TASK_1 : Task1,
    Task.TASK_2 : Task2,
    Task.TASK_3 : Task3,
}



