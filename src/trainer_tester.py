from common_imports import *
import loader
from model import experimental
from features import InputPrimitive
import random


class Common:
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    TAKE_SIZE = 1000

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
        loader.Common.preprocess_dataset(task, raw_train_dataset)
        loader.Common.preprocess_dataset(task, raw_test_dataset)

        print("Transforming dataset")
        train_data, vocabs, encoders = Common.LOADER_TASK_REGISTRY[task].generate_model_train_inputs(
            raw_train_dataset, input_primitives, Common.TASK_REGISTRY[task].FEATURE_VECTOR_LENGTH)
        test_data, test_metadata = Common.LOADER_TASK_REGISTRY[task].generate_model_test_inputs(
            raw_test_dataset, input_primitives, encoders, vocabs, Common.TASK_REGISTRY[task].FEATURE_VECTOR_LENGTH)

        for idx, vocab in enumerate(vocabs):
            print("Vocabulary Size of feat %s: %s" % (idx, len(vocab)))
            print("Random sample: %s" % (str(random.sample(vocab, min(len(vocab), 150)))))

        # Test set should NOT be shuffled
        train_data = train_data.shuffle(
                    Common.BUFFER_SIZE, reshuffle_each_iteration=False)

        print("Train data sample:")
        print(next(iter(train_data)))
        print("Test data sample:")
        print(next(iter(test_data)))

        # Shuffle training data before sampling validation set
        train_data_temp = train_data.shuffle(Common.BUFFER_SIZE,
                                            reshuffle_each_iteration=False)

        print("Cardinality:", tf.data.experimental.cardinality(train_data))
        train_data = train_data_temp.skip(Common.TAKE_SIZE).shuffle(Common.BUFFER_SIZE)
        valid_data = train_data_temp.take(Common.TAKE_SIZE)

        train_data = train_data.batch(Common.BATCH_SIZE)
        valid_data = valid_data.batch(Common.BATCH_SIZE)
        test_data = test_data.batch(Common.BATCH_SIZE)

        print("Sample after tf padding")
        print("Train data sample:")
        print(next(iter(train_data)))
        print("Test data sample:")
        print(next(iter(test_data)))

        # Additional one for padding element
        vocab_size = [len(vocab) + 1 for vocab in vocabs]
        train_metadata = (encoders, vocabs, vocab_size)

        return train_data, valid_data, test_data, train_metadata, test_metadata


    @staticmethod
    def prepare_evaluation_data(task, dataset_path, input_primitives, train_metadata):
        print("Loading dataset")
        raw_test_dataset = Common.LOADER_TASK_REGISTRY[task].load_evaluation_data(dataset_path)

        print("Preprocessing dataset")
        loader.Common.preprocess_dataset(task, raw_test_dataset)

        print("Transforming dataset")
        test_data, test_metadata = Common.LOADER_TASK_REGISTRY[task].generate_model_test_inputs(raw_test_dataset, input_primitives, encoders=train_metadata[0], combined_vocabs=train_metadata[1], feature_vector_length=Common.TASK_REGISTRY[task].FEATURE_VECTOR_LENGTH)

        vocabs = train_metadata[1]
        for idx, vocab in enumerate(vocabs):
            print("Vocabulary Size of feat %s: %s" % (idx, len(vocab)))

        print("Test data sample:")
        print(next(iter(test_data)))

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


class Task1:
    FEATURE_VECTOR_LENGTH = 150     # Doubles as the maximum sentence length
    EPOCHS = 1
    INPUT_PRIMITIVES = [InputPrimitive.TOKEN,
                        InputPrimitive.DEP]
    EMBEDDING_DIM = 128
    LEARNING_RATE = 0.001
    ES_MIN_DELTA = 0.001
    ES_PATIENCE = 5


    @staticmethod
    def get_class_distribution(dataset):
        num_pos = 0
        num_neg = 0
        for data in dataset:
            if data[1].numpy() == 1:
                num_pos += 1
            else:
                num_neg += 1
        return num_pos, num_neg


    @staticmethod
    def calculate_class_weights(dataset):
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
        pos, neg = Task1.get_class_distribution(dataset)
        total = pos + neg
        weight_for_0 = (1 / neg) * (total)/2.0
        weight_for_1 = (1 / pos) * (total)/2.0

        class_weight = {0: weight_for_0, 1: weight_for_1}

        print('No. of class 0: {:d}'.format(neg))
        print('No. of class 1: {:d}'.format(pos))
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))

        return class_weight


    @staticmethod
    def prepare_training_data(dataset_path):
        return Common.prepare_training_data(Task.TASK_1, dataset_path, Task1.INPUT_PRIMITIVES)


    @staticmethod
    def train(train_data, valid_data, vocab_size):
        model_gen_params = [
            { 'dim': Task1.FEATURE_VECTOR_LENGTH, 'vocab_size': i, 'embedding_dim': Task1.EMBEDDING_DIM } for i in vocab_size]
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
                class_weight=Task1.calculate_class_weights(train_data.unbatch()))

        return model


    @staticmethod
    def prepare_evaluation_data(dataset_path, train_metadata):
        return Common.prepare_evaluation_data(Task.TASK_1, dataset_path, Task1.INPUT_PRIMITIVES, train_metadata)


    @staticmethod
    def evaluate(model, test_data, test_metdata, results_path):
        # test_metadata = [(file, context, sent, label)]
        predictions = model.predict(test_data)
        assert len(predictions) == len(test_metdata)

        with open(results_path, mode='w', encoding='UTF-8') as file:
            for i in range(0, len(predictions)):
                prediction = predictions[i]
                sentence = test_metdata[i][2]
                file.write('"%s"\t"%d"\n' % (sentence.raw_sent, prediction))


class Task2:
    FEATURE_VECTOR_LENGTH = 150
    EPOCHS = 1
    INPUT_PRIMITIVES = [InputPrimitive.TOKEN,
                        InputPrimitive.DEP]
    EMBEDDING_DIM = 128
    LEARNING_RATE = 0.001
    ES_MIN_DELTA = 0.001
    ES_PATIENCE = 5

    @staticmethod
    def prepare_training_data(dataset_path):
        return Common.prepare_training_data(Task.TASK_2, dataset_path, Task2.INPUT_PRIMITIVES)


    @staticmethod
    def prepare_evaluation_data(dataset_path, train_metadata):
        return Common.prepare_evaluation_data(Task.TASK_2, dataset_path, Task2.INPUT_PRIMITIVES, train_metadata)


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



