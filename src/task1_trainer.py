# Local packages
from common_imports import *
from model import baseline, experimental
from features import corpus, transform


# define globals
BUFFER_SIZE = 20000
BATCH_SIZE = 64
TAKE_SIZE = 1000
VOCAB_SIZE = None
MAX_FEATURE_VECTOR_LENGTH = 150


def prepare_data(dataset_path, primitive_type):
    """
    Prepare the raw dataset into encoded train, validation and test sets.
    """
    print("Loading dataset")
    raw_train_dataset = corpus.load_files_into_dataset(os.path.join(dataset_path, 'train'))
    raw_test_dataset = corpus.load_files_into_dataset(os.path.join(dataset_path, 'dev'))

    print("Performing NLP")
    raw_train_dataset = corpus.perform_nlp(raw_train_dataset, False)
    raw_test_dataset = corpus.perform_nlp(raw_test_dataset, False)

    print("Transforming dataset")
    train_data, test_data, vocabs, encoders, feature_vector_length = transform.tf_datasets_for_subtask_1(raw_train_dataset, raw_test_dataset, primitive_type, MAX_FEATURE_VECTOR_LENGTH)
    for idx, vocab in enumerate(vocabs):
        print("Vocabulary Size of feat %s: %s" % (idx, len(vocab)))

    # Shuffle only the training set. Since test set doesn't need to be shuffled.
    train_data = train_data.shuffle(
                BUFFER_SIZE, reshuffle_each_iteration=False)

    print("Train data sample:")
    print(next(iter(train_data)))
    print("Test data sample:")
    print(next(iter(test_data)))

    # Shuffle training data before sampling validation set
    train_data_temp = train_data.shuffle(BUFFER_SIZE,
                                         reshuffle_each_iteration=False)

    print("Cardinality:", tf.data.experimental.cardinality(train_data))
    train_data = train_data_temp.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    valid_data = train_data_temp.take(TAKE_SIZE)

    train_data = train_data.batch(BATCH_SIZE)
    valid_data = valid_data.batch(BATCH_SIZE)
    test_data = test_data.batch(BATCH_SIZE)

    print("Sample after tf padding")
    print("Train data sample:")
    print(next(iter(train_data)))
    print("Test data sample:")
    print(next(iter(test_data)))

    # Additional one for padding element
    global VOCAB_SIZE
    VOCAB_SIZE = [len(vocab) + 1 for vocab in vocabs]

    return train_data, valid_data, test_data, encoders, feature_vector_length


def print_mispredictions(gold_dataset, predictions, encoder, filepath):
    mispredictions = []
    for idx, data in enumerate(gold_dataset):
        if (predictions[idx] > 0.5 and data[1].numpy() != 1) or (predictions[idx] < 0.5 and data[1].numpy() != 0):
            tokens = [encoder.value(feat) if feat != 0 else '' for feat in data[0].numpy().tolist()]
            mispredictions.append(' '.join(tokens))

    with open(filepath, 'w', encoding="utf-8") as f:
        for i in mispredictions:
            f.write(i + "\n")


def get_class_distribution(dataset):
    num_pos = 0
    num_neg = 0
    for data in dataset:
        if data[1].numpy() == 1:
            num_pos += 1
        else:
            num_neg += 1
    return num_pos, num_neg


def calculate_class_weights(dataset):
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
    pos, neg = get_class_distribution(dataset)
    total = pos + neg
    weight_for_0 = (1 / neg) * (total)/2.0
    weight_for_1 = (1 / pos) * (total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('No. of class 0: {:d}'.format(neg))
    print('No. of class 1: {:d}'.format(pos))
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    return class_weight


def train(dataset_path):
    print("Preparing data")
    # input_primitives = [transform.InputPrimitive.POS]
    input_primitives = [transform.InputPrimitive.POS_WPUNCT,
                        transform.InputPrimitive.DEP]
    train_data, valid_data, test_data, encoder, feature_vector_length= prepare_data(dataset_path, input_primitives)

    print("Loading model")
    # model = experimental.simplified_baseline(VOCAB_SIZE, 64)
    model = experimental.create_multi_feature_model(
                   [{'dim': feature_vector_length, 'vocab_size': VOCAB_SIZE[0], 'embedding_dim': 128},
                    {'dim': feature_vector_length, 'vocab_size': VOCAB_SIZE[1], 'embedding_dim': 128}])
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.001),
                  metrics=[metrics.Precision(), metrics.Recall(), tfa.metrics.F1Score(num_classes=2, average="macro")])
                  #metrics=[tfa.metrics.F1Score(num_classes=2, average="micro")])
    model.summary()

    if os.name == 'nt':
        log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\"
    else:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=False)

    def to_dict(inputs, labels):
        # inputs = {"Feature_" + str(i+1): inputs[i] for i, _ in enumerate(input_primitives)}
        inputs = tf.reshape(inputs, (len(input_primitives), BATCH_SIZE, -1))
        return (inputs[0], inputs[1]), labels

    # train_data = train_data.map(to_dict)
    # valid_data = valid_data.map(to_dict)
    # test_data = test_data.map(to_dict)

    print("Training model")
    model.fit(train_data,
              epochs=50,
              validation_data=valid_data,
              callbacks=[tensorboard_callback, early_stopping_callback],
              class_weight=calculate_class_weights(train_data.unbatch()))

    eval_loss, eval_precision, eval_recall, eval_fscore = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval precision: {:.3f}, Eval recall: {:.3f}, Eval f-score: {:.3f}'.format(eval_loss, eval_precision, eval_recall, eval_fscore))
    #eval_loss, eval_fscore = model.evaluate(test_data)
    #print('\nEval loss: {:.3f}, Eval f-score: {:.3f}'.format(eval_loss, eval_fscore))

    #predictions = model.predict(test_data)
    # TODO: Need to be updgraded for multi feat dataset
    # print_mispredictions(test_data.unbatch(), predictions, encoders,
    #                      'logs/subtask_1_test_mispredictions.txt')

    return model


if __name__ == '__main__':
    train('../deft_corpus/data/deft_files/')
