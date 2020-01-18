from common_imports import *
import trainer_tester
from util import Serde


TASK_REGISTRY = {
    Task.TASK_1 : trainer_tester.Task1,
    Task.TASK_2 : trainer_tester.Task2,
    Task.TASK_3 : trainer_tester.Task3,
}

CORPUS_PATH = '../deft_corpus/data/deft_files/'
MODEL_SAVE_PATH = '../saved_models/'
EVAL_RESULTS_PATH = '../eval_results/'


def train(task, dataset_path):
    print("Training for task " + str(int(task)) + "...")

    print("Preparing data")
    train_data, valid_data, test_data, train_metadata, _ = TASK_REGISTRY[task].prepare_training_data(dataset_path)

    print("Constructing and training model")
    trained_model = TASK_REGISTRY[task].train(train_data, valid_data, train_metadata[2])

    print("Evaluating model on dev set")
    eval_loss, eval_precision, eval_recall = trained_model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval precision: {:.3f}, Eval recall: {:.3f}'.format(eval_loss, eval_precision, eval_recall))

    return trained_model, train_metadata


def save_model(model, train_metadata, save_directory, filename_prefix):
    print("Saving model and metadata")

    Serde.save_tf_model(model, save_directory + '/' + filename_prefix + '_MODEL.h5')
    Serde.save_metadata(train_metadata, save_directory + '/' + filename_prefix + '_METADATA.pkl')


def load_model(save_directory, filename_prefix):
    print("Loading model and metadata")

    model = Serde.load_tf_model(save_directory + '/' + filename_prefix + '_MODEL.h5')
    train_metadata = Serde.load_metadata(save_directory + '/' + filename_prefix + '_METADATA.pkl')
    return model, train_metadata


def evaluate(task, dataset_path, trained_model, train_metadata):
    print("Evaluating task " + str(int(task)) + "...")

    print("Preparing data")
    test_data, test_metadata = TASK_REGISTRY[task].prepare_evaluation_data(dataset_path, train_metadata)

    print("Evaluating model")
    results_path = EVAL_RESULTS_PATH + '/TASK_' + str(int(task)) + '_RESULTS.deft'
    TASK_REGISTRY[task].evaluate(trained_model, test_data, test_metadata, results_path)

    print("Wrote results to %s" % (results_path))


if __name__ == '__main__':
    model, metadata = train(Task.TASK_1, CORPUS_PATH)
    save_model(model, metadata, MODEL_SAVE_PATH, 'Task1-Test')
    model, metadata = load_model(MODEL_SAVE_PATH, 'Task1-Test')
    evaluate(Task.TASK_1, '../deft_corpus/evaluation/input/ref', model, metadata)

