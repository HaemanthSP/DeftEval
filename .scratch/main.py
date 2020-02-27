from common_imports import *
import trainer_tester
from util import Serde
from sklearn.metrics import classification_report, confusion_matrix


TASK_REGISTRY = {
    Task.TASK_1 : trainer_tester.Task1
}


def train(task, dataset_path):
    print("Training for task " + str(int(task)) + "...")

    print("Preparing data")
    train_data, valid_data, test_data, train_metadata, test_metadata = TASK_REGISTRY[task].prepare_training_data(dataset_path)

    print("Constructing and training model")
    trained_model = TASK_REGISTRY[task].train(train_data, valid_data, train_metadata)

    print("Evaluating model on dev set")
    eval_loss, eval_precision, eval_recall, eval_f1 = trained_model.evaluate(test_data)
    print('\nEval Metrics:\n\tLoss: {:.3f}\n\tPrecision: {:.3f}\n\tRecall: {:.3f}\n\tF1: {:.3f}'.format(eval_loss, eval_precision, eval_recall, eval_f1))

    preds = trained_model.predict(test_data)
    assert preds.shape[0] == len(test_metadata)
    preds = np.array([1 if i > 0.5 else 0 for i in preds], dtype='float32')
    gold_labels = np.array([i[3] for i in test_metadata], dtype='float32')
    #print(confusion_matrix(preds, gold_labels))
    print(classification_report(gold_labels, preds))

    return trained_model, train_metadata


def save_model(model, train_metadata, save_directory, filename_prefix):
    print("Saving model and metadata")

    Serde.save_tf_model(model, os.path.join(save_directory, filename_prefix + '_MODEL'))
    Serde.save_metadata(train_metadata, os.path.join(save_directory, filename_prefix + '_MODEL',
                        'METADATA.pkl'))


def load_model(save_directory, filename_prefix):
    print("Loading model and metadata")

    train_metadata = Serde.load_metadata(os.path.join(save_directory, filename_prefix + '_MODEL',
                                        'METADATA.pkl'))
    model = Serde.load_tf_model(os.path.join(save_directory, filename_prefix + '_MODEL'))
    return model, train_metadata


def evaluate(task, dataset_path, trained_model, train_metadata):
    print("Evaluating task " + str(int(task)) + "...")

    print("Preparing data")
    test_data, test_metadata = TASK_REGISTRY[task].prepare_evaluation_data(dataset_path, train_metadata)

    print("Evaluating model")
    results_path = EVAL_RESULTS_PATH
    TASK_REGISTRY[task].evaluate(trained_model, test_data, test_metadata, train_metadata, results_path)

    print("Wrote results to %s" % (results_path))


if __name__ == '__main__':
    current_task = Task.TASK_1
    model_prefix = 'Task1-Test'
    eval_data_path = os.path.join('..', 'deft_corpus', 'data', 'test_files', 'subtask_1')

    CORPUS_PATH = os.path.join('..', 'deft_corpus', 'data', 'deft_files')
    MODEL_SAVE_PATH = os.path.join('..', 'resources')
    EVAL_RESULTS_PATH = os.path.join('..', 'result', 'task1')

    model, metadata = train(current_task, CORPUS_PATH)
    #save_model(model, metadata, MODEL_SAVE_PATH, model_prefix)
    #model, metadata = load_model(MODEL_SAVE_PATH, model_prefix)
    evaluate(current_task, eval_data_path, model, metadata)