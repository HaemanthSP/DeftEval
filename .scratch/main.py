from common_imports import *
import trainer_tester
from util import Serde
from sklearn.metrics import classification_report, confusion_matrix
import gensim


TASK_REGISTRY = {
    Task.TASK_1 : trainer_tester.Task1
}


class TestSetEvaluationMetrics:
    def __init__(self, pred_probs, test_metadata):
        self.pred_probs = pred_probs
        self.test_metadata = test_metadata    # [(file, context, sent, _)]
        self.high_confidence_true_samples = []
        self.high_confidence_false_samples = []
        self.calculate_metrics()


    def calculate_metrics(self):
        TRUE_POSITIVE_CONFIDENCE_THRESHOLD = 0.8
        FALSE_NEGATIVE_CONFIDENCE_THRESHOLD = 0.2

        assert self.pred_probs is not None
        assert self.test_metadata is not None
        assert len(self.pred_probs) == len(self.test_metadata)

        self.preds = np.array([1 if i > 0.5 else 0 for i in self.pred_probs], dtype='float32')

        for i in range(0, len(self.preds)):
            m = self.test_metadata[i]
            prob = self.pred_probs[i]

            if prob >= TRUE_POSITIVE_CONFIDENCE_THRESHOLD:
                self.high_confidence_true_samples.append((prob, m))
            elif prob <= FALSE_NEGATIVE_CONFIDENCE_THRESHOLD:
                self.high_confidence_false_samples.append((prob, m))


    def print_report(self):
        print("Test set evaluation results:")
        num_high_confidence_true_samples = len(self.high_confidence_true_samples)
        num_high_confidence_false_samples = len(self.high_confidence_false_samples)

        print("\tHigh confidence true-positive: %d" % (num_high_confidence_true_samples))
        for prob, m in random.sample(self.high_confidence_true_samples, min(num_high_confidence_true_samples, 10)):
            print('\t\t%.3f\t%s' % (prob, ' '.join([t.token for t in m[2].tokens])))

        print("\tHigh confidence false-negative: %d" % (num_high_confidence_false_samples))
        for prob, m in random.sample(self.high_confidence_false_samples, min(num_high_confidence_false_samples, 10)):
            print('\t\t%.3f\t%s' % (prob, ' '.join([t.token for t in m[2].tokens])))

    def compare(self, rhs):
        # TODO: implement
        pass



def train(task, dataset_path, extra_train_samples=None):
    print("Training for task " + str(int(task)) + "...")

    print("Preparing data")
    train_data, valid_data, test_data, train_metadata, test_metadata = TASK_REGISTRY[task].prepare_training_data(dataset_path, extra_train_samples)

    print("Constructing and training model")
    trained_model = TASK_REGISTRY[task].train(train_data, valid_data, train_metadata)

    evaluate(trained_model, test_data, test_metadata)
    return trained_model, train_metadata, test_data, test_metadata


def evaluate(trained_model, dev_data, dev_metadata):
    print("Evaluating model on dev set")
    eval_loss, eval_precision, eval_recall, eval_f1 = trained_model.evaluate(dev_data)
    print('\nEval Metrics:\n\tLoss: {:.3f}\n\tPrecision: {:.3f}\n\tRecall: {:.3f}\n\tF1: {:.3f}'.format(eval_loss, eval_precision, eval_recall, eval_f1))

    preds = trained_model.predict(dev_data)
    assert preds.shape[0] == len(dev_metadata)
    preds = np.array([1 if i > 0.5 else 0 for i in preds], dtype='float32')
    gold_labels = np.array([i[3] for i in dev_metadata], dtype='float32')
    print(classification_report(gold_labels, preds))


def save_model(model, train_metadata, save_directory, filename_prefix):
    print("Saving model and metadata")

    Serde.save_tf_model(model, os.path.join(save_directory, filename_prefix + '_MODEL.h5'))
    Serde.save_metadata(train_metadata, os.path.join(save_directory, filename_prefix + '_MODEL-METADATA.pkl'))


def load_model(save_directory, filename_prefix):
    print("Loading model and metadata")

    train_metadata = Serde.load_metadata(os.path.join(save_directory, filename_prefix + '_MODEL-METADATA.pkl'))
    model = Serde.load_tf_model(os.path.join(save_directory, filename_prefix + '_MODEL.h5'))
    return model, train_metadata


def test(task, dataset_path, trained_model, train_metadata, results_path, zip_path):
    print("Evaluating task " + str(int(task)) + "...")

    print("Preparing data")
    test_data, test_metadata = TASK_REGISTRY[task].prepare_evaluation_data(dataset_path, train_metadata)

    print("Evaluating model")
    TASK_REGISTRY[task].evaluate(trained_model, test_data, test_metadata, train_metadata, results_path, zip_path)

    print("Wrote results to %s" % (results_path))


def default_train_eval_test():
    current_task = Task.TASK_1
    model_prefix = 'Task1-Test'
    eval_data_path = os.path.join('..', 'deft_corpus', 'data', 'test_files', 'subtask_1')

    CORPUS_PATH = os.path.join('..', 'deft_corpus', 'data', 'deft_files')
    MODEL_SAVE_PATH = os.path.join('..', 'resources')
    EVAL_RESULTS_PATH = os.path.join('..', 'result', 'task1')
    ZIP_RESULTS_PATH = os.path.join('..', 'result', 'task1')

    model, metadata, _ , _ = train(current_task, CORPUS_PATH)
    save_model(model, metadata, MODEL_SAVE_PATH, model_prefix)
    model, metadata = load_model(MODEL_SAVE_PATH, model_prefix)
    test(current_task, eval_data_path, model, metadata, EVAL_RESULTS_PATH, ZIP_RESULTS_PATH)


def iterative_training(iteration, include_previous_itr_true_samples, include_previous_itr_false_samples):
    current_task = Task.TASK_1
    model_prefix = 'Task1-Iterative-Iteration_'
    eval_data_path = os.path.join('..', 'deft_corpus', 'data', 'test_files', 'subtask_1')

    CORPUS_PATH = os.path.join('..', 'deft_corpus', 'data', 'deft_files')
    MODEL_SAVE_PATH = os.path.join('..', 'resources')
    EVAL_RESULTS_PATH = os.path.join('..', 'result', 'task1')
    ZIP_RESULTS_PATH = os.path.join('..', 'result', 'task1')

    print("==> Begin iteration %d...<==\n" % (iteration))

    print("Attempting to load previous iteration's metadata...")
    prev_itr = iteration - 1
    try:
        prev_itr_metadata = Serde.load_metadata(os.path.join(MODEL_SAVE_PATH,
                                                model_prefix + str(prev_itr) + '_ITR-METADATA'))
    except:
        print("\tError or none found")
        prev_itr_metadata = None

    prev_itr_extra_train_samples = None
    if prev_itr_metadata is not None:
        prev_itr_extra_train_samples = []

        if include_previous_itr_true_samples:
            print('\tIncluding previous high-confidence true samples')
            prev_itr_extra_train_samples += prev_itr_metadata.high_confidence_true_samples

        if include_previous_itr_false_samples:
            print('\tIncluding previous high-confidence false samples')
            prev_itr_extra_train_samples += prev_itr_metadata.high_confidence_false_samples

    if prev_itr_extra_train_samples is not None and len(prev_itr_extra_train_samples) == 0:
        prev_itr_extra_train_samples = None

    trained_model, train_metadata, _, _ = train(current_task, CORPUS_PATH, prev_itr_extra_train_samples)
    test_data, test_metadata = TASK_REGISTRY[current_task].prepare_evaluation_data(eval_data_path, train_metadata)

    predictions = TASK_REGISTRY[current_task].evaluate(trained_model, test_data, test_metadata, train_metadata, EVAL_RESULTS_PATH, ZIP_RESULTS_PATH)
    test_results_metadata = TestSetEvaluationMetrics(predictions, test_metadata)

    test_results_metadata.print_report()
    if prev_itr_metadata is not None:
        test_results_metadata.compare(prev_itr_metadata)

    save_model(trained_model, train_metadata, MODEL_SAVE_PATH, model_prefix + str(iteration))
    Serde.save_metadata(test_results_metadata,
                        os.path.join(MODEL_SAVE_PATH, model_prefix + '_ITR-METADATA_' + str(iteration)))




if __name__ == '__main__':
    # default_train_eval_test()
    iterative_training(2,
                    include_previous_itr_true_samples=True,
                    include_previous_itr_false_samples=True)

