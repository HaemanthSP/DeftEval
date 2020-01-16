from common_imports import *
import trainer, tester
import sys


class Task(Enum):
    TASK_1 = 1,
    TASK_2 = 2,
    TASK_3 = 3


CORPUS_PATH = '../deft_corpus/data/deft_files/'



def train(task, dataset_path):
    registry = {
        Task.TASK_1 : trainer.Task1,
        #Task.TASK_2 : trainer.Task2,
        #Task.TASK_3 : trainer.Task3,
    }

    print("Training for " + str(task) + "...")

    print("Preparing data")
    train_data, valid_data, test_data, encoders, vocab_size = registry[task].prepare_data(dataset_path)

    print("Constructing and training model")
    trained_model = registry[task].train(train_data, valid_data, vocab_size)

    print("Evaluating model on dev set")
    eval_loss, eval_precision, eval_recall = trained_model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval precision: {:.3f}, Eval recall: {:.3f}'.format(eval_loss, eval_precision, eval_recall))

    return trained_model



if __name__ == '__main__':
    train(Task.TASK_1, CORPUS_PATH)
