import random
import datetime
import itertools

import spacy
import pandas as pd
from tqdm import tqdm
from tensorflow.keras import optimizers, metrics
import tensorflow as tf
import tensorflow_text as tft
import tensorflow_datasets as tfds


from model import experimental

#Globals
NLP = spacy.load('en_core_web_lg', parser=False, entity=False)
TAKE_SIZE = 2000
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPOCHS = 30
ES_MIN_DELTA = 0.001
ES_PATIENCE = 5
RESERVED = [',', '.', '"', "'", '/', '(', ')', '-', '_', ';', ':', '?', '!', '[', ']']


def append_pos(sentence):
    tokens = []
    pos = []
    for token in NLP(sentence, disable=["parser", "ner"]):
        tokens.append(token.text)
        # pos.append(token.text if token.pos_ == "PUNCT" else token.pos_)
        if token.pos_ != "PUNCT":
            tokens.append(token.pos_)
    
    # return " ".join(tokens + pos)
    return " ".join(tokens)


def prepare_data(path, encoder=None):

    # Get training data
    df = pd.read_csv(path, sep="\t", names=["Sentence", "Label"])
    target = df.pop("Label")

    # data = tokenize(df.values) 
    df["withpos"] = df.apply(lambda row: append_pos(row["Sentence"]), axis=1)
    
    raw_dataset = tf.data.Dataset.from_tensor_slices((df["withpos"].values, target.values))
    # TODO: Do a preprocessing

    vocab_size = None
    if not encoder: 
        print("\nTokenization...")
        tokenizer = tft.WhitespaceTokenizer()
        tokens_list = tokenizer.tokenize(df["withpos"].to_list()).to_list()
        vocabulary_set = set([x.decode('utf-8').lower() for x in itertools.chain.from_iterable(tokens_list)])
        vocab_size = len(vocabulary_set)
        print("Vocabulary Size: ", vocab_size)
        print(random.sample(vocabulary_set, 20))

        encoder = tfds.features.text.TokenTextEncoder(
            vocabulary_set,
            tokenizer=tfds.features.text.Tokenizer(reserved_tokens=RESERVED),
            lowercase=True,
            strip_vocab=False)
        example_text = next(iter(raw_dataset))[0].numpy()
        print("Example text")
        print(example_text)
        encoded_example = encoder.encode(example_text)
        print("Encoded Example text")
        print(encoded_example)


    def encode(text_tensor, label):
        encoded_text = encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    print("\nEncoding...")
    encoded_data = raw_dataset.map(encode_map_fn)
    
    return encoded_data, vocab_size, encoder


def analyse(model, test_data, encoder): 
    for data, label in test_data.take(1): 
        prediction = model.predict(data) 
        for actual, pred, inp in zip(label, prediction, data): 
            print(actual.numpy(), pred, encoder.decode(inp)) 


def evaluate(model, test_data, test_metdata, results_path):
    # test_metadata = [(file, context, sent, label)]
    predictions = model.predict(test_data)
    assert len(predictions) == len(test_metdata)

    with open(results_path, mode='w', encoding='UTF-8') as file:
        for i in range(0, len(predictions)):
            prediction = predictions[i]
            sentence = test_metdata[i][2]
            file.write('"%s"\t"%d"\n' % (sentence.raw_sent, prediction))


def train():

    # Load the dataset into batches
    train_valid_dataset, vocab_size, encoder = prepare_data("../deft_corpus/task1_data/task1_train.deft")
    test_dataset, _, _ = prepare_data("../deft_corpus/task1_data/task1_dev.deft", encoder=encoder)

    train_data = train_valid_dataset.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

    valid_data = train_valid_dataset.take(TAKE_SIZE)
    valid_data = valid_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))
    test_data = test_dataset.shuffle(BUFFER_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

    print("Loading baseline model")
    # Vocab +2 one for padding and one for <UNK>
    model = experimental.simplified_baseline(vocab_size+2, 64)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(0.0001),
                  metrics=[metrics.Precision(), metrics.Recall()])
            
    model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=ES_MIN_DELTA, patience=ES_PATIENCE, restore_best_weights=True)

    model.fit(train_data,
              epochs=EPOCHS,
              validation_data=valid_data,
              callbacks=[tensorboard_callback, early_stopping_callback])

    eval_loss, eval_precision, eval_recall = model.evaluate(test_data)
    print('\nEval Loss: {:.3f}, Eval Precision: {:.3f}, Eval Recall: {:.3f}'.format(eval_loss, eval_precision, eval_recall))

    analyse(model, test_data, encoder)
    
    return model, test_data, encoder


if __name__ == "__main__":
    train()