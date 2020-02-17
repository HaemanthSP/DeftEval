import numpy as np
from tqdm import tqdm
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Embedding, Bidirectional, SimpleRNN, LSTM, Dense, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# from tensorflow.keras.models import Model, Input
from keras import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from sklearn.metrics import classification_report

# from tf_crf_layer.layer import CRF
# from tf_crf_layer.loss import crf_loss
# from tf_crf_layer.metrics import crf_accuracy

EMBED_DIM = 100
EPOCHS = 10


def read_data(path, metadata=None):

    print("Loading data from path %s" % (path))

    # Load data
    with open(path + '.words.txt', 'r') as word_handle:
        sentences = word_handle.readlines()
        sentences = [sentence.strip().split() for sentence in tqdm(sentences)]
 
    with open(path + '.tags.txt', 'r') as tag_handle:
        tag_sequences = tag_handle.readlines()
        tag_sequences = [tag_sequence.strip().split() for tag_sequence in tqdm(tag_sequences)]
 
     
    # Collect the meta information
    if metadata:
         vocab_words, vocab_tags, word2idx, tag2idx, maxlen = metadata
    else:
        vocab_words, vocab_tags= set(), set()
        maxlen = 0
        for sentence, tag_sequence in tqdm(zip(sentences, tag_sequences)):
            assert(len(sentence) == len(tag_sequence))
            maxlen = max(len(sentence), maxlen)
            vocab_words.update(set(sentence))
            vocab_tags.update(set(tag_sequence))
        word2idx = {word: idx + 1 for idx, word in enumerate(vocab_words)}
        tag2idx = {tag: idx + 1 for idx, tag in enumerate(vocab_tags)}

    # Encode
    X = [[word2idx[w] if w in vocab_words else len(vocab_words) - 1 for w in sentence] for sentence in sentences]
    y = [[tag2idx[t] for t in tag_sequence] for tag_sequence in tag_sequences]

    
    # Pad Sequences
    X = pad_sequences(maxlen=maxlen, sequences=X, padding="post", value=len(vocab_words)-1)
    y = pad_sequences(maxlen=maxlen, sequences=y, padding="post", value=tag2idx["O"])

    # Make the labels categorical
    y = np.array([to_categorical(i, num_classes=len(vocab_tags) + 1) for i in y])

    metadata = vocab_words, vocab_tags, word2idx, tag2idx, maxlen
    return X, y, metadata
    

def train():

    train_x, train_y, metadata = read_data("../ner_dataset/train")
    test_x, test_y, _ = read_data("../ner_dataset/test", metadata)

    # Unwrap metadata
    vocab_words, vocab_tags, maxlen = metadata[0], metadata[1], metadata[-1]

    input_layer = Input(shape=(maxlen,))
    # model = Sequential()
    # model = Embedding(len(vocab_words) + 1, EMBED_DIM, input_length=maxlen, mask_zero=True)(input_layer) 
    model = Embedding(len(vocab_words) + 1, EMBED_DIM, input_length=maxlen)(input_layer) 
    model = Bidirectional(LSTM(50, return_sequences=True))(model)
    model = TimeDistributed(Dense(50))(model)
    crf = CRF(len(vocab_tags) + 1)
    output_layer = crf(model)
    # output_layer =  Dense(len(vocab_tags) + 1, activation='softmax')(model)
    model = Model(input_layer, output_layer)

    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    # model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
    
    print(model.evaluate(test_x, test_y))
    pred = model.predict(test_x)
    pred_max = pred.argmax(axis=-1)
    test_y_max = test_y.argmax(axis=-1)
    pred_labels = np.array([t for row in pred_max for t in row])
    test_y_labels = np.array([t for row in test_y_max for t in row])
    classes = classes = [k for k in metadata[-2].keys()]
    print(classification_report(test_y_labels, pred_labels, labels=classes))
    # print(classification_report(np.vstack(test_y), np.vstack(predictions)))

    return model, test_x, test_y, metadata[-2]


if __name__ == '__main__':
    train() 