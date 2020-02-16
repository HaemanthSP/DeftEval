from tqdm import tqdm
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Bidirectional, SimpleRNN, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences


from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss
from tf_crf_layer.metrics import crf_accuracy

vocab = 3000
EMBED_DIM = 300
BiRNN_UNITS = 48
class_labels_number = 9
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

    metadata = vocab_words, vocab_tags, word2idx, tag2idx, maxlen
    return X, y, metadata
    

def train():

    train_x, train_y, metadata = read_data("../ner_dataset/train")
    test_x, test_y, _ = read_data("../ner_dataset/test", metadata)

    model = Sequential()
    model.add(Embedding(len(metadata[0]) + 1, EMBED_DIM, mask_zero=True)) 
    model.add(Bidirectional(SimpleRNN(50, return_sequences=True)))
    model.add(Dense(50))
    model.add(CRF(class_labels_number))

    model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])


if __name__ == '__main__':
    train() 