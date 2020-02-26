import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

from argparse import ArgumentParser
import csv
import sys
import pickle
import numpy as np
from collections import defaultdict
import spacy
import _data_manager
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import gc


class Numberer:
    def __init__(self):
        self.v2n = dict()
        self.n2v = list()
        self.INVALID_NUMBER = 0

    def number(self, value, add_if_absent=False):
        n = self.v2n.get(value)

        if n is None:
            if add_if_absent:
                n = len(self.n2v) + 1
                self.v2n[value] = n
                self.n2v.append(value)
            else:
                n = self.INVALID_NUMBER

        return n


    def value(self, number):
        assert number > self.INVALID_NUMBER
        return self.n2v[number - 1]


    def max_number(self):
        return len(self.n2v)


def pad_words(tokens,maxlen,append_tuple=False):
	if len(tokens) > maxlen:
		return tokens[:maxlen]
	else:
		dif=maxlen-len(tokens)
		for i in range(dif):
			if append_tuple == False:
				tokens.append('<UNK>')
			else:
				tokens.append(('<UNK>','<UNK>'))
		return tokens


def enrich_X(dataset, metadata, w2v_model, POS_EMBED=True):
	maxlen, pos2id, id2pos, w2v_vocab, w2v_dim = metadata
	print('Vectorizing dataset')
	X=[]
	for idx,sent in tqdm(enumerate(dataset.instances)):
		tokens=[tok.orth_ for tok in sent]
		poss=[tok.pos_ if tok.pos_ != "PUNCT" else tok.text for tok in sent]
		sent_matrix=[]
		for t_idx, token in enumerate(pad_words(tokens,maxlen,append_tuple=False)):
			if POS_EMBED:
				pos_vec=np.zeros(len(pos2id)+1, dtype='float32')
				if t_idx < len(poss) and poss[t_idx] in pos2id:
					pos_vec[pos2id[poss[t_idx]]]=1
				else:
					pos_vec[-1]=1
			if token in w2v_vocab:
				# each word vector is embedding dim + length of one-hot encoded label
				if POS_EMBED:
					vec=np.concatenate([w2v_model[token], pos_vec])
				else:
					vec=w2v_model[token]
				sent_matrix.append(vec)
			else:
				if POS_EMBED:
					sent_matrix.append(np.concatenate([np.zeros(w2v_dim, dtype='float32'), pos_vec]))
				else:
					sent_matrix.append(np.concatenate([np.zeros(w2v_dim, dtype='float32')]))
		sent_matrix=np.array(sent_matrix, dtype='float32')
		X.append(sent_matrix)

	X=np.array(X, dtype='float16')
	return X


def encode_X_words(dataset, metadata):
	print('Vectorizing dataset - Words')
	X=[]

	# Unpack metadata
	maxlen, vocab2id, _, _, _ = metadata

	for sent in tqdm(dataset.instances):
		tokens = [token.orth_.lower() for token in sent]
		tokens = pad_words(tokens, maxlen, append_tuple=False)
		sent_matrix = [vocab2id.get(token, 0) for token in tokens]
		X.append(np.array(sent_matrix, dtype='int32'))

	return np.array(X, dtype='int32')


def encode_X_pos(dataset, metadata):
	print('Vectorizing dataset - POS')
	X=[]

	# Unpack metadata
	maxlen, _, _, pos2id, _ = metadata

	for sent in tqdm(dataset.instances):
		pos_tags = [tok.pos_ if tok.pos_ != "PUNCT" or not include_punct else tok.text for tok in sent]
		pos_tags = pad_words(pos_tags, maxlen, append_tuple=False)
		encoded_pos = [pos2id.get(tag, 0) for tag in pos_tags]
		X.append(np.array(encoded_pos, dtype='int8')) # POS tag vocab is < 127, so int8 is large enough

	return np.array(X, dtype='int8')


def save_metadata(meatadata, outpath):
	"""Save the metadata along with the model for future evaluation"""
	print('Saving metadata to: ',outpath + '/meta')
	with open(outpath + '/meta', 'wb') as file_handle:
		pickle.dump(metadata, file_handle)


def evaluate(datapath, model_path):
	# Load dataset
	eval_dataset =_data_manager.Dataset(datapath,'deft')
	eval_dataset.load_deft()

	# Load model and metadata
	metadata = pickle.load(open(model_path + '/meta', 'rb'))
	X_eval_enriched = enrich_X(eval_dataset, metadata, w2v_model)
	X_eval,y_eval=X_eval_enriched,y_deft_dev

	model = load_model(model_path)

	predictions = model.predict(X_eval)
	count = 0
	threshold = 0.5
	print("# Printing only the false predictions")
	for idx, (z, y, x) in enumerate(zip(y_eval, predictions, eval_dataset.instances)):
		if (z and y[0] > threshold) or (not z and y[0] < threshold):
			continue
		count+=1
		print("\t".join([str(z), str(y), x.text]))

	print("Total number of false predictions: %s" %(count))



def codalab_evaluation(data_dir, out_dir, model_path, embedding_data=None, embedding_path=None):
	# Load W2V embedding
	print("Loading w2v embedding")

	if embedding_data:
		w2v_model,w2v_vocab,w2v_dim=embedding_data
	else:
		w2v_model,w2v_vocab,w2v_dim=_data_manager.load_embeddings(embedding_path)


	model = load_model(model_path)
	metadata = pickle.load(open(model_path + '/meta', 'rb'))

	for fname in tqdm(os.listdir(data_dir)):
		print("processing %s" % (fname))
		file_path = os.path.join(data_dir, fname)
		out_path = os.path.join(out_dir, fname)
		# Load dataset
		eval_dataset =_data_manager.Dataset(file_path,'deft')
		eval_dataset.load_deft(ignore_labels=True)

		# Load model and metadata
		X_eval_enriched = enrich_X(eval_dataset, metadata, w2v_model)
		y_eval=np.array(eval_dataset.labels)
		X_eval,y_eval=X_eval_enriched, y_eval

		predictions = model.predict_classes(X_eval)


		with open(out_path, 'w', encoding='utf-8') as fhandle:
			wr = csv.writer(fhandle, delimiter="\t")
			for sentence, pred in zip(eval_dataset.sentences, predictions):
				wr.writerow([sentence, pred[0]])


if __name__ == '__main__':
	# sys.argv = ['./task1.py',
				# '-wv', '..\\resources\\glove.840B.300d.metadata',
				# '-wv', 'C:\\Users\\shadeMe\\Documents\\ML\\Embeddings\\eng\\glove.840B.300d.w2v.txt',
				# '-dep', 'ml',
				# '-p', '..\\resources\\cnn_glove']

	parser = ArgumentParser()
	parser.add_argument('-wv', '--word-vectors', help='Vector file with words', required=True)
	parser.add_argument('-p', '--path', help='Use or save keras model', required=True)
	parser.add_argument('-e', '--eval', help='Evaluate the model', default=False, type=bool)

	args = vars(parser.parse_args())

	print('Loading spacy')
	nlp=spacy.load('en_core_web_lg')

	outpath=args['path']
	os.makedirs(os.path.dirname(outpath), exist_ok=True)

	# Load embedding
	embeddings=args['word_vectors']
	w2v_model,w2v_vocab,w2v_dim=_data_manager.load_embeddings(embeddings)

	if args['eval']:
		codalab_evaluation('../deft_corpus/data/test_files/subtask_1', '../result/task1/', outpath, embedding_data=[w2v_model,w2v_vocab,w2v_dim])
		codalab_evaluation('../task1_data/dev/', '../result/task1_dev/', outpath, embedding_data=[w2v_model,w2v_vocab,w2v_dim])
		sys.exit(0)

	# load datasets
	deft=_data_manager.Dataset('../task1_data/train_combined.deft','deft')
	deft.load_deft()

	deft_dev =_data_manager.Dataset('../task1_data/dev_combined.deft','deft')
	deft_dev.load_deft()

	# load labels as np arrays
	y_deft_dev=np.array(deft_dev.labels, dtype='float32')

	# load labels as np arrays
	y_deft=np.array(deft.labels, dtype='float32')

	# preprocess
	maxlen=0
	# label to integer mapping
	pos2id = {'<UNK>': 0}
	vocab2id = {'<UNK>': 0}
	pos_id = 0
	vocab_id = 0

	### VECTORIZING WCL
	train_vocab = set()
	print('Getting maxlen')
	for doc in tqdm(deft.instances+deft_dev.instances):
		if len(doc) > maxlen:
			maxlen=len(doc)
		for token in doc:
			if not token.pos_ in pos2id:
				if token.pos_ == "PUNCT":
					if token.text not in pos2id:
						pos2id[token.text]=pos_id
						pos_id += 1
				else:
					pos2id[token.pos_]=pos_id
					pos_id += 1
			if not token.orth_.lower() in vocab2id:
				vocab2id[token.orth_.lower()] = vocab_id
				vocab_id += 1

	print('Maxlen: ',maxlen)

	id2vocab= dict([(idx,word) for word,idx in vocab2id.items()])
	id2pos= dict([(idx,pos) for pos,idx in pos2id.items()])

	print("Word vocab size: ", len(id2vocab))
	print("POS vocab size: ", len(id2pos))

	metadata = maxlen, vocab2id, id2vocab, pos2id, id2pos
	X_train_encoded = encode_X_words(deft, metadata)
	X_train,y_train = shuffle(X_train_encoded, y_deft, random_state=0)

	print('Vectorizing deft_dev')
	X_test_encoded = encode_X_words(deft_dev, metadata)
	X_test, y_test = X_test_encoded, y_deft_dev

	early_stopping_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True)

	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

	# Build the embedding matrix 
	embedding_weights = [w2v_model[id2vocab[idx]] if id2vocab[idx] in w2v_vocab else np.zeros(w2v_dim)
						 for idx in range(len(id2vocab))]
	embedding_weights = np.array(embedding_weights, dtype='float32')
	print("Shape of the embedding: ", embedding_weights.shape)

	nnmodel=_data_manager.build_model(X_train,y_train,"cnn",lstm_units=100, embedding_weights=embedding_weights, vocab_size=len(id2vocab))
	gc.collect()

	nnmodel.fit(X_train, y_train,epochs=10,batch_size=128,validation_data=[X_valid, y_valid], callbacks=[early_stopping_callback], class_weight=_data_manager.calculate_class_weights(y_train))
	nnmodel.save(outpath)
	save_metadata(metadata, outpath)
	print('Saving model to: ',outpath)
	eval_loss, eval_precision, eval_recall, eval_acc = nnmodel.evaluate(X_test, y_test)
	print('\nEval Loss: {:.3f}, Eval Precision: {:.3f}, Eval Recall: {:.3f}, Eval Acc: {:.3f}'.format(eval_loss, eval_precision, eval_recall, eval_acc))

	# Evaluate test data
	predictions = nnmodel.predict_classes(X_test)
	preds=np.array([i[0] for i in predictions], dtype='float32')
	print("Confusion Matrix")
	print(confusion_matrix(preds, y_test))
	from sklearn.metrics import classification_report
	print(classification_report(y_test, preds))

	# Evaluate validation data
	predictions2 = nnmodel.predict_classes(X_valid)
	preds2=np.array([i[0] for i in predictions2], dtype='float32')
	print("Confusion Matrix valid")
	print(confusion_matrix(preds2, y_valid))
	from sklearn.metrics import classification_report
	print(classification_report(y_valid, preds2))

	# Analyse the output of the eval
	evaluate('../task1_data/dev_combined.deft', outpath)

	# Redirect the stdout
	# codalab_evaluation('../deft_corpus/data/test_files/subtask_1', '../result/task1/', outpath, embedding_data=[w2v_model,w2v_vocab,w2v_dim])
	# codalab_evaluation('../task1_data/dev/', '../result/task1_eval/', outpath, embedding_data=[w2v_model,w2v_vocab,w2v_dim])