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
from _data_manager import F1Score
from tensorflow.keras import metrics, regularizers


def pad_words(tokens,maxlen,append_tuple=False,pad_token='<UNK>'):
	if len(tokens) > maxlen:
		return tokens[:maxlen]
	else:
		dif=maxlen-len(tokens)
		for i in range(dif):
			if append_tuple == False:
				tokens.append(pad_token)
			else:
				tokens.append((pad_token, pad_token))
		return tokens


def encode_X_words(dataset, metadata):
	print('Vectorizing dataset - Words')
	X=[]

	# Unpack metadata
	maxlen, vocab2id, _, _, _, = metadata

	for sent in tqdm(dataset.instances):
		tokens = [token.orth_.lower() for token in sent]
		tokens = pad_words(tokens, maxlen, append_tuple=False)
		sent_matrix = [vocab2id.get(token, 0) for token in tokens]
		X.append(np.array(sent_matrix, dtype='int32'))

	return np.array(X, dtype='int32')


def encode_X_pos(dataset, metadata, include_punct=True):
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


def encode_X_deps(dataset, metadata):
	_,vocab2id,_,_,_,dep2id,_,dep_maxlen = metadata

	X_head, X_modifier, X_dep_labels = [], [], []
	print("Build dependency relations")
	for idx, tokens in tqdm(enumerate(dataset.instances)):
		head, modifier, dep_labels = [], [], []
		dep_pairs=[]
		for tok in tokens:
			for c in tok.children:
				head.append(vocab2id.get(tok.orth_.lower(), 0))
				modifier.append(vocab2id.get(c.orth_.lower(), 0))
				dep_labels.append(dep2id.get(c.dep_, 0))
		X_head.append(np.array(pad_words(head, dep_maxlen, append_tuple=False, pad_token=0), dtype='int32'))
		X_modifier.append(np.array(pad_words(modifier, dep_maxlen, append_tuple=False, pad_token=0), dtype='int32'))
		X_dep_labels.append(np.array(pad_words(dep_labels, dep_maxlen, append_tuple=False, pad_token=0), dtype='int8'))

	X_head = np.array(X_head, dtype='int32')
	X_modifier = np.array(X_modifier, dtype='int32')
	X_dep_labels = np.array(X_dep_labels, dtype='int8')
	return X_head, X_modifier, X_dep_labels


def save_metadata(meatadata, outpath):
	"""Save the metadata along with the model for future evaluation"""
	print('Saving metadata to: ',outpath + '/meta')
	with open(outpath + '/meta', 'wb') as file_handle:
		pickle.dump(metadata, file_handle)


def decode_words(X_words, id2vocab):
	import re
	sent = " ".join([id2vocab[word] if word else '<UNK>' for word in X_words])
	return re.sub(r'(<UNK> ){2,}', '<PADDING>', sent)


def evaluate(datapath, model_path):
	# Load dataset
	eval_dataset =_data_manager.Dataset(datapath,'deft')
	eval_dataset.load_deft()

	# Load model and metadata
	metadata = pickle.load(open(model_path + '/meta', 'rb'))
	_, _, id2vocab, _, id2pos = metadata
	X_eval_word = encode_X_words(eval_dataset, metadata)
	X_eval_pos = encode_X_pos(eval_dataset, metadata)
	y_eval = np.array(eval_dataset.labels, dtype='float32')

	model = load_model(model_path, compile=False)
	model.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=[metrics.Precision(), metrics.Recall(), 'accuracy', F1Score()])

	predictions = model.predict_on_batch([X_eval_word, X_eval_pos])
	preds=np.array([1 if i[0]>0.5 else 0 for i in predictions], dtype='float32')
	print("Confusion Matrix")
	print(confusion_matrix(preds, y_eval))
	from sklearn.metrics import classification_report
	print(classification_report(y_eval, preds))

	count = 0
	threshold = 0.5
	print("# Printing only the false predictions")
	# for idx, (z, y, x) in enumerate(zip(y_eval, predictions, eval_dataset.instances)):
	for idx, (z, y, xw, xp) in enumerate(zip(y_eval, predictions, X_eval_word, X_eval_pos)):
		if (z and y[0] > threshold) or (not z and y[0] < threshold):
			continue
		count+=1
		# print("\t".join([str(z), str(y.numpy()[0]), decode_words(xw, id2vocab), decode_words(xp, id2pos)]))
		print("\t".join([str(z), str(y), decode_words(xw, id2vocab), decode_words(xp, id2pos)]))

	print("Total number of false predictions: %s" %(count))



def codalab_evaluation(data_dir, out_dir, model_path, embedding_data=None, embedding_path=None):
	# Load W2V embedding
	print("Loading w2v embedding")

	if embedding_data:
		w2v_model,w2v_vocab,w2v_dim=embedding_data
	else:
		w2v_model,w2v_vocab,w2v_dim=_data_manager.load_embeddings(embedding_path)


	model = load_model(model_path, compile=False)
	model.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=[metrics.Precision(), metrics.Recall(), 'accuracy', F1Score()])
	metadata = pickle.load(open(model_path + '/meta', 'rb'))

	for fname in tqdm(os.listdir(data_dir)):
		print("processing %s" % (fname))
		file_path = os.path.join(data_dir, fname)
		out_path = os.path.join(out_dir, fname)
		# Load dataset
		eval_dataset =_data_manager.Dataset(file_path,'deft')
		eval_dataset.load_deft(ignore_labels=True)


		X_eval_word = encode_X_words(eval_dataset, metadata)
		X_eval_pos = encode_X_pos(eval_dataset, metadata)
		# y_eval = np.array(eval_dataset.labels, dtype='float32')

		predictions = model.predict_on_batch([X_eval_word, X_eval_pos])


		with open(out_path, 'w', encoding='utf-8') as fhandle:
			wr = csv.writer(fhandle, delimiter="\t")
			for sentence, pred in zip(eval_dataset.sentences, predictions):
				wr.writerow([sentence, 1 if pred[0] > 0.5 else 0])


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
		# evaluate()
		evaluate('../task1_data/dev_combined.deft', outpath)
		# codalab_evaluation('../deft_corpus/data/test_files/subtask_1', '../result/task1/', outpath, embedding_data=[w2v_model,w2v_vocab,w2v_dim])
		# codalab_evaluation('../task1_data/dev/', '../result/task1_dev/', outpath, embedding_data=[w2v_model,w2v_vocab,w2v_dim])
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
	dep_maxlen=0

	# label to integer mapping
	pos2id = {'<UNK>': 0}
	dep2id = {'<UNK>': 0}
	vocab2id = {'<UNK>': 0}
	pos_id = 1
	dep_id = 1
	vocab_id = 1

	### VECTORIZING WCL
	train_vocab = set()
	print('Getting maxlen')
	for doc in tqdm(deft.instances+deft_dev.instances):
		sent_dep_maxlen = 0
		if len(doc) > maxlen:
			maxlen=len(doc)
		for token in doc:
			# Pos tags
			if not token.pos_ in pos2id:
				if token.pos_ == "PUNCT":
					if token.text not in pos2id:
						pos2id[token.text]=pos_id
						pos_id += 1
				else:
					pos2id[token.pos_]=pos_id
					pos_id += 1

			# Word tokens
			if not token.orth_.lower() in vocab2id:
				vocab2id[token.orth_.lower()] = vocab_id
				vocab_id += 1

			# Dep tags
			if not token.dep_ in dep2id:
				dep2id[token.dep_]=dep_id
				dep_id+=1
			for c in token.children:
				if not c.dep_ in dep2id:
					dep2id[c.dep_]=dep_id
					dep_id+=1
				sent_dep_maxlen +=1

			if sent_dep_maxlen > dep_maxlen:
				dep_maxlen = sent_dep_maxlen

	print('Maxlen: ',maxlen)

	id2vocab= dict([(idx,word) for word,idx in vocab2id.items()])
	id2pos= dict([(idx,pos) for pos,idx in pos2id.items()])
	id2dep= dict([(idx,dep) for dep,idx in dep2id.items()])

	print("Word vocab size: ", len(id2vocab))
	print("POS vocab size: ", len(id2pos))
	print("Dep vocab size: ", len(id2dep))

	# metadata = maxlen, vocab2id, id2vocab, pos2id, id2pos, dep2id, id2dep, dep_maxlen
	metadata = maxlen, vocab2id, id2vocab, pos2id, id2pos
	X_train_word = encode_X_words(deft, metadata)
	# X_train_pos = encode_X_pos(deft, metadata)
	# X_train_head, X_train_modifier, X_train_deps = encode_X_deps(deft, metadata)
	X_train_word, y_train = shuffle(X_train_word, y_deft, random_state=0)
	# X_train_word, X_train_pos, y_train = shuffle(X_train_word, X_train_pos, y_deft, random_state=0)
	# X_train_word, X_train_pos, X_train_head, X_train_modifier, X_train_deps, y_train = shuffle(
		# X_train_word, X_train_pos, X_train_head, X_train_modifier, X_train_deps, y_deft, random_state=0)

	print("\n\n\ntraining shape: ", X_train_word.shape)
	print("data sample: ", X_train_word[0])

	print('Vectorizing deft_dev')
	X_test_word = encode_X_words(deft_dev, metadata)
	# X_test_pos = encode_X_pos(deft_dev, metadata)
	# X_test_head, X_test_modifier, X_test_deps = encode_X_deps(deft_dev, metadata)
	y_test = y_deft_dev

	early_stopping_callback = EarlyStopping(
        # monitor='val_F1Score', min_delta=0.0001, patience=10, restore_best_weights=True)
        monitor='val_loss', min_delta=0.001, patience=10, restore_best_weights=True)

	X_train_word, X_valid_word, y_train, y_valid = train_test_split(X_train_word, y_train, test_size=0.10, random_state=42)
	# X_train_word, X_valid_word, X_train_pos, X_valid_pos, y_train, y_valid = train_test_split(X_train_word, X_train_pos, y_train, test_size=0.10, random_state=42)
	# X_train_word, X_valid_word, X_train_pos, X_valid_pos, X_train_head, X_valid_head, X_train_modifier, X_valid_modifier, X_train_deps, X_valid_deps, y_train, y_valid = train_test_split(X_train_word, X_train_pos, X_train_head, X_train_modifier, X_train_deps, y_train, test_size=0.10, random_state=42)

	# valid_inps = [X_valid_word, X_valid_pos, X_valid_head, X_valid_modifier, X_valid_deps]
	# train_inps = [X_train_word, X_train_pos, X_train_head, X_train_modifier, X_train_deps]
	# test_inps = [X_test_word, X_test_pos, X_test_head, X_test_modifier, X_test_deps]

	# valid_inps = [X_valid_word, X_valid_pos]
	# train_inps = [X_train_word, X_train_pos]
	# test_inps = [X_test_word, X_test_pos]

	valid_inps = X_valid_word
	train_inps = X_train_word
	test_inps = X_test_word

	print("Validation Inputs")
	print(valid_inps)
	print("\n\n")
	# Build the embedding matrix 
	embedding_weights = [w2v_model[id2vocab[idx]] if id2vocab[idx] in w2v_vocab else np.zeros(w2v_dim)
						 for idx in range(len(id2vocab))]
	embedding_weights = np.array(embedding_weights, dtype='float32')
	print("Shape of the embedding: ", embedding_weights.shape)

	# nnmodel=_data_manager.build_model(X_train,y_train,"cnn",lstm_units=100, embedding_weights=embedding_weights, vocab_size=len(id2vocab))
	# nnmodel=_data_manager.build_model2(maxlen, embedding_weights=[embedding_weights, None], vocab_size=[len(id2vocab), len(id2pos)])
	nnmodel=_data_manager.build_baseline_bilstm(maxlen, vocab_size=len(id2vocab))
	# nnmodel=_data_manager.build_model3([maxlen, dep_maxlen], embedding_weights=[embedding_weights, None], vocab_size=[len(id2vocab), len(id2pos), len(id2dep)])
	gc.collect()

	# nnmodel.fit([X_train_word, X_train_pos], y_train,epochs=25,batch_size=128,validation_data=[[X_valid_word, X_valid_pos], y_valid], callbacks=[early_stopping_callback], class_weight=_data_manager.calculate_class_weights(y_train))
	nnmodel.fit(train_inps,
			    y_train,
				epochs=100,
				batch_size=64,
				validation_data=(valid_inps, y_valid),
				callbacks=[early_stopping_callback])
				# class_weight=_data_manager.calculate_class_weights(y_train))
	nnmodel.save(outpath)
	save_metadata(metadata, outpath)
	print('Saving model to: ',outpath)
	eval_loss, eval_precision, eval_recall, eval_acc, eval_f1  = nnmodel.evaluate(test_inps, y_test)
	print('\nEval Loss: {:.3f}, Eval Precision: {:.3f}, Eval Recall: {:.3f}, Eval Acc: {:.3f}, Eval F1: {:.3f}'.format(eval_loss, eval_precision, eval_recall, eval_acc, eval_f1))

	# Evaluate test data
	predictions = nnmodel.predict_on_batch(test_inps)
	preds=np.array([1 if i[0]>0.5 else 0 for i in predictions], dtype='float32')
	print("Confusion Matrix")
	print(confusion_matrix(preds, y_test))
	from sklearn.metrics import classification_report
	print(classification_report(y_test, preds))

	# Evaluate validation data
	predictions2 = nnmodel.predict_on_batch(valid_inps)
	preds2=np.array([1 if i[0]>0.5 else 0 for i in predictions2], dtype='float32')
	print("Confusion Matrix valid")
	print(confusion_matrix(preds2, y_valid))
	from sklearn.metrics import classification_report
	print(classification_report(y_valid, preds2))

	# Analyse the output of the eval
	# evaluate('../task1_data/dev_combined.deft', outpath)

	# Redirect the stdout
	# codalab_evaluation('../deft_corpus/data/test_files/subtask_1', '../result/task1/', outpath, embedding_data=[w2v_model,w2v_vocab,w2v_dim])
	# codalab_evaluation('../task1_data/dev/', '../result/task1_eval/', outpath, embedding_data=[w2v_model,w2v_vocab,w2v_dim])