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

def get_maxlen(*args):
	maxlen=-999
	for dataset in args:
		for sent in dataset:
			try:
				doc=nlp(unicode(sent))
				if len(doc) > maxlen:
					maxlen=len(doc)
			except UnicodeDecodeError:
				print('Cant process sentence: ',sent,' with label: ',label)
	return maxlen

def get_dependency_repr(sent_list, modelwords, vocabwords, dimwords, maxlen, deps2ids):
	out_wordpairs=[]
	out_deps=[]
	for idx,sent in enumerate(sent_list):
		if idx % 100 == 0:
			print('Done ',idx,' of ',len(sent_list))
		sent=nlp(sent)
		word_pairs,pos_pairs,dep_pairs=_data_manager.parse_sent(sent)
		word_pairs_matrix=_data_manager.vectorize_wordpairs(word_pairs,modelwords,vocabwords,dimwords,maxlen,mode='avg')
		dep_labels=[j for i,j in dep_pairs]
		labels_matrix=_data_manager.vectorize_deprels(dep_labels,maxlen,dimwords,deps2ids)
		out_wordpairs.append(word_pairs_matrix)
		out_deps.append(labels_matrix)
	out_wordpairs=np.array(out_wordpairs, dtype='float32')
	out_deps=np.array(out_deps, dtype='float32')
	return out_wordpairs,out_deps

def pad_words(tokens,maxlen,append_tuple=False):
	if len(tokens) > maxlen:
		return tokens[:maxlen]
	else:
		dif=maxlen-len(tokens)
		for i in range(dif):
			if append_tuple == False:
				tokens.append('UNK')
			else:
				tokens.append(('UNK','UNK'))
		return tokens

def enrich_X(dataset, metadata, modelwords):
	maxlen, deps2ids, ids2deps, poss2ids, ids2poss, vocabwords, dimwords, dependencies = metadata
	print('Vectorizing dataset')
	X=[]
	for idx,sent in tqdm(enumerate(dataset.instances)):
		# tokens=[tok.orth_ for tok in nlp(sent.lower())]
		tokens=[tok.orth_ for tok in sent]
		poss=[tok.pos_ if tok.pos_ != "PUNCT" else tok.text for tok in sent]
		sent_matrix=[]
		for t_idx, token in enumerate(pad_words(tokens,maxlen,append_tuple=False)):
			pos_vec=np.zeros(len(poss2ids)+1, dtype='float32')
			if t_idx < len(poss) and poss[t_idx] in poss2ids:
				pos_vec[poss2ids[poss[t_idx]]]=1
			else:
				pos_vec[-1]=1
			if token in vocabwords:
				# each word vector is embedding dim + length of one-hot encoded label
				vec=np.concatenate([modelwords[token],np.zeros(len(ids2deps)+1, dtype='float32'), pos_vec])
				sent_matrix.append(vec)
			else:
				# TODO add POS here
				sent_matrix.append(np.concatenate([np.zeros(dimwords+len(ids2deps)+1, dtype='float32'), pos_vec]))
		sent_matrix=np.array(sent_matrix, dtype='float32')
		X.append(sent_matrix)

	X=np.array(X, dtype='float16')

	X_wordpairs=[]
	X_deps=[]
	print("Build dependency relations")
	for idx,sent in tqdm(enumerate(dataset.instances)):
		# tokens=nlp(sent.lower())
		tokens=sent
		word_pairs=[]
		dep_pairs=[]
		for tok in tokens:
			for c in tok.children:
				word_pairs.append((tok.orth_,c.orth_))
				dep_pairs.append((tok.dep_,c.dep_))
		padded_wp=pad_words(word_pairs,maxlen,append_tuple=True)
		padded_deps=pad_words(dep_pairs,maxlen,append_tuple=True)
		dep_labels=[j for i,j in dep_pairs]
		avg_sent_matrix=[]
		avg_label_sent_matrix=[]
		for idx,word_pair in enumerate(word_pairs):
			head,modifier=word_pair[0],word_pair[1]
			if head in vocabwords and not head=='UNK':
				head_vec=modelwords[head]
			else:
				head_vec=np.zeros(dimwords)
			if modifier in vocabwords and not modifier=='UNK':
				modifier_vec=modelwords[modifier]
			else:
				modifier_vec=np.zeros(dimwords)

			# TODO: Can do something better than averaging
			avg=_data_manager.avg(np.array([head_vec,modifier_vec], dtype='float32'))
			if dep_labels[idx] != 'UNK':
				dep_idx=deps2ids[dep_labels[idx]]
			else:
				dep_idx=-1
			dep_vec=np.zeros(len(deps2ids)+len(poss2ids)+2)
			dep_vec[dep_idx]=1
			avg_label_vec=np.concatenate([avg,dep_vec])
			avg_sent_matrix.append(np.concatenate([avg,np.zeros(len(deps2ids)+len(poss2ids)+2)]))
			avg_label_sent_matrix.append(avg_label_vec)
		wp=np.array(avg_sent_matrix, dtype='float16')
		labs=np.array(avg_label_sent_matrix, dtype='float16')
		X_wordpairs.append(wp)
		X_deps.append(labs)

	gc.collect()

	X_wordpairs=np.array(X_wordpairs, dtype='float16')
	X_deps=np.array(X_deps, dtype='float16')

	if dependencies == 'ml':
		X_enriched=np.concatenate([X,X_deps],axis=1)
	elif dependencies == 'm':
		X_enriched=np.concatenate([X,X_wordpairs],axis=1)
	else:
		X_enriched=np.concatenate([X, X_deps, X_wordpairs],axis=1)

	return X_enriched


def save_metadata(meatadata, outpath):
	"""Save the metadata along with the model for future evaluation"""
	print('Saving model to: ',outpath)
	with open(outpath + '/meta', 'wb') as file_handle:
		pickle.dump(metadata, file_handle)


def evaluate(datapath, model_path):
	# Load dataset
	eval_dataset =_data_manager.Dataset(datapath,'deft')
	eval_dataset.load_deft()

	# Load model and metadata
	metadata = pickle.load(open(model_path + '/meta', 'rb'))
	X_eval_enriched = enrich_X(eval_dataset, metadata, modelwords)
	X_eval,y_eval=X_eval_enriched,y_deft_dev

	# model = _data_manager.build_model(X_eval,y_eval,"cblstm",lstm_units=100)
	model = load_model(model_path)

	predictions = model.predict(X_eval)
	count = 0
	for idx, (z, y, x) in enumerate(zip(y_eval, predictions, eval_dataset.instances)):
		if np.random.randint(2):
			count+=1
			print("\t".join([str(z), str(y), x.text]))
			if count > 30:
				break


def calculate_class_weights(dataset):
	def get_class_distribution(labels):
		num_pos = 0
		num_neg = 0
		for data in labels:
			if data == 1:
				num_pos += 1
			else:
				num_neg += 1
		return num_pos, num_neg

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


def codalab_evaluation(data_dir, out_dir, model_path, embedding_data=None, embedding_path=None):
	# Load W2V embedding
	print("Loading w2v embedding")

	if embedding_data:
		modelwords,vocabwords,dimwords=embedding_data
	else:
		modelwords,vocabwords,dimwords=_data_manager.load_embeddings(embedding_path)


	model = load_model(model_path)
	metadata = pickle.load(open(model_path + '/meta', 'rb'))

	for fname in os.listdir(data_dir):
		print("processing %s" % (fname))
		file_path = os.path.join(data_dir, fname)
		out_path = os.path.join(out_dir, fname)
		# Load dataset
		eval_dataset =_data_manager.Dataset(file_path,'deft')
		eval_dataset.load_deft()

		# Load model and metadata
		X_eval_enriched = enrich_X(eval_dataset, metadata, modelwords)
		y_eval=np.array(eval_dataset.labels)
		X_eval,y_eval=X_eval_enriched, y_eval

		predictions = model.predict_classes(X_eval)

		with open(out_path, 'w') as fhandle:
			wr = csv.writer(fhandle, delimiter="\t")
			for sentence, pred in zip(eval_dataset.sentences, predictions):
				wr.writerow([sentence, pred[0]])



if __name__ == '__main__':
	sys.argv = ['./task1.py',
				'-wv', 'C:\\Users\\shadeMe\\Documents\\ML\\Embeddings\\eng\\glove.840B.300d.w2v.txt',#glove.6B.200d.w2vformat.txt',
				'-dep', 'ml',
				'-p', '..\\resources']

	parser = ArgumentParser()

	parser.add_argument('-wv', '--word-vectors', help='Vector file with words', required=True)
	parser.add_argument('-dep', '--dependencies', help='Option for using dependencies (m=mean, l=label, n=none)', required=True,
		choices=['ml', 'm', 'n'])
	parser.add_argument('-p', '--path', help='Use or save keras model', required=True)

	args = vars(parser.parse_args())

	print('Loading spacy')
	nlp=spacy.load('en_core_web_lg')

	outpath=args['path']

	# load datasets
	deft=_data_manager.Dataset('../task1_data/train_combined.deft','deft')
	deft.load_deft()

	deft_dev =_data_manager.Dataset('../task1_data/dev_combined.deft','deft')
	deft_dev.load_deft()

	# load labels as np arrays
	y_deft_dev=np.array(deft_dev.labels, dtype='float32')

	embeddings=args['word_vectors']
	modelwords,vocabwords,dimwords=_data_manager.load_embeddings(embeddings)

	# load labels as np arrays
	y_deft=np.array(deft.labels, dtype='float32')

	# preprocess
	# get token and dependencies (head, modifier) maxlens and pos and dep ids
	maxlen=0
	maxlen_dep=0
	# label to integer mapping
	deps2ids={}
	poss2ids={}
	depid=0
	posid=0

	### VECTORIZING WCL
	print('Getting maxlen')
	for idx,sent in tqdm(enumerate(deft.instances+deft_dev.instances)):
		try:
			sent_maxlen_dep=0
			# doc=nlp(sent)
			doc=sent
			if len(doc) > maxlen:
				maxlen=len(doc)
			for token in doc:
				if not token.dep_ in deps2ids:
					deps2ids[token.dep_]=depid
					depid+=1
				if not token.pos_ in poss2ids:
					if token.pos_ == "PUNCT":
						if token.text not in poss2ids:
							poss2ids[token.text]=posid
							posid+=1
					else:
						poss2ids[token.pos_]=posid
						posid+=1
				for c in token.children:
					if not c.dep_ in deps2ids:
						deps2ids[c.dep_]=depid
						depid+=1
					sent_maxlen_dep+=1
			if sent_maxlen_dep > maxlen_dep:
				maxlen_dep=sent_maxlen_dep
		except UnicodeDecodeError:
			print( 'Cant process sentence: ',sent,' with label: ',label)

	maxlen=max(maxlen,maxlen_dep)

	print('Maxlen: ',maxlen)

	ids2deps=dict([(idx,dep) for dep,idx in deps2ids.items()])
	ids2poss=dict([(idx,pos) for pos,idx in poss2ids.items()])

	# vectorize wcl, needs to be done in second pass to have maxlen
	metadata = maxlen, deps2ids, ids2deps, poss2ids, ids2poss, vocabwords, dimwords, args['dependencies']
	X_train_enriched = enrich_X(deft, metadata, modelwords)
	X_train,y_train=shuffle(X_train_enriched,y_deft,random_state=0)

	### VECTORIZING W00

	# no maxlen, no ids2deps


	# vectorize deft_dev, needs to be done in second pass to have maxlen
	print('Vectorizing deft_dev')
	# X_test,y_test=shuffle(X_enriched,y_deft_dev,random_state=0)
	X_test_enriched = enrich_X(deft_dev, metadata, modelwords)
	X_test,y_test=X_test_enriched,y_deft_dev

	early_stopping_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True)

	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.02, random_state=42)
	nnmodel=_data_manager.build_model(X_train,y_train,"cblstm",lstm_units=100)
	gc.collect()

	nnmodel.fit(X_train, y_train,epochs=10,batch_size=128,validation_data=[X_valid, y_valid], callbacks=[early_stopping_callback], class_weight=calculate_class_weights(y_train))
	nnmodel.save(outpath)
	save_metadata(metadata, outpath)
	print('Saving model to: ',outpath)
	eval_loss, eval_precision, eval_recall, eval_acc = nnmodel.evaluate(X_test, y_test)
	print('\nEval Loss: {:.3f}, Eval Precision: {:.3f}, Eval Recall: {:.3f}, Eval Acc: {:.3f}'.format(eval_loss, eval_precision, eval_recall, eval_acc))
	predictions = nnmodel.predict_classes(X_test)
	preds=np.array([i[0] for i in predictions], dtype='float32')
	print("Confusion Matrix")
	print(confusion_matrix(preds, y_test))
	from sklearn.metrics import classification_report
	print(classification_report(y_test, preds))

	predictions2 = nnmodel.predict_classes(X_valid)
	preds2=np.array([i[0] for i in predictions2], dtype='float32')
	print("Confusion Matrix valid")
	print(confusion_matrix(preds2, y_valid))
	from sklearn.metrics import classification_report
	print(classification_report(y_valid, preds2))

	# save_metadata(metadata, outpath)
	evaluate('../task1_data/dev_combined.deft', outpath)
	codalab_evaluation('../task1_data/dev/', '../result/task1/', outpath, embedding_data=[modelwords,vocabwords,dimwords])