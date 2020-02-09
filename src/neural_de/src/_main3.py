from argparse import ArgumentParser
import sys
sys.path.append('src')
import numpy as np
from collections import defaultdict
import spacy
import _data_manager
import os
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score
from keras.models import load_model
from keras.callbacks import EarlyStopping
from tqdm import tqdm

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
	out_wordpairs=np.array(out_wordpairs)
	out_deps=np.array(out_deps)
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

if __name__ == '__main__':

	parser = ArgumentParser()

	parser.add_argument('-wv', '--word-vectors', help='Vector file with words', required=True)
	parser.add_argument('-dep', '--dependencies', help='Option for using dependencies (m=mean, l=label, n=none)', required=True,
		choices=['ml', 'm', 'n'])
	parser.add_argument('-p', '--path', help='Use or save keras model', required=True)

	args = vars(parser.parse_args())

	print('Loading spacy')
	nlp=spacy.load('en_core_web_lg')

	outpath=args['path']

	# wcl is dataset of wikipedia defs (Navigli and Velardi, 2010 ACL)
	deft=_data_manager.Dataset('data/deft_dataset/train.deft','deft')
	deft.load_deft()

	# load datasets
	# w00 is manually selected defs from acl anthology (Jin et al., 2013 EMNLP)
	deft_dev =_data_manager.Dataset('data/deft_dataset/dev.deft','deft')
	deft_dev.load_deft()

	# load labels as np arrays
	y_deft_dev=np.array(deft_dev.labels)

	embeddings=args['word_vectors']
	modelwords,vocabwords,dimwords=_data_manager.load_embeddings(embeddings)

	# load labels as np arrays
	y_deft=np.array(deft.labels)

	# preprocess
	# get token and dependencies (head, modifier) maxlens and pos and dep ids
	maxlen=0
	maxlen_dep=0
	# label to integer mapping
	deps2ids={}
	depid=0
	### VECTORIZING WCL
	print('Getting maxlen')
	for idx,sent in enumerate(deft.instances+deft_dev.instances):
		if idx % 500 == 0:
			print('Done ',idx,' of ',len(deft.instances+deft_dev.instances))
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

	# vectorize wcl, needs to be done in second pass to have maxlen
	print('Vectorizing deft')
	X=[]
	for idx,sent in enumerate(deft.instances):
		if idx % 500 == 0:
			print('Done ',idx,' of ',len(deft.instances))
		# tokens=[tok.orth_ for tok in nlp(sent.lower())]
		tokens=[tok.orth_ for tok in sent]
		sent_matrix=[]
		for token in pad_words(tokens,maxlen,append_tuple=False):
			if token in vocabwords:
				# each word vector is embedding dim + length of one-hot encoded label
				vec=np.concatenate([modelwords[token],np.zeros(len(ids2deps)+1)])
				sent_matrix.append(vec)
			else:
				sent_matrix.append(np.zeros(dimwords+len(ids2deps)+1))
		sent_matrix=np.array(sent_matrix)
		X.append(sent_matrix)

	X=np.array(X)

	X_wordpairs=[]
	X_deps=[]
	print("Build dependency relations")
	for idx,sent in enumerate(deft.instances):
		if idx % 500 == 0:
			print('Done ',idx,' of ',len(deft.instances))
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
			avg=_data_manager.avg(np.array([head_vec,modifier_vec]))
			if dep_labels[idx] != 'UNK':
				dep_idx=deps2ids[dep_labels[idx]]
			else:
				dep_idx=-1
			dep_vec=np.zeros(len(deps2ids)+1)
			dep_vec[dep_idx]=1
			avg_label_vec=np.concatenate([avg,dep_vec])
			avg_sent_matrix.append(np.concatenate([avg,np.zeros(len(deps2ids)+1)]))
			avg_label_sent_matrix.append(avg_label_vec)
		wp=np.array(avg_sent_matrix)
		labs=np.array(avg_label_sent_matrix)
		X_wordpairs.append(wp)
		X_deps.append(labs)

	X_wordpairs=np.array(X_wordpairs)
	X_deps=np.array(X_deps)

	
	if args['dependencies'] == 'ml':
		X_enriched=np.concatenate([X,X_deps],axis=1)
	elif args['dependencies'] == 'm':
		X_enriched=np.concatenate([X,X_wordpairs],axis=1)
	else:
		X_enriched=X

	X_train,y_train=shuffle(X_enriched,y_deft,random_state=0)

	### VECTORIZING W00

	# no maxlen, no ids2deps

	# vectorize deft_dev, needs to be done in second pass to have maxlen
	print('Vectorizing deft_dev')
	X=[]
	for idx,sent in enumerate(deft_dev.instances):
		if idx % 500 == 0:
			print('Done ',idx,' of ',len(deft_dev.instances))
		# tokens=[tok.orth_ for tok in nlp(sent.lower())]
		tokens=[tok.orth_ for tok in sent]
		sent_matrix=[]
		for token in pad_words(tokens,maxlen,append_tuple=False):
			if token in vocabwords:
				# each word vector is embedding dim + length of one-hot encoded label
				vec=np.concatenate([modelwords[token],np.zeros(len(ids2deps)+1)])
				sent_matrix.append(vec)
			else:
				sent_matrix.append(np.zeros(dimwords+len(ids2deps)+1))
		sent_matrix=np.array(sent_matrix)
		X.append(sent_matrix)

	X=np.array(X)

	X_wordpairs=[]
	X_deps=[]
	for idx,sent in enumerate(deft_dev.instances):
		if idx % 10 == 0:
			print('Done ',idx,' of ',len(deft_dev.instances))
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
			avg=_data_manager.avg(np.array([head_vec,modifier_vec]))
			if dep_labels[idx] != 'UNK':
				dep_idx=deps2ids[dep_labels[idx]]
			else:
				dep_idx=-1
			dep_vec=np.zeros(len(deps2ids)+1)
			dep_vec[dep_idx]=1
			avg_label_vec=np.concatenate([avg,dep_vec])
			avg_sent_matrix.append(np.concatenate([avg,np.zeros(len(deps2ids)+1)]))
			avg_label_sent_matrix.append(avg_label_vec)
		wp=np.array(avg_sent_matrix)
		labs=np.array(avg_label_sent_matrix)
		X_wordpairs.append(wp)
		X_deps.append(labs)

	X_wordpairs=np.array(X_wordpairs)
	X_deps=np.array(X_deps)

	
	if args['dependencies'] == 'ml':
		X_enriched=np.concatenate([X,X_deps],axis=1)
	elif args['dependencies'] == 'm':
		X_enriched=np.concatenate([X,X_wordpairs],axis=1)
	else:
		X_enriched=X

	X_test,y_test=shuffle(X_enriched,y_deft_dev,random_state=0)

	early_stopping_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True)

	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
	nnmodel=_data_manager.build_model(X_train,y_train,"cblstm",lstm_units=100)
	nnmodel.fit(X_train, y_train,epochs=10,batch_size=64,validation_data=[X_valid, y_valid], callbacks=[early_stopping_callback])
	preds=np.array([i[0] for i in nnmodel.predict_classes(X_test)])
	from sklearn.metrics import classification_report
	print(classification_report(y_test, preds))
	print('Saving model to: ',outpath)
	nnmodel.save(outpath)