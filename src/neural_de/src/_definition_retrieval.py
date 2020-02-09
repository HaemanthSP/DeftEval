"""
Adapted from: https://github.com/philipperemy/keras-attention-mechanism
"""
import os
import numpy as np
import sys
import spacy
import _data_manager
import gensim
from keras.models import load_model
from collections import defaultdict
from argparse import ArgumentParser

def load_corpus(corpuspath,size):
	return [line.strip().lower() for line in open(corpuspath,'r')][:size]

def get_dependency_repr(sent_list, modelwords, vocabwords, dimwords, maxlen, deps2ids):
	print('Vectorizing dependencies')
	out_wordpairs=[]
	out_deps=[]
	for idx,sent in enumerate(sent_list):
		if idx % 500 == 0:
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

def dep_vectorize(sent, modelwords, vocabwords, dimwords, maxlen, deps2ids):
	sent=nlp(sent)
	word_pairs,pos_pairs,dep_pairs=_data_manager.parse_sent(sent)
	word_pairs_matrix=_data_manager.vectorize_wordpairs(word_pairs,modelwords,vocabwords,dimwords,maxlen,mode='avg')
	dep_labels=[j for i,j in dep_pairs]
	labels_matrix=_data_manager.vectorize_deprels(dep_labels,maxlen,dimwords,deps2ids)
	return word_pairs_matrix,labels_matrix

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
	parser.add_argument('-d', '--corpus-file', help='Corpus file', required=True)
	parser.add_argument('-m', '--max-sents', help='Set the maximum number of sentences to process (important for preprocessing time)', required=True)
	parser.add_argument('-wv', '--word-vectors', help='Vector file with words', required=True)
	parser.add_argument('-p', '--path', help='Use keras model (only dependency enriched)', required=True)
	parser.add_argument('-dep', '--dependencies', help='Option for using dependencies (m=mean, l=label, n=none)', required=True,
		choices=['ml', 'm', 'n'])
	parser.add_argument('-t', '--threshold', help='Confidence threshold for definition retrieval (float between 0 and 1)', required=True)

	args = vars(parser.parse_args())


	corpus=args['corpus_file']
	embeddings_path=args['word_vectors']
	max_sents=int(args['max_sents'])
	threshold=float(args['threshold'])

	print( 'Loading embeddings...')
	### LOAD EMBEDDINGS ###
	if args['word_vectors'] is not None:
		embeddings=args['word_vectors']
		modelwords,vocabwords,dimwords=_data_manager.load_embeddings(embeddings_path)

	print( 'Loading spacy...')
	nlp=spacy.load('en')

	dataset=load_corpus(corpus,max_sents)
	print('Dataset sample:')
	for i in dataset[:5]: print(i)

	deps2ids={}
	depid=0
	for idx,sent in enumerate(dataset):
		if idx % 500 == 0:
			print('Done ',idx,' of ',len(dataset))
		try:
			doc=nlp(sent)
			for token in doc:
				if not token.dep_ in deps2ids:
					deps2ids[token.dep_]=depid
					depid+=1
				for c in token.children:
					if not c.dep_ in deps2ids:
						deps2ids[c.dep_]=depid
						depid+=1
		except UnicodeDecodeError:
			print( 'Cant process sentence: ',sent,' with label: ',label)

	maxlen=428
	print('Maxlen: ',maxlen)	

	ids2deps=dict([(idx,dep) for dep,idx in deps2ids.items()])

		# vectorize
	print('Vectorizing dataset')
	X=[]
	for idx,sent in enumerate(dataset):
		if idx % 500 == 0:
			print('Done ',idx,' of ',len(dataset))
		tokens=[tok.orth_ for tok in nlp(sent.lower())]
		sent_matrix=[]
		for token in pad_words(tokens,maxlen,append_tuple=False):
			if token in vocabwords:
				# each word vector is embedding dim + length of one-hot encoded label
				vec=np.concatenate([modelwords[token],np.zeros(46+1)])
				sent_matrix.append(vec)
			else:
				sent_matrix.append(np.zeros(dimwords+46+1))
		sent_matrix=np.array(sent_matrix)
		X.append(sent_matrix)
		
	X=np.array(X)

	X_wordpairs=[]
	X_deps=[]
	for idx,sent in enumerate(dataset):
		if idx % 10 == 0:
			print('Done ',idx,' of ',len(dataset))
		tokens=nlp(sent.lower())
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
			dep_vec=np.zeros(46+1)
			dep_vec[dep_idx]=1
			avg_label_vec=np.concatenate([avg,dep_vec])
			avg_sent_matrix.append(np.concatenate([avg,np.zeros(46+1)]))
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

	# Load keras model
	print( 'Loading keras model...')
	nnmodel = load_model(args['path'])
	print('Loaded model from: ',args['path'])	
	print('now run the model over each sentence')

	for idx,sent in enumerate(dataset):
		pred=nnmodel.predict(np.array([X_enriched[idx]]))[0][0]
		if pred > threshold:
			print('Sent: ',sent,' -> ',pred)