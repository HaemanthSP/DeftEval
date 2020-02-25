import gensim
import os
import sys
import spacy
import numpy as np
from tensorflow.keras import metrics, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Bidirectional, Dropout, Conv1D, MaxPooling1D, Embedding, Flatten, LSTM
from tqdm import tqdm
import pickle
nlp=spacy.load('en_core_web_lg')


def build_model(x,y,model_type,lstm_units=100,validation_data=''):
	# hyperparams
	kernel_size = 4
	filters = 50
	pool_size = 2
	strides=1
	# train opts
	epochs=100
	batch_size=100
	nnmodel = Sequential()
	nnmodel.add(Conv1D(256,
		kernel_size,
		padding='valid',
		activation='relu',
		strides=strides,
		input_shape=(x.shape[1], x.shape[2])))
	nnmodel.add(Conv1D(256,
		kernel_size,
		padding='valid',
		activation='relu',
		strides=strides))
	nnmodel.add(Conv1D(256,
		kernel_size,
		padding='valid',
		activation='relu',
		strides=strides))
	# nnmodel.add(MaxPooling1D(pool_size=pool_size))
	# nnmodel.add(Conv1D(128,
		# kernel_size,
		# padding='valid',
		# activation='relu',
		# strides=strides))
	# nnmodel.add(Conv1D(128,
		# kernel_size,
		# padding='valid',
		# activation='relu',
		# strides=strides))
	# nnmodel.add(Conv1D(128,
		# kernel_size,
		# padding='valid',
		# activation='relu',
		# strides=strides))
	# nnmodel.add(MaxPooling1D(pool_size=pool_size))
	nnmodel.add(Conv1D(64,
		kernel_size,
		padding='valid',
		activation='relu',
		strides=strides))
	nnmodel.add(Conv1D(64,
		kernel_size,
		padding='valid',
		activation='relu',
		strides=strides))
	nnmodel.add(Conv1D(64,
		kernel_size,
		padding='valid',
		activation='relu',
		strides=strides))
	# nnmodel.add(MaxPooling1D(pool_size=pool_size))
	nnmodel.add(MaxPooling1D(pool_size=pool_size))
	if model_type=='cnn':
		nnmodel.add(Flatten())
		nnmodel.add(Dense(48))
		nnmodel.add(Dropout(0.5))
		nnmodel.add(Dense(24))
		nnmodel.add(Dropout(0.5))
		nnmodel.add(Dense(12))
		nnmodel.add(Dropout(0.5))
	elif model_type=='cblstm':
		# nnmodel.add(Bidirectional(LSTM(lstm_units)))
		# nnmodel.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=(x.shape[1], x.shape[2])))
		# nnmodel.add(Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001))))
		# nnmodel.add(Bidirectional(LSTM(lstm_units, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001))))
		nnmodel.add(Bidirectional(LSTM(lstm_units, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)), input_shape=(x.shape[1], x.shape[2])))
		# nnmodel.add(Dense(50))
		nnmodel.add(Dropout(0.5))
	else:
		sys.exit('Model type must be "cnn" or "blstm"')
	nnmodel.add(Dense(1))
	nnmodel.add(Activation('sigmoid'))
	nnmodel.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=[metrics.Precision(), metrics.Recall(), 'accuracy'])
	print('Train with ',len(x))
	print(nnmodel.summary())
	#nnmodel.fit(x,y,epochs=epochs,batch_size=batch_size,validation_data=validation_data)
	return nnmodel

def avg(nparray):
	return np.mean(nparray,axis=0)

def pad(sent,maxlen):
	if len(sent) > maxlen:
		return sent[:maxlen]
	else:
		dif=maxlen-len(sent)
		for i in range(dif):
			sent.append('UNK')
	return sent

def vectorize_sentence(sent,model,vocab,model_dim,maxlen):
	out=[]
	for token in pad(sent,maxlen):
		if type(token) == spacy.tokens.token.Token:
			# lower case dataset
			w=token.orth_.lower()
			if w in vocab:
				out.append(model[w])
			else:
				out.append(np.zeros(model_dim, dtype='float32'))
		else:
			# if its an 'UNK'
			out.append(np.zeros(model_dim, dtype='float32'))
	return np.array(out, dtype='float32')

def parse_sent(sent):
	"""
	Dependency parse a sentence and extend with (head, modifier) info.
	"""
	out_words=[]
	out_pos=[]
	out_deps=[]
	for token in sent:
		for c in token.children:
			out_words.append((token.orth_,c.orth_))
			out_pos.append((token.pos_,c.pos_))
			out_deps.append((token.dep_,c.dep_))
	return out_words,out_pos,out_deps

def vectorize_wordpairs(head_modifier_sent,model,vocab,model_dim,maxlen_dep,mode='avg'):
	out=[]
	for item in pad(head_modifier_sent,maxlen_dep):
		flag=False
		if not item=='UNK':
			head,modifier=item[0],item[1]
			if head and modifier:
				if head in vocab and modifier in vocab:
					if mode=='avg':
						deparray=np.array([model[head],model[modifier]], dtype='float32')
						avgdep=avg(deparray)
						out.append(avgdep)
						flag=True
					else:
						sys.exit('This mode: ',mode,' not implemented')
		if not flag:
			out.append(np.zeros(model_dim))
	out=np.array(out, dtype='float32')
	return out

def vectorize_deprels(label_list,maxlen_dep,embedding_dim,labeldict):
	out=[]
	for label in pad(label_list,maxlen_dep):
		onehot=np.zeros(embedding_dim)
		if label and not label=='UNK':
			onehot[labeldict[label]]=1
			out.append(np.array(onehot, dtype='float32'))
		else:
			out.append(onehot, dtype='float32')
	out=np.array(out, dtype='float32')
	#print('Out shape for labels: ',out.shape)
	return out

def load_embeddings(embeddings_path):
	print('Loading embeddings:',embeddings_path)
	try:
		model,vocab,dims = pickle.load(open(embeddings_path, mode='rb'))
		return model, vocab, dims
	except:
		try:
			model=gensim.models.Word2Vec.load(embeddings_path)
		except:
			try:
				model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path,datatype=np.float16)
			except:
				try:
					model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path,binary=True)
				except:
					sys.exit('Couldnt load embeddings')
	vocab=model.index2word
	dims=model.__getitem__(vocab[0]).shape[0]
	vocab=set(vocab)
	return model,vocab,dims


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


class Dataset(object):

	def __init__(self, path, name):

		self.path=path
		self.name=name
		self.instances=[]
		self.sentences=[]
		self.labels=[]

	def load_wcl(self):

		if self.name=='wcl':
			# only wikipedia (manually annotated) defs
			for root, subdirs, files in os.walk(self.path):
				for filename in files:
					if filename.startswith('wiki_'):
						print('f: ',filename)
						label=filename.split('_')[-1].replace('.txt','')
						doc=os.path.join(root,filename)
						lines = open(doc, 'r').readlines()
						for idx,line in enumerate(lines):
							if line.startswith('#'):
								target=lines[idx+1].split(':')[0]
								sent=line[2:].replace('TARGET',target).strip().lower()
								if label=='good':
									self.labels.append(1)
								else:
									self.labels.append(0)
								self.instances.append(sent)
			self.labels=np.array(self.labels)
			print('Loaded ',self.name,' data')
		else:
			sys.exit('Dataset name must be "wcl" ')

	def load_w00(self):
		if self.name =='w00':
			for infile in os.listdir(self.path):
				if infile=='annotated.word':
					sents=open(os.path.join(self.path,infile),'r').readlines()
				elif infile=='annotated.meta':
					labels=open(os.path.join(self.path,infile),'r').readlines()
			if sents and labels:
				for idx,sent in enumerate(sents):
					sent=sent.strip().lower()
					label=int(labels[idx].split(' $ ')[0])
					self.instances.append(sent)
					self.labels.append(label)
			self.labels=np.array(self.labels)
			print('Loaded ',self.name,' data')
		else:
			sys.exit('Dataset name must be "w00" ')


	def load_deft(self, ignore_labels=False):
		if self.name=='deft':
			# only task1 deft
			with open(os.path.join(self.path), 'r', encoding='utf-8') as handle:
				lines = handle.readlines()

			for line in tqdm(lines):
				splits = line.split('\t')
				sentence = splits[0]
				label = '0' if ignore_labels else splits[1]

				raw_sentence = sentence.strip('"').strip('\n')
				self.sentences.append(raw_sentence)
				self.instances.append(nlp(raw_sentence.strip().lower()))
				self.labels.append(int(label.strip('\n').strip('"')))
			self.labels=np.array(self.labels)
			print('Loaded ',self.name,' data')
		else:
			sys.exit('Dataset name must be "deft" ')