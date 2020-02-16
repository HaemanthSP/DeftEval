import gensim
import os
import sys
import spacy
import numpy as np
from keras import metrics, regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional, Dropout, Conv1D, MaxPooling1D, Embedding, Flatten, LSTM
from tqdm import tqdm

nlp=spacy.load('en_core_web_lg')


def build_model(x,y,model_type,lstm_units=100,validation_data=''):
	# hyperparams
	kernel_size = 3
	filters = 50
	pool_size = 4 
	strides=1
	# train opts
	epochs=100
	batch_size=100
	nnmodel = Sequential()
	nnmodel.add(Conv1D(filters,
		kernel_size,
		padding='valid',
		activation='relu',
		strides=strides,
		input_shape=(x.shape[1], x.shape[2])))
	nnmodel.add(MaxPooling1D(pool_size=pool_size))
	nnmodel.add(Conv1D(filters,
		kernel_size,
		padding='valid',
		activation='relu',
		strides=strides))
	nnmodel.add(MaxPooling1D(pool_size=pool_size))
	if model_type=='cnn':
		nnmodel.add(Flatten())
		nnmodel.add(Dropout(0.5))
	elif model_type=='cblstm':
		nnmodel.add(Bidirectional(LSTM(lstm_units)))
		# nnmodel.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=(x.shape[1], x.shape[2])))
		# nnmodel.add(Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001))))
		# nnmodel.add(Bidirectional(LSTM(lstm_units, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001))))
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
				out.append(np.zeros(model_dim))
		else:
			# if its an 'UNK'
			out.append(np.zeros(model_dim))
	return np.array(out)

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
						deparray=np.array([model[head],model[modifier]])
						avgdep=avg(deparray)
						out.append(avgdep)
						flag=True
					else:
						sys.exit('This mode: ',mode,' not implemented')
		if not flag:
			out.append(np.zeros(model_dim))
	out=np.array(out)
	return out

def vectorize_deprels(label_list,maxlen_dep,embedding_dim,labeldict):
	out=[]
	for label in pad(label_list,maxlen_dep):
		onehot=np.zeros(embedding_dim)
		if label and not label=='UNK':
			onehot[labeldict[label]]=1
			out.append(np.array(onehot))
		else:
			out.append(onehot)
	out=np.array(out)
	#print('Out shape for labels: ',out.shape)			
	return out

def load_embeddings(embeddings_path):
	print('Loading embeddings:',embeddings_path)
	try:
		model=gensim.models.Word2Vec.load(embeddings_path)
	except:
		try:
			model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path)
		except:
			try:
				model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path,binary=True)
			except:
				sys.exit('Couldnt load embeddings')
	vocab=model.index2word
	dims=model.__getitem__(vocab[0]).shape[0]
	vocab=set(vocab)
	return model,vocab,dims

class Dataset(object):

	def __init__(self, path, name):

		self.path=path
		self.name=name
		self.instances=[]
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


	def load_deft(self):
		if self.name=='deft':
			# only task1 deft
			with open(os.path.join(self.path), 'r') as handle:
				lines = handle.readlines()

			for line in tqdm(lines[:1000]):
				sentence, label = line.strip().split('\t') 
				self.instances.append(nlp(sentence.strip('"').lower()))
				self.labels.append(int(label.strip('"')))
			self.labels=np.array(self.labels)
			print('Loaded ',self.name,' data')
		else:
			sys.exit('Dataset name must be "deft" ')