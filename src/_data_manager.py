import gensim
import os
import sys
import spacy
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Bidirectional, Dropout, Conv1D, MaxPooling1D, Embedding, Flatten, LSTM
from tensorflow.keras.initializers import Constant
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
import pickle
from tensorflow.python.keras.utils import metrics_utils
from tensorflow import math as math_ops
from tensorflow.python.keras.utils.generic_utils import to_list
import tensorflow.keras.backend as K


nlp=spacy.load('en_core_web_lg')


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='F1Score', class_id=None, dtype=None):
        super(F1Score, self).__init__(name=name, dtype=dtype)

        self.class_id = class_id
        default_threshold = 0.5
        self.thresholds = metrics_utils.parse_init_thresholds(
            None, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer='zeros')


    def update_state(self, y_true, y_pred, sample_weight=None):
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        precision = math_ops.divide_no_nan(self.true_positives,
                                    self.true_positives + self.false_positives)
        recall = math_ops.divide_no_nan(self.true_positives,
                                    self.true_positives + self.false_negatives)

        precision = precision[0] if len(self.thresholds) == 1 else precision
        recall = recall[0] if len(self.thresholds) == 1 else recall

        mul_val = precision * recall
        add_val = precision + recall

        f1_score = 2 * math_ops.divide_no_nan(mul_val, add_val)

        return 1 - f1_score

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'class_id': self.class_id
        }
        base_config = super(F1Score, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_model3(maxlen, embedding_weights=None, vocab_size=None):
	"""
	combine features from various inputs and build a model
	"""
	input1 = tf.keras.Input(shape=(maxlen[0],), name="Input_word", dtype=tf.int32)
	input2 = tf.keras.Input(shape=(maxlen[0],), name="Input_pos", dtype=tf.int8)
	input3 = tf.keras.Input(shape=(maxlen[1],), name="Input_head", dtype=tf.int32)
	input4 = tf.keras.Input(shape=(maxlen[1],), name="Input_modif", dtype=tf.int32)
	input5 = tf.keras.Input(shape=(maxlen[1],), name="Input_deps", dtype=tf.int8)

	word_embedder = Embedding(vocab_size[0], 300, embeddings_initializer=Constant(embedding_weights[0]), trainable=False, mask_zero=True)
	# embed_1 = Embedding(vocab_size[0], 100, embeddings_regularizer=regularizers.l2(0.001), trainable=True, mask_zero=True)(input1)
	# embed_2 = Embedding(vocab_size[1], 100, embeddings_regularizer=regularizers.l2(0.001), trainable=True, mask_zero=True)(input2)
	pos_embedder = Embedding(vocab_size[1], 32, embeddings_regularizer=regularizers.l2(0.001), trainable=True, mask_zero=True)
	deps_embedder = Embedding(vocab_size[2], 32, embeddings_regularizer=regularizers.l2(0.001), trainable=True, mask_zero=True)

	concate1 = tf.keras.layers.concatenate([word_embedder(input1), pos_embedder(input2)])
	output1 = Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)))(concate1)
	output1 = Conv1D(128, 3, activation='relu')(output1)
	output1 = MaxPooling1D(pool_size=2)(output1)
	output1 = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)))(output1)
	output1 = Conv1D(64, 3, activation='relu')(output1)
	output1 = MaxPooling1D(pool_size=2)(output1)
	output1 = Flatten()(output1)

	concate2 = tf.keras.layers.concatenate([word_embedder(input3), word_embedder(input4), deps_embedder(input5)])
	# concate2 = tf.keras.layers.concatenate([word_embedder(input3), word_embedder(input4)])
	# output2 = Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)))(concate2)
	# output2 = Conv1D(128, 3, activation='relu')(output2)
	output2 = Conv1D(64, 3, activation='relu')(concate2)
	output2 = MaxPooling1D(pool_size=2)(output2)
	# output2 = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)))(output2)
	# output2 = Bidirectional(LSTM(50, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)))(output2)
	output2 = Conv1D(32, 3, activation='relu')(output2)
	output2 = MaxPooling1D(pool_size=2)(output2)
	output2 = Flatten()(output2)


	concate = tf.keras.layers.concatenate([output1, output2])

	# output = Dense(48)(concate)
	# output = Dropout(0.5)(output)
	output = Dense(24)(concate)
	output = Dropout(0.5)(output)
	output = Dense(1, activation='sigmoid')(output)

	inputs_list = [input1, input2, input3, input4, input5]
	model = tf.keras.Model(inputs=inputs_list, outputs=output)
	model.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=[metrics.Precision(), metrics.Recall(), 'accuracy', F1Score()])
		# metrics=[metrics.Precision(), metrics.Recall(), 'accuracy'])
	print(model.summary())

	return model


def build_model2(maxlen, embedding_weights=None, vocab_size=None):
	"""
	combine features from various inputs and build a model
	"""
	input1 = tf.keras.Input(shape=(maxlen,), name="Input_1", dtype=tf.int32)
	input2 = tf.keras.Input(shape=(maxlen,), name="Input_2", dtype=tf.int8)
	embed_1 = Embedding(vocab_size[0], 300, embeddings_initializer=Constant(embedding_weights[0]), trainable=False, mask_zero=True)(input1)
	# embed_1 = Embedding(vocab_size[0], 100, embeddings_regularizer=regularizers.l2(0.001), trainable=True, mask_zero=True)(input1)
	# embed_2 = Embedding(vocab_size[1], 100, embeddings_regularizer=regularizers.l2(0.001), trainable=True, mask_zero=True)(input2)
	embed_2 = Embedding(vocab_size[1], 100, trainable=True, mask_zero=True)(input2)
	concate = tf.keras.layers.concatenate([embed_1, embed_2])
	output = Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)))(concate)
	# output = Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)))(output)
	# output = Conv1D(256, 3, activation='relu')(concate)
	output = Conv1D(128, 3, activation='relu')(output)
	# output = Conv1D(256, 3, activation='relu')(output)
	output = MaxPooling1D(pool_size=2)(output)
	output = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)))(output)
	# output = Conv1D(64, 3, activation='relu')(output)
	output = Conv1D(64, 3, activation='relu')(output)
	# output = Conv1D(64, 3, activation='relu')(output)
	output = MaxPooling1D(pool_size=2)(output)
	output = Flatten()(output)
	# output = Dense(64)(output)
	# output = Dropout(0.5)(output)
	# output = Dense(48)(output)
	# output = Dropout(0.5)(output)
	output = Dense(24)(output)
	output = Dropout(0.5)(output)
	output = Dense(1, activation='sigmoid')(output)

	inputs_list = [input1, input2]
	model = tf.keras.Model(inputs=inputs_list, outputs=output)
	model.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=[metrics.Precision(), metrics.Recall(), 'accuracy', F1Score()])
		# metrics=[metrics.Precision(), metrics.Recall(), 'accuracy'])
	print(model.summary())

	return model


def build_model(x,y,model_type,lstm_units=100,validation_data='', embedding_weights=None, vocab_size=None):
	# hyperparams
	kernel_size = 2
	filters = 50
	pool_size = 2
	strides=1
	# train opts
	epochs=100
	batch_size=100
	nnmodel = Sequential()
	if embedding_weights is not None:
		nnmodel.add(Embedding(vocab_size, 300, embeddings_initializer=Constant(embedding_weights), input_length=x.shape[1], trainable=True, mask_zero=True))
		nnmodel.add(Conv1D(256,
			kernel_size,
			padding='valid',
			activation='relu',
			strides=strides))
	else:
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
		nnmodel.add(Bidirectional(LSTM(lstm_units, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001))))
		# nnmodel.add(Bidirectional(LSTM(lstm_units, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001)), input_shape=(x.shape[1], x.shape[2])))
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

def build_baseline1(x,y,model_type,lstm_units=100,validation_data='', embedding_weights=None, vocab_size=None):
	# hyperparams
	kernel_size = 3
	filters = 100
	pool_size = 4
	strides=1
	# train opts
	epochs=10
	batch_size=100
	nnmodel = Sequential()

	nnmodel.add(Conv1D(filters,
		kernel_size,
		padding='valid',
		activation='relu',
		strides=strides,
		input_shape=(x.shape[1], x.shape[2])))
	nnmodel.add(MaxPooling1D(pool_size=pool_size))

	if model_type=='cnn':
		nnmodel.add(Flatten())
		nnmodel.add(Dropout(0.5))
	elif model_type=='cblstm':
		nnmodel.add(Bidirectional(LSTM(lstm_units)))
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

def build_baseline0(x,y,lstm_units=100,validation_data='', embedding_weights=None, vocab_size=None):

	# hyperparams
	nnmodel = Sequential()
	nnmodel.add(Bidirectional(LSTM(lstm_units), input_shape=(x.shape[1], x.shape[2])))
	nnmodel.add(Dropout(0.5))
	nnmodel.add(Dense(50, activation='relu'))
	nnmodel.add(Dropout(0.5))
	nnmodel.add(Dense(1))
	nnmodel.add(Activation('sigmoid'))
	nnmodel.compile(loss='binary_crossentropy',
					optimizer='adam',
					metrics=[metrics.Precision(), metrics.Recall(), 'accuracy'])
	print('Train with ',len(x))
	print(nnmodel.summary())
	return nnmodel

def build_baseline_bilstm(maxlen, vocab_size=None):

	# hyperparams
	nnmodel = Sequential()
	nnmodel.add(Embedding(vocab_size, 300, input_length=maxlen, trainable=True, mask_zero=True))
	nnmodel.add(Bidirectional(LSTM(200)))
	nnmodel.add(Dropout(0.5))
	nnmodel.add(Dense(50, activation='relu'))
	nnmodel.add(Dropout(0.5))
	nnmodel.add(Dense(1))
	nnmodel.add(Activation('sigmoid'))
	nnmodel.compile(loss='binary_crossentropy',
					optimizer='adam',
					metrics=[metrics.Precision(), metrics.Recall(), 'accuracy', F1Score()])
					# metrics=[metrics.Precision(), metrics.Recall(), 'accuracy'])
	print(nnmodel.summary())
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