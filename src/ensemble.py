# stacked generalization with neural net meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
import pickle
from tqdm import tqdm
import numpy as np
from tensorflow.keras import metrics, regularizers

import task1_custom as t1
import _data_manager
from _data_manager import F1Score

# load models from file
def load_all_models(model_paths):
    all_models = list()
    for filename in model_paths:
           # load model from file
           model = load_model(filename, compile=False)
        #    model.compile(loss='binary_crossentropy', optimizer='adam',
            #    metrics=[metrics.Precision(), metrics.Recall(), 'accuracy', F1Score()])
           # add to list of members
           all_models.append(model)
           print('>loaded %s' % filename)
    return all_models

# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
    	model = members[i]
    	for layer in model.layers:
    		# make not trainable
    		layer.trainable = False
    		# rename to avoid 'unique layer name' issue
    		layer.name = 'ensemble_' + str(i+1) + '_' + layer.name + '_' + str(np.random.randint(100))
    # define multi-headed input
    ensemble_visible = [input_tensor for model in members for input_tensor in model.inputs]
    print("Input", ensemble_visible)
    # concatenate merge output from each model
    ensemble_outputs = [model.outputs[0] for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(1, activation='softmax')(hidden)
    print("Output:", output)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='binary_crossentropy',
    	optimizer='adam',
    	metrics=[metrics.Precision(), metrics.Recall(), 'accuracy', F1Score()])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
	# prepare input data
	# X = [input_arr for _ in range(len(model.input)) for input_arr in inputX]
	# encode output data
	# inputy_enc = to_categorical(inputy)
	# fit model
	model.fit(inputX, inputy, epochs=300, verbose=0)

# make a prediction with a stacked model
def predict_stacked_model(model, X_eval, y_eval):
	# prepare input data
	X = [X_eval for _ in range(len(model.input))]
	# make prediction

	predictions = model.predict_on_batch(X)
	preds=np.array([1 if i[0]>0.5 else 0 for i in predictions], dtype='float32')
	print("Confusion Matrix")
	print(confusion_matrix(preds, y_eval))
	from sklearn.metrics import classification_report
	print(classification_report(y_eval, preds))


def prepare_data(datapath, model_path):
    eval_dataset =_data_manager.Dataset(datapath,'deft')
    eval_dataset.load_deft()

    # Load model and metadata
    metadata = pickle.load(open(model_path + '/meta', 'rb'))
    _, _, id2vocab, _, id2pos = metadata
    X_eval_word = t1.encode_X_words(eval_dataset, metadata)
    X_eval_pos = t1.encode_X_pos(eval_dataset, metadata)
    y_eval = np.array(eval_dataset.labels, dtype='float32')
    return X_eval_word, X_eval_pos, y_eval


if __name__ == '__main__':
    # model_paths = ["../resources/deft_hybrid_glove_f75_a833", "../resources/deft_hybrid_m2_f75_a84"]
    model_paths = ["../resources/deft_hybrid_m5", "../resources/deft_hybrid_m3"]
    # model_paths = ["../resources/deft_hybrid_m5"]

    X_train_word, X_train_pos, y_train = prepare_data("../task1_data/train_combined.deft", model_paths[0])
    X_test_word, X_test_pos, y_test = prepare_data("../task1_data/dev_combined.deft", model_paths[0])

    # load all models
    members = load_all_models(model_paths)
    print('Loaded %d models' % len(members))

    # define ensemble model
    stacked_model = define_stacked_model(members)
    # fit stacked model on test dataset
    fit_stacked_model(stacked_model, [X_train_word, X_train_pos] * len(members), y_train)
    # make predictions and evaluate
    predict_stacked_model(stacked_model, [X_test_word, X_test_pos])