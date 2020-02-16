# Task1 (Binary classification)

## Dataset:
Class wise distribution of data:

| Dataset | Positive  | Negative  | Total | Positive% |
|-------|:---------:|:---------:|:-----:|:---------:|
| train | 5569      | 11090     | 16659 |  33.43    |
| dev | 273       | 537       | 810   |  33.70    |


Based on the experiments with the baseline model:
* The train set is not very representative of the dev set.

We also resampled the train and dev datasets by combining and shuffling them together while retaining the above cardinality.


## Benchmarking

|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| Baseline | Bilstm(64) D[100,50,1] | 10 | 0.0884 | 0.9475 | 0.9496 | 0.9385 | 0.5765 | 0.5784 | 0.8599 | 0.6971 | 0.6154 |
| Baseline | Bilstm(64) D[100,50,1] | 03 | 0.4139 | 0.7518 | 0.6407 | 0.4771 | 0.6997 | 0.6235 | 0.4500 | 0.6960 | 0.6560 |



## Experiment:

Attempts to find the hyperparameter for the baseline model that avoids the high variance effect.

|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| Baseline | Bilstm(64) D[100,50,1] | 10 | 0.0884 | 0.9475 | 0.9496 | 0.9385 | 0.5765 | 0.5784 | 0.8599 | 0.6971 | 0.6154 |
| Baseline_simplified_1 | Bilstm(64) D[50,1] | 10 | 0.1165 | 0.9311 | 0.9368 | 0.9509 | 0.6408 | 0.5275 | 0.7582 | 0.7104 | 0.5751 |
| Baseline_simplified_2 | Bilstm(32) D[20,1] | 10 | 0.1470 | 0.9149 | 0.9248 | 0.6603 | 0.6573 | 0.6413 | 0.6391 | 0.6680 | 0.6044 |
| Baseline_simplified_3 | Bilstm(32) D[20(0.5),1] | 10 | 0.1963 | 0.9012 | 0.9158 | 0.7034 | 0.6444 | 0.5512 | 0.620 | 0.717 | 0.586 |

* After analyzing the plots for above experiments. After Epoch 3 the model starts overfitting.

|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| Baseline | Bilstm(64) D[100,50,1] | 3 | 0.4139 | 0.7518 | 0.6407 | 0.4771 | 0.6997 | 0.6235 | 0.450 | 0.696 | 0.656 |
| Baseline_simplified_1 | Bilstm(64) D[50,1] | 3 | 0.4432 | 0.7295 | 0.5925 | 0.4913 | 0.6825 | 0.5436 | 0.452 | 0.714 | 0.557 |
| Baseline_simplified_2 | Bilstm(32) D[20,1] | 3 | 0.4840 | 0.7281 | 0.4834 | 0.4819 | 0.6456 | 0.5 | 0.4582 | 0.7360 | 0.5311 |
| Baseline_simplified_3 | Bilstm(32) D[20(0.5),1] | 3 | 0.5485 | 0.6908 | 0.3309 | 0.5016 | 0.6923 | 0.4650 | 0.496 | 0.726 | 0.447 |

* Among all the above experiments the baseline with 3 epoch seems better (adding to bechmarking)


* Now the model has high bias. Lets increase feature to reduce it.
* In that line, by looking at the decoded mis-prediction which doesnt seem to contain any punctuation(Which could be critical for certain type of definitions). Tokenizer need attention

Vocab size with filtering: 25427
Vocab size without filtering: 26121 (to Minimize: Might need to lower the text before encoding)

|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| Multi feat | E(w32, pos16) [2x[2xBilstm(16) D[16]]Bilstm(16) D[16,1] | 10 | 0.1465 | 0.9231 | 0.9366 | 0.7822 | 0.5827 | 0.6727 | 0.718 | 0.623 | 0.689 |
| Multi feat | [E(w32, pos16) 2x[2xBilstm(8) D[8]]Bilstm(8) D[8,1] | 10 | 0.1888 | 0.8881 | 0.9409 | 0.7271 | 0.6739 | 0.6045 | 0.627 | 0.682 | 0.652 |
| Multi feat | E(w32, pos16) [2x[Bilstm(8) D[8]]Bilstm(8) D[8,1] | 10 | 0.2116 | 0.8781 | 0.9303 | 0.6369 | 0.6646 | 0.6105 | 0.561 | 0.690 | 0.685 |
| Multi feat | E(w16, pos16) [2x[Bilstm(8) D[8]]Bilstm(8) D[8,1] | 10 | 0.2552 | 0.8735 | 0.8753 | 0.6263 | 0.5963 | 0.6132 | 0.500 | 0.699 | 0.707 |
| Multi feat | E(w10, pos10) [2x[Bilstm(8)] Bilstm(8) D[8,1] | 10 | 0.3061 | 0.8256 | 0.8578 | 0.5696 | 0.6825 | 0.6301 | 0.493 | 0.707 | 0.670 |
| Multi feat | E(w10) [[Bilstm(8)] Bilstm(8) D[8,1] | 10 | 0.2584 | 0.8526 | 0.8691 | 0.5572 | 0.6097 | 0.6667 | 0.523 | 0.654 | 0.685 |
| Multi feat | E(pos10) [[Bilstm(8)] Bilstm(8) D[8,1] | 10 | 0.5887 | 0.0000 | 0.0000 | 0.5954 | 0.0000 | 0.0000 | 0.580 | 0.000 | 0.000 |
| Multi feat | E(pos128) [[Bilstm(64)] Bilstm(32) D[32,1] | 30 | 0.5365 | 0.6647 | 0.3759 | 0.5479 | 0.6765 | 0.4023 | 0.529 | 0.684 | 0.396 |
| Multi feat | E(pos128) [[Bilstm(64)] Bilstm(32) D[32,1] | 54 | 0.5241 | 0.6768 | 0.4168 | 0.5161 | 0.5966 | 0.4369 | ----- | ----- | ----- |
| Multi feat | E(pos128) [[Bilstm(64)] Bilstm(32) D[32,1] | 100 | 0.4892 | 0.6897 | 0.4984 | 0.5190 | 0.5856 | 0.4738 | 0.508 | 0.708 | 0.443 |
| Multi feat | E(pos300) [[Bilstm(64)] Bilstm(32) D[32,1] | 30 | 0.5250 | 0.6759 | 0.3974 | 0.5415 | 0.6352 | 0.4519 | 0.517 | 0.701 | 0.429 |
| Multi feat | E(pos300) [[2xBilstm(64) D[64]]Bilstm(32) D[32, 16,1] | 30 | 0.5237 | 0.6780 | 0.4020 | 0.5602 | 0.7815 | 0.2555 | 0.532 | 0.750 | 0.275 |
| Multi feat | E(pos300) [[2xBilstm(64) D[64]]Bilstm(32) D[32, 16,1] | 1000 | 0.0333 | 0.9611 | 0.9705 | 2.2021 | 0.6265 | 0.6174 | 2.006 | 0.626 | 0.608 |
| Multi feat | E(pos64, dep64) Bilstm(32) D[32, 16,1] | 30 | 0.4926 | 0.6909 | 0.5100 | 0.5158 | 0.6809 | 0.5000 | - | - | - |
| Multi feat | E(pos128, dep128) Bilstm(32) D[32, 1] | 16 (ES-5) | 0.5432 | 0.5739 | 0.7359 | 0.5397 | 0.5947 | 0.7762 | 0.527 | 0.587 | 0.744 |
| Multi feat | E(pos128, dep128) Bilstm(32) D[32, 1] | 20 (ES-5) | 0.4829 | 0.7196 | 0.5421 | 0.5147 | 0.6475 | 0.5736 | 0.483 | 0.778 | 0.538 |
| Multi feat | E(pos128, dep128) [Bilstm(16)] D[16, 1] | 25 (ES-10) | 0.4245 | 0.7587 | 0.6202 | 0.4885 | 0.6851 | 0.6152 | 0.478 | 0.702 | 0.579 |
| Multi feat | E(pos32, w2, dep32) [Bilstm(16)] D[16, 1] | 7 (ES-5) | 0.1609 | 0.9160 | 0.9082 | 0.8031 | 0.6367 | 0.5657 | 0.450 | 0.744 | 0.586 |
| Multi feat | E(w2) [Bilstm(32)] D[16, 1] | 7 (ES-5) | 0.1196 | 0.9388 | 0.9454 | 0.7383 | 0.5983 | 0.6762 | 0.475 | 0.681 | 0.634 |
| Multi feat | E(w1) [Bilstm(32)] D[16, 1] | 7 (ES-5) | 0.2004 | 0.8919 | 0.8919 | 0.6672 | 0.6416 | 0.6104 | 0.469 | 0.697 | 0.597 |

Even the embedding only word with dim 1 leads to overfitting. This seems to be the effect of suspiciously high vocabulary size 22k from 16k sentences

Reasons for vocabulary explosion:
1. Lots of urls
2. Numbers like values and years
3. Strange alpha numeric combination due to noise in text
4. Improperly parsed word due to irregular use of Punctuations
5. Lack of space around "+ - \} \{ \) \( \[ \] = /" etc.. causes bad tokens

Possible solution:
1. urls - replace with a marker (addresses 1)
2. Replace token less than freq less than n (50) with its pos tag.. n has to be decided (addresses issues 2, 3, 4)
3. Add space around "+ - \} \{ \) \( \[ \] = /" with space around (warning: after replacing urls)

After all the above changes.. the vocabulary is drastically reduced, until it prevents overfitting with 1-dimensional embedding.

Some definition of definitions are not so clear
* " Organisms are individual living entities ." 1 : This is a definition
* " Organelles are small structures that exist within cells ." 0 : this is not a definition
* " 149 . Nucleic acids are the most important macromolecules for the continuity of life ."	"0"
* " They carry the genetic blueprint of a cell and carry instructions for the functioning of the cell ."	"1"
* " 4888 . Recall from The Macroeconomic Perspective that if exports exceed imports , the economy is said to have a trade surplus ."	"0"
* " If imports exceed exports , the economy is said to have a trade deficit ."	"1"
* " 4918 . To build a useful macroeconomic model , we need a model that shows what determines total supply or total demand for the economy , and how total demand and total supply interact at the macroeconomic level ."	"1"
* " 221 . However , the cell membrane detaches from the wall and constricts the cytoplasm ."	"1"

|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| Multi feat CNN| [E(wxpos128(freq>70), dep128, head128) 2xCNN(64) Bilstm(32) D[16,1] | 13ES(5) | 0.2414 | 0.8621 | 0.8742 | 0.5384 | 0.7601 | 0.6740 | 0.527 | 0.650 | 0.612 |
| Multi feat CNN| [E(wxpos128(freq>90), dep128, head128) 2xCNN(64) Bilstm(32) D[16,1] | 9ES(5) | 0.3389 | 0.7971 | 0.7922 | 0.5176 | 0.6577 | 0.6242 | 0.523 | 0.736 | 0.491 |
| Multi feat CNN| [E(wxpos128(freq>70), dep128, head128) 2xCNN(64) Bilstm(32) D[16,1] | 9ES(5) | 0.3014 | 0.8345 | 0.8233 | 0.4869 | 0.8074 | 0.5881 | 0.530 | 0.619 | 0.648 |
| Multi feat CNN| [E(wxpos128(freq>10), dep128, head128) 2xCNN(64) Bilstm(32) D[16,1] | 7ES(5) | 0.1188 | 0.9405 | 0.9462 | 0.7438 | 0.6722 | 0.6128 | 0.488 | 0.735 | 0.601 |
| Multi feat CNN| [E(wxpos128(freq>10), dep128, head128) 3xCNN(64) Bilstm(32) D[16,1] | 7ES(5) | 0.1217 | 0.9386 | 0.9354 | 0.7720 | 0.6769 | 0.6864 | 0.495 | 0.733 | 0.512 |
| Multi feat CNN| [E(wxpos128(freq>70), dep128, head128) 3xCNN(64) Bilstm(32) D[16,1] | 7ES(5) | 0.1217 | 0.9386 | 0.9354 | 0.7720 | 0.6769 | 0.6864 | 0.495 | 0.733 | 0.512 |

* Epoch 2/100
* 245/245 [==============================] - 52s 211ms/step - loss: 0.4848 - precision: 0.7055 - recall: 0.6138 - val_loss: 0.4901 - val_precision: 0.8162 - val_recall: 0.4507
* Epoch 3/100
* 245/245 [==============================] - 49s 199ms/step - loss: 0.4524 - precision: 0.7221 - recall: 0.6449 - val_loss: 0.4388 - val_precision: 0.7765 - val_recall: 0.6119
* Epoch 4/100
* 245/245 [==============================] - 48s 196ms/step - loss: 0.4169 - precision: 0.7498 - recall: 0.6846 - val_loss: 0.4304 - val_precision: 0.7031 - val_recall: 0.7493
* Epoch 5/100
* 245/245 [==============================] - 49s 201ms/step - loss: 0.3919 - precision: 0.7608 - recall: 0.7177 - val_loss: 0.4585 - val_precision: 0.6362 - val_recall: 0.8090
* Epoch 6/100
* 245/245 [==============================] - 49s 198ms/step - loss: 0.3801 - precision: 0.7699 - recall: 0.7421 - val_loss: 0.4300 - val_precision: 0.7331 - val_recall: 0.7134
* Epoch 7/100
* 245/245 [==============================] - 50s 203ms/step - loss: 0.3453 - precision: 0.8013 - recall: 0.7809 - val_loss: 0.4415 - val_precision: 0.7940 - val_recall: 0.6328
* Epoch 8/100
* 245/245 [==============================] - 49s 201ms/step - loss: 0.3172 - precision: 0.8179 - recall: 0.8048 - val_loss: 0.4453 - val_precision: 0.7759 - val_recall: 0.6716
* Epoch 9/100
* 245/245 [==============================] - 48s 196ms/step - loss: 0.3014 - precision: 0.8345 - recall: 0.8233 - val_loss: 0.4869 - val_precision: 0.8074 - val_recall: 0.5881
* 
* Eval loss: 0.530, Eval precision: 0.619, Eval recall: 0.648

Adding regularization to embedding and recurrent kernel seems to have positive impact on training

Suspecting: issue of not masking while padding could be an issue: specifically with a recurrent unit

Attempting to use a fully convolutional network


|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| Multi feat full CNN| [E(wxpos128(freq>2), dep128, head128) 3xCNN(128,64,64)(5,4,3) D[100(0.5), 50(0.5),1] | 7ES(5) | 0.3773 | 0.8640 | 0.8524 | 0.6638 | 0.6543 | 0.7278 | 0.516 | 0.850 | 0.501 |
| Multi feat | [E(wxpos128(freq>2), dep128, head128) [[Bilstm(64) ] Bilstm(32) D[24(0.5), 8(0.5), 1] | 9(ES) | 0.4822 | 0.8285 | 0.6911 | 0.5578 | 0.7079 | 0.6458 | 0.517 | 0.777 | 0.589 |
| Multi feat | [E(wxpos128(freq>2), dep128, head128) [[Bilstm(64) ] Bilstm(32) D[24(0.5), 8(0.5), 1] | 11(ES) | 0.4822 | 0.8285 | 0.6911 | 0.5578 | 0.7079 | 0.6458 | 0.517 | 0.777 | 0.589 |
Epoch 11/100
60/60 [==============================] - 49s 821ms/step - loss: 0.4392 - precision: 0.8957 - recall: 0.7740 - val_loss: 0.7434 - val_precision: 0.6178 - val_recall: 0.7429
Eval loss: 0.602, Eval precision: 0.822, Eval recall: 0.638


# After simplification

|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| ReIncBase| W128, BiLstm(64)D[100, 50, 1] | 8ES(5) | 0.1092 | 0.9337 | 0.9304 | 1.0690 | 0.7190 | 0.5006 | 0.468 | 0.651 | 0.686 |
| ReIncBase| W128, BiLstm(64)D[100, 50, 1] | 8ES(5) | 0.1092 | 0.9337 | 0.9304 | 1.0690 | 0.7190 | 0.5006 | 0.446 | 0.719 | 0.635 |


- [x] Punctuation based definitions are missed: Should preserve punctuations
- [ ] Definition like structure triggers it as definition: Could be rectified by use of POS
- Sub-patterns in the long sentences are mis-leading.


|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| ReIncBase| W128, BiLstm(64)D[100, 50, 1] | 8ES(5) | 0.1228 | 0.9261 | 0.9267 | 1.0549 | 0.7161 | 0.4961 | 0.479 | 0.736 | 0.467 |
Epoch 8/30
241/241 [==============================] - 23s 95ms/step - loss: 0.1228 - precision: 0.9261 - recall: 0.9267 - val_loss: 1.0549 - val_precision: 0.7161 - val_recall: 0.4961
14/14 [==============================] - 0s 34ms/step - loss: 0.4790 - precision: 0.7356 - recall: 0.4672

Eval Loss: 0.479, Eval Precision: 0.736, Eval Recall: 0.467

',', '.', '/', '(', ')', '-', '_', ';', ':', '?', '!', '[', ']'
Including all the punctuations drastically affects the result.

I suppose punctuation without support of pos tags confused the model

Adding POS harms the results.. Concatenating the sequence of tags after the sentence
Epoch 7/30
241/241 [==============================] - 77s 318ms/step - loss: 0.0910 - precision: 0.9508 - recall: 0.9539 - val_loss: 1.0400 - val_precision: 0.6560 - val_recall: 0.6343
14/14 [==============================] - 1s 99ms/step - loss: 0.4725 - precision: 0.7151 - recall: 0.4489

Eval Loss: 0.473, Eval Precision: 0.715, Eval Recall: 0.449 

But adding word and pos as pair boosts
Epoch 8/30
241/241 [==============================] - 95s 393ms/step - loss: 0.0846 - precision: 0.9555 - recall: 0.9581 - val_loss: 1.0319 - val_precision: 0.6553 - val_recall: 0.6575
14/14 [==============================] - 2s 119ms/step - loss: 0.4606 - precision: 0.7026 - recall: 0.5949

Eval Loss: 0.461, Eval Precision: 0.703, Eval Recall: 0.595

- [ ] Need to add class weights
- [ ] Also glove


# Experiments with Syntax aware definition extraction


|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall | Test F1 Score | Test Accuracy |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| SW BiLSTM - dep | BiLstm(100(0.5))D(1) | 10 | 0.3136 | 0.8307 | 0.7324 | 0.5058 | 0.6870 | 0.6532 | - | 0.79 | 0.79 | 0.79 | 0.79 
| SW C-BiLSTM - dep | C(50)P(4)BiLstm(100(0.5))D(1) | 9ES(5) | 0.1291 | 0.9340 | 0.9247 | 0.6799 | 0.6465 | 0.6761 | 0.397 | 0.739 | 0.650 | 0.69 | 0.813 
| SW C-BiLSTM - dep | C(200)P(4)BiLstm(100(0.5))D(1) | 7ES(5) | 0.1068 | 0.9437 | 0.9458 | 0.6230 | 0.6807 | 0.6831 | 0.437 | 0.859 | 0.423 | 0.57 | 0.79 
| SW C-BiLSTM - dep | C(100)P(4)BiLstm(100(0.5))D(1) | 8ES(5) | 0.1040 | 0.9484 | 0.9444 | 0.7189 | 0.6650 | 0.7025 | 0.396 | 0.778 | 0.588 | 0.67 | 0.813
| SW C-BiLSTM - wp | C(100)P(4)BiLstm(100(0.5))D(1) | 8ES(5) | 0.1156 | 0.9399 | 0.9414 | 0.6938 | 0.7000 | 0.5915 | 0.424 | 0.646 | 0.752 | 0.69 | 0.787
| SW C-BiLSTM - wp | C(50)P(4)BiLstm(100(0.5))D(1) | 9ES(5) | 0.1268 | 0.9317 | 0.9275 | 0.7069 | 0.7077 | 0.5669 | 0.406 | 0.745 | 0.650 | 0.69 | 0.815
| SW C-BiLSTM | C(50)P(4)BiLstm(100(0.5))D(1) | 8ES(5) | 0.1447 | 0.9271 | 0.9075 | 0.6170 | 0.6538 | 0.6849 | 0.410 | 0.728 | 0.635 | 0.68 | 0.806
| SW C-BiLSTM - dep | C(50)P(4)BiLstm(100)D(50(0.5), 1) | 9ES(5) | 0.1212 | 0.9399 | 0.9348 | 0.7698 | 0.7152 | 0.5792 | 0.416 | 0.787 | 0.566 | 0.66 | 0.811
| SW C-BiLSTM - dep | BiLstm(100)(0.5)D(1) | 9ES(5) | 0.3288 | 0.8211 | 0.7188 | 0.4875 | 0.6723 | 0.6250 | 0.428 | 0.713 | 0.599 | 0.65 | 0.793
| SW C-BiLSTM - dep | BiLstm(100)(100))(0.5)D(1) | 10ES(5) | 0.2648 | 0.8563 | 0.7923 | 0.5233 | 0.6499 | 0.6144 | 0.407 | 0.764 | 0.591 | 0.67 | 0.809


# Adding punctuation and POS to this model
by adding additional one hot dimensions for each punctuations


|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall | Test F1 Score | Test Accuracy |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| SW C-BiLSTM - dep & wp + pos | C(50)P(4)BiLstm(100(0.5))D(1) | 8ES(5) | 0.1533 | 0.9141 | 0.9117 | 0.6540 | 0.7500 | 0.5335 | 0.405 | 0.784 | 0.635 | 0.70 | 0.826 
| SW C-BiLSTM - dep + pos | C(50)P(4)BiLstm(100(0.5))D(1) | 8ES(5) | 0.1343 | 0.9280 | 0.9225 | 0.6933 | 0.7210 | 0.5915 | 0.399 | 0.726 | 0.704 | 0.71 | 0.819 
| SW C-BiLSTM - wp + pos | C(50)P(4)BiLstm(100(0.5))D(1) | 9ES(5) | 0.1057 | 0.9496 | 0.9468 | 0.8048 | 0.7212 | 0.5739 | 0.414 | 0.713 | 0.635 | 0.67 | 0.800 
| SW C-BiLSTM - dep + pos | C(50)P(4)BiLstm(100)(100)(0.5))D(1) | 8ES(5) | 0.1303 | 0.9347 | 0.9295 | 0.6935 | 0.6280 | 0.7342 | 0.411 | 0.666 | 0.785 | 0.72 | 0.804 
| SW C-BiLSTM - dep + pos | C(50)P(4)(50)P(4)BiLstm(100)(0.5))D(1) | 7ES(5) | 0.1252 | 0.9339 | 0.9293 | 0.6503 | 0.7128 | 0.6074 | 0.412 | 0.712 | 0.668 | 0.69 | 0.806 
| SW C-BiLSTM - dep + pos | C(50)P(4)(50)P(4)BiLstm(100)(100)(0.5))D(1) | 7ES(5) | 0.1318 | 0.9274 | 0.9374 | 0.7142 | 0.7426 | 0.5282 | 0.394 | 0.734 | 0.646 | 0.69 | 0.811 
| SW C-BiLSTM - dep + pos | C(50)P(2)(50)P(2)BiLstm(100)(0.5))D(1) | 9ES(5) | 0.1024 | 0.9456 | 0.9486 | 0.6950 | 0.6673 | 0.6496 | 0.396 | 0.754 | 0.650 | 0.70 | 0.819 
| SW C-BiLSTM - dep + pos | C(50)P(2)(50)P(2)BiLstm(100)(100)(0.5))D(1) | 7ES(5) | 0.1528 | 0.9300 | 0.9159 | 0.6495 | 0.7171 | 0.6338 | 0.414 | 0.745 | 0.672 | 0.71 | 0.820 
| SW C-BiLSTM - dep + pos | C(50)P(4)(50)P(4)BiLstm(100)(0.5))D(1) | 9ES(5) | 0.0805 | 0.9663 | 0.9655 | 0.7568 | 0.7058 | 0.6250 | 0.395 | 0.730 | 0.748 | 0.74 | 0.829 
| SW C-BiLSTM - dep & wp + pos | C(50)P(4)(50)P(4)BiLstm(100)(0.5))D(1) | 8ES(5) | 0.0955 | 0.9533 | 0.9564 | 0.6804 | 0.7049 | 0.6602 | 0.378 | 0.761 | 0.697 | 0.73 | 0.832 
| SW C-BiLSTM - dep & wp + poswpunc | C(50)P(4)(50)P(4)BiLstm(100)(0.5))D(1) | 8ES(5) | 0.1042 | 0.9453 | 0.9512 | 0.6974 | 0.6584 | 0.7025 | 0.389 | 0.795 | 0.609 | 0.69 | 0.824 
| SW C-BiLSTM - dep & wp + poswpunc | C(50)P(4)(50)P(4)BiLstm(100)(0.5))D(1) | 8ES(5) | 0.0985 | 0.9511 | 0.9538 | 0.6659 | 0.6867 | 0.6637 | 0.382 | 0.757 | 0.737 | 0.74 | 0.839 