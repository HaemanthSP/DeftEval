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
