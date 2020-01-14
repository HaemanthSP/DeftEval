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
