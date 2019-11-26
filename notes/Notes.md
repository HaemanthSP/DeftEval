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



## Experiment:


|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| Baseline | Bilstm(64) D[100,50,1] | 10 | 0.0884 | 0.9475 | 0.9496 | 0.9385 | 0.5765 | 0.5784 | 0.8599 | 0.6971 | 0.6154 |
| Baseline_simplified | Bilstm(64) D[50,1] | 10 | 0.1165 | 0.9311 | 0.9368 | 0.9509 | 0.6408 | 0.5275 | 0.7582 | 0.7104 | 0.5751 |
| Baseline_simplified | Bilstm(32) D[20,1] | 10 | 0.1470 | 0.9149 | 0.9248 | 0.6603 | 0.6573 | 0.6413 | 0.6391 | 0.6680 | 0.6044 |
