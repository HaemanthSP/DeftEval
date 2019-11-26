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


## Experiment:


|  Model Name | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| Baseline | Bilstm | 10 | 0.0884 | 0.9475 | 0.9496 | 0.9385 | 0.5765 | 0.5784 | 0.8599 | 0.6971 | 0.6154 |
