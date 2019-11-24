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


|  Model | Model | Epochs | Train Loss | Train Precision| Train Recall| Val. Loss| Val. Precision| Val. Recall| Test Loss | Test Precision | Test Recall |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| Baseline | Bilstm | 10 | 0.0919 | 0.9444 | 0.9491 | 0.0625 | 0.9727 | 0.9582 | 0.855 | 0.654 | 0.623 | 
| Baseline (Resampled) | Bilstm | 10 | 0.0973 | 0.9418 | 0.9463 | 0.0836 | 0.9216 | 0.9855 | 0.966 | 0.591 | 0.651 | 
