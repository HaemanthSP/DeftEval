# Dataset
---------

## Task1 (Binary classification)

Class wise distribution of data:

| Dataset | Positive  | Negative  | Total | Positive% |
|-------|:---------:|:---------:|:-----:|:---------:|
| train | 5569      | 11090     | 16659 |  33.43    | 
| dev | 273       | 537       | 810   |  33.70    |


Based on the experiments with the baseline model.
- train set is not well representative of the dev set.
- 


Experiment:
-----------

|  Model ref | Model | epochs | Train Loss| Train Accuracy| Valid. loss| Valid. Accuracy| Test loss | Test Accuracy |
|-------|:---------:|:---------:|:---------:|:---------:|:-----:|:---------:|:-----:|:---------:|
| Baseline | Bilstm | 10 | 0.1074 | 95.86% | 0.0910 | 96.4% | 0.733 | 75.3% | 
