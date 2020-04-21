# Semantics

* SADE : Syntax aware definition extraction


| Model | W2V | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|
| SADE | Glove vector | 0.72 | 0.62 | 0.67 |
| SADE | Google-w2v | 0.72 | 0.58 | 0.64 |
| Bilstm | Glove vector | 0.74 | 0.63 | 0.68 |
| Bilstm | Google-w2v | 0.71 | 0.6 | 0.65 |
| Bilstm | Self trained |0.74 | 0.60 | 0.66 |


# Feat

| Model | W2V | Feat | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| Bilstm | Glove vector | POS | 0.74 | 0.66 | 0.69 |
| Bilstm | Glove vector | POS + PUNCT | 0.74 | 0.635 | 0.685 |
| SADE | Glove vector | POS | 0.77 | 0.67 | 0.715 |
| SADE | Glove vector | POS + PUNCT | 0.795 | 0.625 | 0.70 |