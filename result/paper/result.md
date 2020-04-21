# Semantics

* SADE : Syntax aware definition extraction


| Model | W2V | Feats | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| SADE | Glove vector | Tokens, Deps | 0.72 | 0.62 | 0.67 |
| SADE | Google-w2v | Tokens, Deps | 0.72 | 0.58 | 0.64 |
| Bilstm | Glove vector | Tokens, Deps | 0.74 | 0.63 | 0.68 |
| Bilstm | Google-w2v | Tokens, Deps | 0.71 | 0.6 | 0.65 |
| Bilstm | Self trained | Tokens | 0.74 | 0.60 | 0.66 |

Upcomming
---------

| Model | W2V | Feats | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| SADE | Glove vector | Tokens | | | |
| SADE | Google-w2v | Tokens | | | |
| Bilstm | Glove vector | Tokens | 0.70 | 0.64 | 0.67 |
| Bilstm | Google-w2v | Tokens | 0.70 | 0.61 | 0.65 |



# Feat

| Model | W2V | Feat | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| Bilstm | Glove vector |Tokens, Deps, POS | 0.76 | 0.585 | 0.66 |
| Bilstm | Glove vector |Tokens, Deps, POS + PUNCT | 0.76 | 0.62 | 0.68 |
| SADE | Glove vector |Tokens, Deps, POS | 0.77 | 0.67 | 0.715 |
| SADE | Glove vector |Tokens, Deps, POS + PUNCT | 0.795 | 0.625 | 0.70 |


| Model | W2V | Feat | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| Bilstm | Glove vector |Tokens, POS | | | |
| Bilstm | Glove vector |Tokens, POS + PUNCT | | | |
| SADE | Glove vector |Tokens, POS | | | |
| SADE | Glove vector |Tokens, POS + PUNCT | | | |