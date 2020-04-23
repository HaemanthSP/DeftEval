# Semantics

* SADE : Syntax aware definition extraction

--
----

| Model | W2V | Feats | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| SADE | Glove vector | Tokens, Deps | 0.72 | 0.62 | 0.67 |
| SADE | Google-w2v | Tokens, Deps | 0.72 | 0.58 | 0.64 |
| Bilstm | Glove vector | Tokens, Deps | 0.74 | 0.63 | 0.68 |
| Bilstm | Google-w2v | Tokens, Deps | 0.71 | 0.6 | 0.65 |

--
-----

| Model | W2V | Feats | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| SADE | Glove vector | Tokens | 0.785 | 0.59 | 0.675 |
| SADE | Google-w2v | Tokens | 0.775 | 0.515 | 0.615 |
| Bilstm | Glove vector | Tokens | 0.70 | 0.64 | 0.67 |
| Bilstm | Google-w2v | Tokens | 0.70 | 0.61 | 0.65 |
| Bilstm | Self trained | Tokens | 0.74 | 0.60 | 0.66 |


# Feat

| Model | W2V | Feat | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| Bilstm | Glove vector |Tokens, Deps, POS | 0.76 | 0.585 | 0.66 |
| Bilstm | Glove vector |Tokens, Deps, POS + PUNCT | 0.76 | 0.62 | 0.68 |
| SADE | Glove vector |Tokens, Deps, POS | 0.77 | 0.67 | 0.715 |
| SADE | Glove vector |Tokens, Deps, POS + PUNCT | 0.795 | 0.625 | 0.70 |

--
----

| Model | W2V | Feat | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| Bilstm | Glove vector |Tokens, POS | 0.755 | 0.625 | 0.685 |
| Bilstm | Glove vector |Tokens, POS + PUNCT | 0.765 | 0.645 | 0.695 |
| SADE | Glove vector |Tokens, POS | 0.74 | 0.655 | 0.69 |
| SADE | Glove vector |Tokens, POS + PUNCT | 0.77 | 0.645 | 0.705 |


# Our model

| Model | W2V | Feat | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|
| Ours | Glove vector |Tokens, POS + PUNCT | 0.77 | 0.645 | 0.7 |
| Ours | Glove vector |Tokens, Deps, POS + PUNCT | 0.75 | 0.635 | 0.686 |
| Ours | Google-w2v |Tokens, POS + PUNCT | 0.71 | 0.71 | 0.71 |
| Ours | Google-w2v |Tokens, Deps, POS + PUNCT | 0.74 | 0.70 | 0.72 |