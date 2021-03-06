# Semantics

* SADE : Syntax aware definition extraction

--
----
Without any token embeddings:

| Model |  Feats | Precision | Recall | F1-Score |
|-------|:---------:|:---------:|:---------:|:---------:|
| Bilstm |POS, Deps | 0.56 | 0.75 | 0.64 |

--
-----

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
| Ours - I | Glove vector |Tokens, POS + PUNCT | 0.77 | 0.645 | 0.7 |
| Ours - II | Glove vector |Tokens, Deps, POS + PUNCT | 0.75 | 0.705 | 0.73 |
| Ours - I | Google-w2v |Tokens, POS + PUNCT | 0.76 | 0.62 | 0.685 |
| Ours - II | Google-w2v |Tokens, Deps, POS + PUNCT | 0.79 | 0.615 | 0.69 |


# Points to note:

* Only the successful variations are retained in the following experiments
* Glove seems to give a boost that Google-w2v in all cases
* Use of actual Punctutation instead of just "PUNCT" pos tag. Helps in most cases. (Shows the dependece of definition on punctuations)
* Use of dependency features has a positive effect on the results (reasonable)
* Pre-trained embedding performs better than the self trained.
* Feature extraction from the dependency relation independent of the word feature has positive effect (intuitively)


# Model description

Ours - I : (_data_manager.py:144 build_model2)
==========


This archetecture takes two inputs:
Input 1: Tokens
Input 2: Any Word level feature (eg. POS)

For Input-1: use pre trained embedding
For Input-2: Embedding is learned on the fly

Embedding of Input-1 and Input-2 are concatenated at word level

Followed by 2 units of feature extraction combo

each unit contains
    Bilstm
    Conv
    Maxpool

Later, flattened and fed to a classifier layer through a fully connected layer.


Ours - II: (_data_manager.py:87 build_model3)
==========


This archetecture takes five inputs:

word level:
Input 1: Tokens
Input 2: Any Word level feature (eg. POS)

len(Input 1) == len(Input 2)

Sentence level(dep relation):
Input 3: Head words
Input 4: Modifier words
Input 5: dependency labels

len(Input 3) == len(Input 4) == len(Input 5)

Feature extraction from Input 1 and Input 2 are same as Ours - I

Where as,
Feature extraction from other inputs are slightly different.
Bilstm layers are skipped. as the dependency relations are sequentialy independent.

After the feature extraction. Hidden representations from both the track are concatenated and rest is similar to Ours - I


Adv
---

* Processing(feature extration) wordlevel(tokens, pos) and sentence level features(dependency relations) seperately. and combinig them at higher level
* SADE uses (convolution, maxpooling), ours uses (Bilstm, conv, maxpooling). Usage of bilstm before conv to make use of the sequential information(sentence).
* Bilstm is not used processing dependency relations in Ours - II. As the dependency relations are sequentialy independent of each other.

# Points to note 2

Epochs: 100
Early stopping: patience: 10; monitor: val_loss
batch size: 128
loss: Binary cross entropy
optimizer: Adam

Data split:
test data: 10% of all data
validation data: 10% of train

W2V embedding dimension: 300
Raw vocab size = 26k
* Vocab size of 26k from 17k samples is too high
Reasons for vocabulary explosion:
1. Lots of urls
2. Numbers like values and years
3. Strange alpha numeric combination due to noise in text
4. Improperly parsed word due to irregular use of Punctuations
5. Lack of space around "+ - \} \{ \) \( \[ \] = /" etc.. causes bad tokens

Ambiguity:
Some definition of definitions are not so clear
* " Organisms are individual living entities ." 1 : This is a definition
* " Organelles are small structures that exist within cells ." 0 : this is not a definition
* " 149 . Nucleic acids are the most important macromolecules for the continuity of life ."	"0"
* " They carry the genetic blueprint of a cell and carry instructions for the functioning of the cell ."	"1"
* " 4888 . Recall from The Macroeconomic Perspective that if exports exceed imports , the economy is said to have a trade surplus ."	"0"
* " If imports exceed exports , the economy is said to have a trade deficit ."	"1"
* " 4918 . To build a useful macroeconomic model , we need a model that shows what determines total supply or total demand for the economy , and how total demand and total supply interact at the macroeconomic level ."	"1"
* " 221 . However , the cell membrane detaches from the wall and constricts the cytoplasm ."	"1"