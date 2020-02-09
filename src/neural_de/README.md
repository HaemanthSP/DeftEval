# README #

***Currently work in progress***

This is the repository associated with the NAACL 2018 short paper __Syntactically Aware Neural Definition Extraction__ [1]. It is possible to reproduce the results in Tables 1 and 2 with the following commands.

# Table 1 #
```python3 src/_main.py -d data/wcl_datasets_v1.2/ -wv vectors -dep X```

* `-d` the path for the target dataset (for a 10-f CV experiment). 
* `-wv` the path for your word embeddings (we used w2v Google News)
* `-dep` whether (and which) dependency information to include. Available options are: `n` no syntax, `m` only head-modifier averaged vector, and `ml` head-modifier averaged vector and dependency label.

For example:

```python3 src/_main.py -d data/wcl_datasets_v1.2/ -wv data/GoogleNews-vectors-negative300.bin -dep ml```

# Table 2 #
```python3 src/_main2.py -wv data/GoogleNews-vectors-negative300.bin -dep ml -p X```

* `-p` the path of the definition extraction keras model. If it does not exist, it will be trained and saved in that same path.

For example:

``` python3 src/_main2.py -wv data/GoogleNews-vectors-negative300.bin -dep ml -p data/models/wcl_ml ```

# Retrieve Definitions from Corpora #

If you wish to extract definitions from a target corpus, we provide an additional script.

```python3 src/_definition_retrieval.py -d X -dep J -m Y -wv vectors -dep Z -p W -t T```

* `-d` corpus file, should be a text file with one sentence per line
* `-m` maximum number of sentences to be processed (in order)
* `-t` confidence threshold for the retrieved definitions (float, between 0 and 1)

For example.

```python3 src/_definition_retrieval.py -d data/dev_acl-arc_500k_raw.txt -dep ml -m 100 -wv ../resources/embeddings/GoogleNews-vectors-negative300.bin -p data/models/wcl_ml -t 0.8```

Note that the `-dep` optio n must match the option with which the loaded model was trained. We provide a pretrained model on the WCL dataset at `data/models/wcl_ml`.

It is possible to obtain a subset of the ACL-ARC corpus [2] for quick experimentation from https://bitbucket.org/luisespinosa/defext (see README).

The script will print to stdout all definitions found with confidence above threshold.

### Dependencies ###

- gensim
- numpy
- keras
- spacy
- The pretrained word2vec embeddings used in the paper can be downloaded from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

### References ###

[1] Espinosa-Anke, L., Schockaert, S. (2018). Syntactically Aware Neural Architectures for Definition Extraction. NAACL 2018 Short Papers.

[2] Bird, S., Dale, R., Dorr, B. J., Gibson, B., Joseph, M. T., Kan, M. Y., ... & Tan, Y. F. (2008). The acl anthology reference corpus: A reference dataset for bibliographic research in computational linguistics.
