# Bitcoin Tweet Sentiment Analysis

*Based on the second assignment of the course "Deep Learning in Text Analytics"
(Prof. Stefan Lessmann, Humboldt-Universität zu Berlin). Reorganized and revised.*

## Question

How well do different text-analysis methods measure sentiment in short financial
social media posts? And how much do machine learning and transformer models improve
on the dictionary (word-list) approach that is widely used to measure tone in
accounting and finance research?

## Data

1,500 training and 500 test tweets about Bitcoin, each labeled positive or
negative. About 81% of the tweets are positive, so AUC and F1 are emphasized
over accuracy.

## Models

| # | Model | Idea |
|---|---|---|
| 1 | Dictionary (Harvard IV-4) | Count positive vs. negative words, no training |
| 2 | TF-IDF + XGBoost | Bag-of-words features + gradient boosted trees |
| 3 | Bidirectional GRU | Recurrent network, embeddings learned from scratch |
| 4 | Bidirectional LSTM + Word2Vec | Recurrent network, pretrained word embeddings |
| 5 | DistilBERT (zero-shot) | Transformer trained on movie reviews, applied as is |
| 6 | DistilBERT (fine-tuned) | Transformer fine-tuned on the tweets |

## Results

The full results table, confusion matrices and ROC curves are in
[`sentiment_analysis.ipynb`](sentiment_analysis.ipynb), section 5.

## Project structure

The notebook is the "do-file": it runs the analysis step by step and shows the
results. The reusable code behind it is in `src/`.

```
02-btc-tweet-sentiment/
├── sentiment_analysis.ipynb   analysis and results
├── src/
│   ├── preprocessing.py       tweet cleaning
│   ├── baselines.py           dictionary score, TF-IDF + XGBoost
│   ├── rnn.py                 vocabulary, embeddings, GRU/LSTM builder
│   ├── bert.py                DistilBERT zero-shot and fine-tuning
│   └── evaluation.py          metrics and plots shared by all models
├── data/                      tweet datasets (train / test)
└── requirements.txt
```

## How to run

1. Install Python 3.12 and the packages: `pip install -r requirements.txt`
2. Download the pretrained Word2Vec file `w2v_imdb_full_d100_e500.model`
   ([Google Drive](https://drive.google.com/file/d/1J1edk-jyPsO0PCiY-NEFiRxo-PIYART3/view?usp=drive_link))
   and place it in `data/`.
3. Open `sentiment_analysis.ipynb` from this folder and run all cells.
   The DistilBERT models are downloaded automatically from Hugging Face on first use.

## Revision notes

Compared with the original course submission, the code was reorganized into
`src/` modules and the following issues were corrected:

* **Results table:** the RNN metrics were stored in the wrong order, so values
  appeared in the wrong rows (e.g. the reported AUC was actually the F1 score).
  Evaluation now returns metrics by name, which rules out this kind of error.
* **TF-IDF features:** the vectorizer split tweets into single characters instead
  of words. It now uses words.
* **Dictionary model:** it was evaluated on the training set; it is now evaluated
  on the test set like all other models.
* **Fine-tuned DistilBERT:** the 0.5 cutoff was applied to raw model outputs
  (logits) instead of probabilities. A softmax is now applied first, and the
  test set is no longer passed to the trainer.
* **Preprocessing:** negation words (*not, no, nor*) are kept, since removing
  them can flip the sentiment. Transformer models receive lightly cleaned text,
  closer to the natural text they were pretrained on.
* **Word2Vec:** words outside the 4,000-word vocabulary were counted as missing
  from the pretrained embeddings, which overstated the number of missing words.
