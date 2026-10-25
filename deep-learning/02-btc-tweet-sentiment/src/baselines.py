"""Non-neural baselines: dictionary sentiment and TF-IDF + XGBoost."""

import numpy as np
import pysentiment2 as ps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier


def dictionary_polarity(texts):
    """
    Polarity score in [-1, 1] from the Harvard IV-4 dictionary:
    (positive words - negative words) / (positive words + negative words).
    No training is involved, so the score can be evaluated on any sample.
    """
    hiv4 = ps.HIV4()
    return np.array([hiv4.get_score(hiv4.tokenize(text))["Polarity"] for text in texts])


def train_tfidf_xgboost(train_texts, y_train, n_estimators=200, seed=42):
    """
    Fit TF-IDF word features and an XGBoost classifier as one sklearn pipeline.

    The vectorizer is fitted on the training texts only. Class imbalance is
    handled with scale_pos_weight = #negatives / #positives.
    Use `.predict_proba(test_texts)[:, 1]` on the returned pipeline.
    """
    y_train = np.asarray(y_train)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = make_pipeline(
        TfidfVectorizer(),
        XGBClassifier(
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="logloss",
        ),
    )
    model.fit(train_texts, y_train)
    return model
