"""Text cleaning for informal social media text (tweets)."""

import re
import string
import warnings

import emoji
import nltk
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]

# Negation words carry sentiment ("not bullish"), so they are kept even
# though NLTK lists them as stopwords.
NEGATIONS = {"no", "not", "nor"}

URL_PATTERN = re.compile(r"http\S+|www\S+")
MENTION_PATTERN = re.compile(r"@\w+")


def download_nltk_resources():
    """Download the NLTK data needed by TextPreprocessor (run once)."""
    for resource in NLTK_RESOURCES:
        nltk.download(resource, quiet=True)


class TextPreprocessor:
    """
    Cleans tweets for bag-of-words and RNN models.

    Steps: lowercase, strip HTML, URLs, @mentions, emojis, punctuation and
    accents, then tokenize, optionally remove stopwords and lemmatize.
    Hashtag words are kept (only the '#' symbol is removed) because they
    often carry content, e.g. #bullish.
    """

    def __init__(self, remove_stopwords=True, lemmatize=True, keep_negations=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words("english"))
        if keep_negations:
            self.stop_words -= NEGATIONS
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        text = URL_PATTERN.sub("", text)
        # Mentions must be removed before punctuation, otherwise the '@' is
        # already gone and the pattern no longer matches.
        text = MENTION_PATTERN.sub("", text)
        text = emoji.replace_emoji(text, replace="")
        # unidecode first, so that e.g. curly quotes and full-width '！' become
        # ASCII punctuation and are removed in the next step.
        text = unidecode(text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()

        tokens = nltk.word_tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)


def light_clean(text: str) -> str:
    """
    Minimal cleaning for transformer models (DistilBERT).

    Transformers are pretrained on natural text, so casing, punctuation and
    stopwords are kept; only URLs, mentions and extra whitespace are removed.
    """
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
