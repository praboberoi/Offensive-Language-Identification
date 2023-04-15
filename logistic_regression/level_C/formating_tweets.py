import pandas as pd
import spacy
import re
import emoji
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
import os
from spellchecker import SpellChecker
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from functools import partial

# Load the English language model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Initialize the SpellChecker
# spell = SpellChecker()

# Define the pre-processing function
def preprocess(text):
    # Remove @user
    text = re.sub(r'@[\w]+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove dates
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b', '', text)

    # Convert emojis to text
    text = emoji.demojize(text)

    # Lemmatization
    doc = nlp(text)
    tweet_tokens = [token.lemma_ for token in doc if (not token.is_stop and not token.is_punct)]
    return ' '.join(tweet_tokens)

def format_tweets(df):
    dask_df = dd.from_pandas(df, npartitions=os.cpu_count())
    dask_df['text'] = dask_df['text'].map_partitions(
        lambda x: x.dropna().apply(preprocess),
        meta=('text', object)
    )
    with ProgressBar():
        result = dask_df.compute()
    return result

if __name__ == '__main__':
    df = pd.read_csv("./task_c_distant_ann.tsv", sep="\t", header=0)
    df = format_tweets(df)
    df.to_csv('./formated_file_c.tsv', sep="\t", header=True, index=False)
