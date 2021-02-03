#!/usr/bin/env python
"""
Create different models from data
"""
import click
import logging
import gensim
from gensim import corpora
from os.path import join
import src.constants as const
import pandas as pd
from gensim.models import LdaMulticore, LdaModel
from gensim.models.wrappers import LdaMallet
import sys


import utils

log = logging.getLogger(__name__)


@click.group()
def cli():
    """
    Create models from preprocessed data.
    """
    utils.configure_logging()
    pass


@cli.command("lda-gensim")
@click.option("-t", "--topics", "n_topics", type=int, default=20, help="Number of LDA topics")
@click.option("-m", "--multicore", "multicore", type=bool, is_flag=True, default=True, help="Use Multicore Model")
def create_lda_gensim(n_topics, multicore):
    """
    Create LDA model using GenSim.
    """
    df = pd.read_pickle(const.JOURNALS_DF)
    column_name = "dc:description:tokenized"

    if column_name not in df.columns:
        log.error(f"Could not find column: '{column_name}'. Preprocess with tokenization.")
        sys.exit(1)

    nan = df[column_name].isna()
    log.info(f"Empty: {nan.sum()}")

    documents = df[~nan][column_name].str.split().values

    log.info("Creating dict")
    dictionary = corpora.Dictionary(documents)

    log.info("Creating corpus")
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in documents]

    if multicore:
        lda_model = LdaMulticore(corpus=corpus,
                                 id2word=dictionary,
                                 random_state=42,
                                 num_topics=n_topics,
                                 passes=10,
                                 chunksize=1000,
                                 batch=False,
                                 alpha='asymmetric',
                                 decay=0.5,
                                 offset=64,
                                 eta=None,
                                 eval_every=0,
                                 iterations=100,
                                 gamma_threshold=0.001,
                                 per_word_topics=True)
    else:
        lda_model = LdaModel(corpus=corpus,
                             id2word=dictionary,
                             random_state=42,
                             num_topics=n_topics,
                             passes=10,
                             chunksize=1000,
                             alpha='asymmetric',
                             decay=0.5,
                             offset=64,
                             eta=None,
                             eval_every=0,
                             iterations=100,
                             gamma_threshold=0.001,
                             per_word_topics=True)

    path = join(const.MODELS_DIR, "lda-gensim")
    log.info(f"Saving to {path}")
    lda_model.save(path)

    log.info("Predicting on corpus")
    df_with_topics = predict_lda_gensim(df.copy(), dictionary, lda_model, column_name)
    path = join(const.ARTIFACTS_DIR, "journals-with-topics.pkl")
    log.info(f"Saving results to {path}")
    df_with_topics.to_pickle(join(const.ARTIFACTS_DIR, "journals-with-topics.pkl"))


def predict_lda_gensim(df, dictionary, lda_model, column_name) -> pd.DataFrame:
    log.info("Predicting...")
    nan = df[column_name].isna()
    log.info(f"Empty: {nan.sum()}")

    documents = df[~nan][column_name].str.split().values
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in documents]

    predictions = lda_model[corpus]
    predictions = list(predictions)
    topics = []
    props = []
    for result in predictions:
        topics.append([topic[0] for topic in result[0]])
        props.append([topic[1] for topic in result[0]])
    df["lda:topics"] = pd.Series(topics, index=df[~nan].index)
    df["lda:topics:top"] = pd.Series([topic[0] for topic in topics if len(topic) > 0], index=df[~nan].index)
    df["lda:topics:props"] = pd.Series(props, index=df[~nan].index)
    df["lda:topics:props:top"] = pd.Series([p[0] for p in props if len(p) > 0], index=df[~nan].index)
    return df


@cli.command("lda-mallet")
@click.argument("mallet_path") # path to mallet binary
@click.option("-t", "--topics", "n_topics", type=int, default=20, help="Number of LDA topics")
def create_lda_mallet(mallet_path, n_topics):
    """
    Create lda model using MALLET
    """
    df = pd.read_pickle(const.JOURNALS_DF)
    column_name = "dc:description:tokenized"

    if column_name not in df.columns:
        log.error(f"Could not find column: '{column_name}'. Preprocess with tokenization.")
        sys.exit(1)

    df_other = df[df[column_name] != ""]
    texts = df_other[column_name]

    documents = texts.str.split().values

    log.info("Creating dict")
    dictionary = corpora.Dictionary(documents)

    log.info("Creating corpus")
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in documents]

    lda_mallet = LdaMallet(mallet_path,
                           corpus=corpus,
                           num_topics=n_topics,
                           alpha=50,
                           id2word=dictionary,
                           workers=10,
                           prefix=None,
                           optimize_interval=0,
                           iterations=1000,
                           topic_threshold=0.0,
                           random_seed=42)

    lda_model2 = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_mallet)
    lda_model2.save(join(const.MODELS_DIR, "lda-gensim-mallet"))


if __name__ == '__main__':
    cli()
