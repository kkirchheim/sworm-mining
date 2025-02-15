"""
Contains code to prepare a single dataframe to be used as data source for the bokeh demo server
"""
import logging
import click
import pandas as pd
import numpy as np
import pickle

# own imports
import utils as utils
from constants import *

log = logging.getLogger(__name__)


def pre(x):
    if type(x) == list:
        return ", ".join(x)
    return str(x)


@click.group()
def cli():
    utils.configure_logging()


@cli.command("dataframe")
def main():
    """
    Prepare a single hash map to be used as data source for the django demo server
    """

    df = pd.read_pickle(join(ARTIFACTS_DIR, "journals-with-stm-topics.pkl"))
    df = df[~df["dc:description"].isna()]
    X_embedded = pickle.load(open(join(BOKEH_DIR, "X-embedding-stm-tfidf.pkl"), "rb"))
    y_pred = list(df["stm:topics"].apply(lambda x: x[0]))
    topics_list = list(pd.read_csv(join(ARTIFACTS_DIR, "Topics_40.csv"), delimiter=";").columns[1:])

    preprocess_journal_names(df)
    preprocess_country(df)

    df["citedby-count"] = df["citedby-count"].apply(lambda x: x or 0)

    # for some reason we have to divide by 1x10^6
    df["ts"] = df["prism:coverDate"].values.astype(np.int64) // 1e6
    df["author:name:pretty"] = df["author:name"].apply(pre)

    # load raw topic thetas
    df_stm_theta = pd.read_csv(CLEAN_TOPICS_STM, low_memory=False)
    df_stm_theta["index"] = df.index
    df_stm_theta.set_index("index", inplace=True)

    process_topics(df, topics_list)

    data = {"x1": X_embedded[:, 0],
            "x2": X_embedded[:, 1],
            "x1_backup": X_embedded[:, 0],
            "x2_backup": X_embedded[:, 1],
            "label": df["stm:topics:top"],
            "label2": y_pred,
            "topics": df["stm:topics:pretty"],
            "title": df["dc:title"],
            "date": df["prism:coverDate"],
            "timestamp": df["ts"],
            "author": df["author:name:pretty"],
            "journal": df["prism:publicationName"],
            "journal-issn": df["prism:issn"],
            "cluster": [topics_list[c] for c in y_pred],
            "abstract": df["dc:description"],
            "doi": df["prism:doi"],
            "citations": df["citedby-count"],
            "country": df["affiliation:country"],
            }

    df_django = pd.DataFrame(data)
    df_django.index.rename("index", inplace=True)
    path = join(BOKEH_DIR, "django-data.pkl")
    log.info(f"Saving to {path}")
    todrop = ~(df_django["journal-issn"].isna())
    log.info(f"Dropping {(~todrop).sum()}")
    df_django = df_django[todrop]
    df_django.to_pickle(path)

    path = join(BOKEH_DIR, "django-theta.pkl")
    log.info(f"Saving to {path}")
    df_stm_theta = df_stm_theta[todrop]
    df_stm_theta.to_pickle(path)

    path = join(BOKEH_DIR, "topic-list.pkl")
    log.info(f"Saving to {path}")
    pd.DataFrame(data=topics_list, columns=["topic"]).to_pickle(path)

    path = join(BOKEH_DIR, "journal-list.pkl")
    log.info(f"Saving to {path}")
    pd.DataFrame(data=df["prism:publicationName"].unique(), columns=["journal"]).to_pickle(path)


def process_topics(df, topics_list):
    topic_strings = []
    topic_top = []
    for topics, props in zip(df["stm:topics"], df["stm:topics:probs"]):
        s = ""
        m = len(topics)

        for n, (topic, prop) in enumerate(zip(topics, props)):
            s += f"{topics_list[topic]} ({prop:.1%})"
            if n < m - 1:
                s += ", "

        top_topic = topics[np.argmax(props)]

        topic_strings.append(s)
        topic_top.append(top_topic)
    df["stm:topics:pretty"] = pd.Series(topic_strings, index=df.index)
    df["stm:topics:top"] = pd.Series(topic_top, index=df.index)


def preprocess_country(df):
    df["affiliation:country"] = df["affiliation:country"] \
        .apply(lambda x: None if type(x) is float else x[0]) \
        .apply(lambda x: x or "Unknown")


def preprocess_journal_names(df):
    df["prism:publicationName"] = df["prism:publicationName"] \
        .apply(lambda x: x.replace("&amp;", "and")) \
        .apply(lambda x: x.title()) \
        .apply(lambda x: x.replace("Affilia - Journal Of Women And Social Work", "Affilia")) \
        .apply(lambda x: x.replace("The Social Service Review", "Social Service Review"))
    log.info(df["prism:publicationName"].unique())
    log.info(len(df["prism:publicationName"].unique()))


@cli.command("topics")
def prepare_topics():
    """
    Extract top-topics from from dataframe
    """
    df = pd.read_pickle(JOURNALS_DF)
    df_topics = pd.read_csv(CLEAN_TOPICS_STM, low_memory=False)

    data = np.array(df_topics.values[:, 1:])

    relevant_topics = np.argsort(data, axis=1)[:, :10:-1][:, :10]

    topics = []
    props = []

    for n, a in enumerate(relevant_topics):
        doc_topics = list(a)
        doc_topics_probs = list(data[n, a])

        tops = []
        probs = []

        for t, p in zip(doc_topics, doc_topics_probs):
            if p < 0.1 and len(tops) > 0:
                break
            else:
                tops.append(t)
                probs.append(p)

        topics.append(tops)
        props.append(probs)

    df = df[~df[ABSTRACTS].isna()]
    df["stm:topics"] = pd.Series(topics, index=df.index)
    df["stm:topics:probs"] = pd.Series(props, index=df.index)
    pd.to_pickle(df, join(ARTIFACTS_DIR, "journals-with-stm-topics.pkl"))


if __name__ == "__main__":
    main()
