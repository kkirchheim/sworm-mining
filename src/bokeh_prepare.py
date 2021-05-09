"""
Contains code to prepare a single dataframe to be used as data source for the bokeh demo server
"""
import click
import pandas as pd
import pickle
import numpy as np
import logging

# own imports
import utils as utils
from constants import *

log = logging.getLogger(__name__)


def pre(x):
    if type(x) == list:
        return ", ".join(x)
    return str(x)


@click.command()
def main():
    """
    Prepare a single hash map to be used as data source for the bokeh demo server
    """
    utils.configure_logging()

    # load data
    df = pd.read_pickle(join(ARTIFACTS_DIR, "journals-with-stm-topics.pkl"))
    df = df[~df["dc:description"].isna()]
    X_embedded = pickle.load(open(join(BOKEH_DIR, "X-embedding-stm-tfidf.pkl"), "rb"))
    y_pred = list(df["stm:topics"].apply(lambda x: x[0]))
    topics_list = list(pd.read_csv(join(ARTIFACTS_DIR, "Topics_40.csv"), delimiter=";").columns[1:])

    df["prism:publicationName"] = df["prism:publicationName"]\
        .apply(lambda x: x.replace("&amp;", "and"))\
        .apply(lambda x: x.title())\
        .apply(lambda x: x.replace("Affilia - Journal Of Women And Social Work", "Affilia"))\
        .apply(lambda x: x.replace("The Social Service Review", "Social Service Review"))

    log.info(df["prism:publicationName"].unique())
    log.info(len(df["prism:publicationName"].unique()))

    df["affiliation:country"] = df["affiliation:country"]\
        .apply(lambda x: None if type(x) is float else x[0])\
        .apply(lambda x: x or "Unknown")

    df["citedby-count"] = df["citedby-count"].apply(lambda x: x or 0)

    # for some reason we have to divide by 1x10^6
    df["ts"] = df["prism:coverDate"].values.astype(np.int64) // 1e6
    df["author:name:pretty"] = df["author:name"].apply(pre)

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

    data = {"x1": X_embedded[:, 0],
            "x2": X_embedded[:, 1],
            "x1_backup": X_embedded[:, 0],
            "x2_backup": X_embedded[:, 1],
            "label": df["stm:topics:top"],
            "label2": y_pred,
            "topics": df["stm:topics:pretty"],
            "title": df["dc:title"],
            "date": df["prism:coverDate"].apply(lambda x: x.strftime("%d.%m.%Y")),
            "timestamp": df["ts"],
            "author": df["author:name:pretty"],
            "journal": df["prism:publicationName"],
            "cluster": [topics_list[c] for c in y_pred],
            "abstract": df["dc:description"],
            "doi": df["prism:doi"],
            "citations": df["citedby-count"],
            "country": df["affiliation:country"]
            }

    path = join(BOKEH_DIR, "data.pkl")
    log.info(f"Saving to {path}")
    pd.DataFrame(data).to_pickle(path)

    path = join(BOKEH_DIR, "topic-list.pkl")
    log.info(f"Saving to {path}")
    pd.DataFrame(data=topics_list, columns=["topic"]).to_pickle(path)

    path = join(BOKEH_DIR, "journal-list.pkl")
    log.info(f"Saving to {path}")
    pd.DataFrame(data=df["prism:publicationName"].unique(), columns=["journal"]).to_pickle(path)


if __name__ == "__main__":
    main()
