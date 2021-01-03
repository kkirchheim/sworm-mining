#!/usr/bin/env python
"""
Scripts to create files for gephi
"""
from os.path import join
import pandas as pd
import csv
from collections import OrderedDict
import logging
import click
import time

#
import constants as const
import utils


log = logging.getLogger(__name__)


def get_cooccurences(documents, all_values):
    # a dic of dicts, one dic per keyword, containing co-occurences with other dicts
    occurrences = OrderedDict((str(key), OrderedDict((str(key), 0) for key in all_values)) for key in all_values)
    # Find the co-occurrences:
    for document in documents:
        for i in range(len(document)):
            for word in document[:i] + document[i + 1:]:
                if str(document[i]) in all_values and word in all_values:
                    occurrences[str(document[i])][word] += 1
    return occurrences


@click.group()
def cli():
    """
    Extract gephi files
    """
    pass


@cli.command("affiliation")
def authors_cooc():
    """
    Create cooc-matrix for affiliations
    """
    utils.configure_logging()
    log.info("Loading dataframe")
    df = pd.read_pickle(const.JOURNALS_DF)

    doc_affiliations = df[const.AFFILIATION_ID]
    doc_affiliations = doc_affiliations[~doc_affiliations.isna()]
    all_affiliations = doc_affiliations.explode().value_counts().index
    log.info(f"Unique Affiliations: {len(all_affiliations)}")

    log.info("Calculating co-occurences")
    t1 = time.perf_counter()
    occurrences = get_cooccurences(doc_affiliations, all_affiliations)
    t2 = time.perf_counter()
    log.info(f"Took {t2 - t1} s")
    co_occur = pd.DataFrame.from_records(occurrences)
    co_occur.to_csv(join(const.ARTIFACTS_DIR, "affiliations-cooc.csv"), sep=";", quoting=csv.QUOTE_ALL)

    log.info("Done.")


@cli.command("authors")
@click.option("-n", "--n-authors", "n_authors", type=int, default=1000)
def authors_cooc(n_authors):
    """
    Create cooc-matrix for authors
    """
    utils.configure_logging()
    log.info("Loading dataframe")
    df = pd.read_pickle(const.JOURNALS_DF)

    doc_authors = df["author:id"]
    doc_authors = doc_authors[~doc_authors.isna()]
    all_authors = doc_authors.explode().value_counts()[:n_authors].index
    log.info(f"Unique Authors: {len(all_authors)}")

    log.info("Calculating keyword co-occurences")
    t1 = time.perf_counter()
    occurrences = get_cooccurences(doc_authors, all_authors)
    t2 = time.perf_counter()
    log.info(f"Took {t2 - t1} s")
    co_occur = pd.DataFrame.from_records(occurrences)
    co_occur.to_csv(join(const.ARTIFACTS_DIR, "authors-cooc.csv"), sep=";", quoting=csv.QUOTE_ALL)

    log.info("Done.")


@cli.command("keywords")
def keywords_cooc():
    """
    Create cooc-matrix for keywords
    """
    utils.configure_logging()
    log.info("Loading dataframe")
    df = pd.read_pickle(const.JOURNALS_DF)

    doc_keywords = df["dc:description:keywords"]
    all_keywords = doc_keywords.explode().unique()
    log.info(f"Unique Keywords: {len(all_keywords)}")

    # create cooc matrix and save to disk
    log.info("Calculating keyword co-occurences")
    occurrences = get_cooccurences(doc_keywords, all_keywords)
    co_occur = pd.DataFrame.from_records(occurrences)
    co_occur.to_csv(join(const.ARTIFACTS_DIR, "abstracts-cooc.csv"), sep=";", quoting=csv.QUOTE_ALL)

    log.info("Done.")


if __name__ == "__main__":
    cli()


