#!/usr/bin/env python
"""
Scripts for preprocessing
"""
from os.path import join
from os import listdir
import numpy as np
import pandas as pd
import spacy
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import click
import logging
import pickle5 as pickle
import csv
import re
from gensim.utils import simple_preprocess, lemmatize
from collections import OrderedDict, Counter

# local imports
import constants as const
import utils
from src.constants import AFFILIATION_COUNTRY, AFFILIATION_CITY, AFFILIATION_NAME, AFFILIATION_ID, GEPHI_LABEL, \
    GEPHI_ID, AUTHOR_AFFIL, ABSTRACTS, IDENTIFIER, AUTHOR_NAME, AUTHOR_ID

log = logging.getLogger(__name__)

# globals
cls = spacy.util.get_lang_class("en")  # 1. Get Language instance, e.g. English()
nlp = cls()


def setup_nltk():
    import nltk
    nltk.data.path.append(const.CACHE_DIR)
    nltk.download('wordnet', download_dir=const.CACHE_DIR)
    nltk.download('punkt', download_dir=const.CACHE_DIR)
    nltk.download('stopwords', download_dir=const.CACHE_DIR)


def tokenize(x):
    if x is None or type(x) is not str:
        return ""

    doc = nlp(x)
    filtered = [token.lemma_ for token in doc if not token.is_stop and token.text.isalpha()]
    return " ".join(filtered).lower()


def preprocess_authkeywords(x):
    if type(x) is not str:
        return []
    else:
        return [y.strip() for y in x.split("|")]


def extract_keywords_from_abstract(max_features, top_keywords, tokenized):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(tokenized)
    x = X.todense()

    log.info(f"Memory: {x.nbytes / 1024 / 1024} MB")

    # top extracted keywords
    top_keywords_index = np.argsort(x, axis=1)[:, ::-1][:, :top_keywords]
    doc_keywords = []
    remap = {index: k for k, index in vectorizer.vocabulary_.items()}
    for n, values in enumerate(top_keywords_index):
        doc = []
        if tokenized[n] != "":
            for i in values.tolist()[0]:
                doc.append(str(remap[i]))
        doc_keywords.append(doc)

    all_keywords = vectorizer.vocabulary_.keys()
    return doc_keywords, all_keywords


def tokenize_abstracts(df):
    t1_start = time.perf_counter()
    tokeized = df[ABSTRACTS].apply(tokenize)
    t1_end = time.perf_counter()
    log.info(f"Time: {t1_end - t1_start}")
    return tokeized


def convert_dates(df):
    df["prism:coverDate_D"] = df["prism:coverDate"].round("D")


class GensimProcessor:
    def __init__(self, n_docs):
        from nltk.corpus import stopwords
        self.stop_words = stopwords.words("english")
        self.n_docs = n_docs
        self.errors = 0
        self.processed = 0

    def __call__(self, doc):
        tokens = simple_preprocess(doc, deacc=True)
        out = []
        for token in tokens:
            try:
                if token not in self.stop_words:  # remove stopwords
                    lemmatized_word = lemmatize(token, allowed_tags=re.compile('(NN|JJ|RB)'))  # lemmatize
                    if lemmatized_word:
                        out += [lemmatized_word[0].split(b'/')[0].decode('utf-8')]
                else:
                    continue
            except Exception:
                log.error(f"Could not process: {token}")
                self.errors += 1

        self.processed += 1
        if self.processed % 1000 == 0:
            log.info(f"Processed: {self.processed / self.n_docs:.2%} Errors: {self.errors}")

        return out


def extract_affiliations(df):
    existing_affil = df["affiliation"][~df["affiliation"].isna()]
    affids = existing_affil.apply(lambda x: [d["afid"] for d in x])
    affilname = existing_affil.apply(lambda x: [d["affilname"] for d in x])

    aff_city = existing_affil.apply(lambda x: [d["affiliation-city"] for d in x])
    aff_country = existing_affil.apply(lambda x: [d["affiliation-country"] for d in x])
    return affids, affilname, aff_city, aff_country


def process_author_affiliation(x):
    results = []
    try:
        for d in x:
            if "afid" in d:
                r = [e["$"] for e in d["afid"]]
            else:
                r = []

            results.append(r)
    except Exception as ex:
        log.exception(ex)
        log.info(d)

    return results


def extract_authors(df):
    existing_author = df["author"][~df["author"].isna()]
    authid = existing_author.apply(lambda x: [d["authid"] for d in x])
    authname = existing_author.apply(lambda x: [d["authname"] for d in x])
    authaffil = existing_author.apply(process_author_affiliation)
    auth_givenname = existing_author.apply(lambda x: [d["given-name"] for d in x])
    auth_surname = existing_author.apply(lambda x: [d["surname"] for d in x])

    return authid, authname, authaffil, auth_givenname, auth_surname


def gensim_preprocess(df) -> pd.Series:
    """
    Preprocessing for gensim
    """
    n_docs = (~df["dc:description:tokenized"].isnull()).sum()
    processor = GensimProcessor(n_docs)

    texts = df["dc:description:tokenized"][~df["dc:description:tokenized"].isnull()].apply(processor)
    return texts


def merge_dataframes(files) -> pd.DataFrame:
    """
    Create initial dataframe from individual downloaded dataframes. Duplicate dc:identifiers are being dropped.
    :param files:
    :return:
    """
    df = pd.DataFrame()
    n_rows_dropped = 0
    n_files_dropped = 0

    for file in files:
        indx = df.index.unique()
        log.debug(f"Unique Indexes: {len(indx)}")

        try:
            log.info(f"Loading '{file}'")
            tmp_df = pickle.load(open(file, "rb"))
            duplicate = tmp_df[IDENTIFIER].isin(indx)
            n_duplicates = duplicate.sum()

            if n_duplicates > 0:
                n_rows_dropped += n_duplicates
                log.warning(f"Duplicate indexes: {n_duplicates}")

            tmp_df = tmp_df[~duplicate].set_index(IDENTIFIER)

            df = df.append(tmp_df, ignore_index=False, verify_integrity=True, sort=True)

        except KeyError as e:
            log.error("File is missing 'dc:identifier'")
            # log.exception(e)
            n_files_dropped += 1

    if n_rows_dropped > 0:
        log.error(f"Dropped Rows: {n_rows_dropped}")
    if n_files_dropped > 0:
        log.error(f"Dropped DFs: {n_files_dropped}")

    return df


def create_author_map(df, all_authors):
    """
    Counting occurences here is much faster then grouping the dataframe.
    """
    author_map = OrderedDict()

    df = df[~df[AUTHOR_ID].isna()]
    for index, data in df[[AUTHOR_ID, AUTHOR_NAME, AUTHOR_AFFIL]].iterrows():
        idents = data[AUTHOR_ID]
        names = data[AUTHOR_NAME]

        for n, (ident, name) in enumerate(zip(idents, names)):
            affil = data[AUTHOR_AFFIL][n]

            if ident in all_authors:
                if ident not in author_map:
                    author_map[ident] = {}
                    author_map[ident]["name"] = name
                    author_map[ident]["count"] = 0
                    author_map[ident]["affiliations"] = []

                author_map[ident]["affiliations"].extend(affil)
                author_map[ident]["count"] += 1

    for key, value in author_map.copy().items():
        counter = Counter(value["affiliations"])
        c = counter.most_common(1)
        if len(c) > 0:
            author_map[key]["affiliations:top"] = c[0][0]
        else:
            author_map[key]["affiliations:top"] = None

    return author_map


def create_affiliation_map(df, all_affiliations):
    """
    Counting occurences here is much faster then grouping the dataframe.
    """
    affil_map = OrderedDict()

    df = df[~df[AFFILIATION_ID].isna()]
    for index, data in df[[AFFILIATION_ID, AFFILIATION_NAME, AFFILIATION_CITY, AFFILIATION_COUNTRY,
                           AUTHOR_ID]].iterrows():
        idents = data[AFFILIATION_ID]
        names = data[AFFILIATION_NAME]
        country = data[AFFILIATION_COUNTRY][0]
        city = data[AFFILIATION_CITY][0]

        for n, (ident, name) in enumerate(zip(idents, names)):

            if ident in all_affiliations:
                if ident not in affil_map:
                    affil_map[ident] = {}
                    affil_map[ident]["name"] = name
                    affil_map[ident]["count"] = 0
                    affil_map[ident]["country"] = country
                    affil_map[ident]["city"] = city

                affil_map[ident]["count"] += 1

    return affil_map


@click.group()
def cli():
    """
    Command line interface
    """
    pass


@cli.command("affiliations")
def affiliations():
    """
    Extract affiliation information
    """
    utils.configure_logging()
    log.info("Loading dataframe")
    journals_db = pd.read_pickle(const.JOURNALS_DF)
    affil_ids = journals_db[AFFILIATION_ID]
    affil_ids = affil_ids[~affil_ids.isna()]
    affil_ids = affil_ids.explode().value_counts().index
    log.info(f"Unique Affiliations: {len(affil_ids)}")

    log.info("Creating Database")
    t1 = time.perf_counter()
    affil_map = create_affiliation_map(journals_db, all_affiliations=affil_ids)
    affil_db = pd.DataFrame()

    author_ids = affil_map.keys()
    author_data = affil_map.values()

    affil_db[GEPHI_ID] = pd.Series(author_ids)
    affil_db[GEPHI_LABEL] = pd.Series(k["name"] for k in author_data)  # for gephi
    affil_db["affiliation:country"] = pd.Series(k["country"] for k in author_data)
    affil_db["affiliation:city"] = pd.Series(k["city"] for k in author_data)
    affil_db["affiliation:count"] = pd.Series(k["count"] for k in author_data)

    t2 = time.perf_counter()
    log.info(f"Time: {t2 - t1:.2f} s")

    affil_db.set_index(GEPHI_ID, inplace=True)

    log.info("Saving Database")
    affil_db.to_pickle(const.AFFILIATIONS_DF)

    log.info("Exporting Nodelist")
    affil_db.to_csv(const.AFFILIATIONS_DF_CSV, sep=";", quoting=csv.QUOTE_ALL)

    log.info("Done")


@cli.command("authors")
def authors():
    """
    Extract author information
    """
    utils.configure_logging()
    log.info("Loading dataframe")
    journals_db = pd.read_pickle(const.JOURNALS_DF)

    author_ids = journals_db[AUTHOR_ID]
    author_ids = author_ids[~author_ids.isna()]
    all_authors = author_ids.explode().value_counts().index
    # log.info(all_authors)
    log.info(f"Unique Authors: {len(all_authors)}")

    log.info("Creating Database")
    t1 = time.perf_counter()
    author_map = create_author_map(journals_db, author_ids.explode().value_counts().index)
    authors_db = pd.DataFrame()

    author_ids = author_map.keys()
    author_data = author_map.values()

    authors_db[GEPHI_ID] = pd.Series(author_ids)
    authors_db[GEPHI_LABEL] = pd.Series(k["name"] for k in author_data)  # for gephi
    authors_db["author:name"] = pd.Series(k["name"] for k in author_data)  # for gephi

    # authors_db["occurences"] = pd.Series(k["count"] for k in author_data)
    authors_db["author:affiliations"] = pd.Series(k["affiliations"] for k in author_data)
    aff_counts = [Counter(k["affiliations"]) for k in author_data]
    authors_db["author:affiliations:unique"] = pd.Series(list(k.keys()) for k in aff_counts)
    authors_db["author:affiliations:counts"] = pd.Series(list(k.values()) for k in aff_counts)
    authors_db["author:affiliations:top"] = pd.Series(k["affiliations:top"] for k in author_data)
    authors_db["author:count"] = pd.Series(k["count"] for k in author_data)

    t2 = time.perf_counter()
    log.info(f"Time: {t2 - t1:.2} s")

    authors_db.set_index(GEPHI_ID, inplace=True)

    log.info("Saving Database")
    authors_db.to_pickle(const.AUTHORS_DF)

    log.info("Exporting Nodelist")
    authors_db.to_csv(const.AUTHORS_DF_CSV, sep=";", quoting=csv.QUOTE_ALL)

    log.info("Done")


@cli.command("journals")
@click.option("-m", "--max-features", "max_features", type=int, default=5000,
              help="Number of features to consider (5000)")
@click.option("-k", "--n-keywords", "n_keywords", type=int, default=20,
              help="Number of keywords to extract (20)")
def journals(max_features, n_keywords=20):
    """
    Process dataframes of downloaded journals.
    """
    utils.configure_logging()
    setup_nltk()

    log.info("Loading dataframes")
    files = [join(const.JOURNALS_DIR, f) for f in listdir(const.JOURNALS_DIR) if f.endswith(".pkl")]
    df = merge_dataframes(files)

    log.info(f"Documents: {len(df)}")
    log.info(f"Empty Abstracts: {df[ABSTRACTS].isna().sum()}")
    log.info(f"Abstracts: {(~df[ABSTRACTS].isna()).sum()}")

    log.info("Converting dates")
    convert_dates(df)

    log.info("Extracting Authors")
    authid, authname, authaffil, auth_givenname, auth_surname = extract_authors(df)
    df[AUTHOR_ID] = authid
    df[AUTHOR_NAME] = authname
    df[AUTHOR_AFFIL] = authaffil
    df["author:name:given"] = auth_givenname
    df["author:name:sur"] = auth_surname
    log.info(f"Got {len(authid.explode().unique())} unique authors")

    log.info("Extracting affiliations")
    affids, affilname, aff_city, aff_country = extract_affiliations(df)
    df[AFFILIATION_ID] = affids
    df[AFFILIATION_NAME] = affilname
    df[AFFILIATION_CITY] = aff_city
    df[AFFILIATION_COUNTRY] = aff_country
    log.info(f"Got {len(affids.explode().unique())} unique affiliations")
    log.info(f"Got {len(aff_country.explode().unique())} affiliation countries")

    log.info("Preprocessing Author Keywords")
    df["authkeywords:preprocessed"] = df["authkeywords"].apply(preprocess_authkeywords)

    log.info("Tokenizing abstracts")
    tokenized = tokenize_abstracts(df)
    df["dc:description:tokenized"] = tokenized

    log.info("Extracting keywords")
    doc_keywords, all_keywords = extract_keywords_from_abstract(max_features, n_keywords, tokenized)
    df["dc:description:keywords"] = pd.Series(data=doc_keywords, index=df.index)

    texts = gensim_preprocess(df)
    df["dc:description:tokenized:gensim"] = texts

    df.to_csv(const.JOURNALS_DF_CSV)
    df.to_pickle(const.JOURNALS_DF)
    log.info("Done.")


if __name__ == "__main__":
    cli()
