#!/usr/bin/env python
"""
Scripts for preprocessing
"""
import csv
import logging
import re
import time
from collections import OrderedDict, Counter
from functools import partial
from os import listdir

import click
import numpy as np
import pandas as pd
import pickle5 as pickle  # legacy
import spacy
import tqdm
from gensim.utils import simple_preprocess, lemmatize
from sklearn.feature_extraction.text import TfidfVectorizer

# local imports
import utils
from constants import *
from src.utils import setup_nltk_cache

log = logging.getLogger(__name__)


class SpacyTokenizer:
    def __init__(self, model="en_core_web_lg"):
        self.nlp = spacy.load(model)

    def tokenize(self, document: str):
        """
        Remove stopwords, non-alpha numeric words, and lemmatize

        :param document: document
        :return: tokenized document
        """
        if type(document) is str:
            tokens = self.nlp(document)
            filtered = [token.lemma_ for token in tokens if not token.is_stop and token.text.isalpha()]
            return " ".join(filtered).lower()

        if type(document) is list:
            n_docs = len(document)
            log.info(f"Piping through Spacy... ({n_docs})")
            disable = ["tagger", "parser", "ner", "textcat"]
            docs_tokens = self.nlp.pipe(document)
            log.info("Gathering results")

            d = []

            for n, doc in enumerate(docs_tokens):
                f = [token.lemma_ for token in doc]  # if not token.is_stop and token.text.isalpha()
                r = " ".join(f).lower()

                if n % 1000 == 0:
                    log.info(f"Progress: {n / n_docs:.2%}")
                d.append(r)

            return d

        return ""

    def adjectives(self, document: str):
        if type(document) is str:
            doc_adjective_pairs = []

            tokens = self.nlp(document)
            filtered = [token for token in tokens if not token.is_stop and token.text.isalpha()]

            for token in filtered:
                if token.pos_ == "NOUN":
                    current_pairs = [t for t in token.children if t.pos_ == "ADJ"]

                    if len(current_pairs) > 0:
                        doc_adjective_pairs.append(current_pairs)

            return doc_adjective_pairs

        if type(document) is list:
            n_docs = len(document)
            log.info(f"Piping through Spacy... ({n_docs})")
            all_adjective_pairs = []

            disable = ["parser", "ner", "textcat"]
            docs = self.nlp.pipe(document, n_threads=-1, batch_size=10, disable=disable)

            for n, doc in enumerate(docs):
                doc_adjective_pairs = []
                filtered = [token for token in doc if not token.is_stop and token.text.isalpha()]
                for token in filtered:
                    if token.pos_ == "NOUN":
                        current_pairs = [t for t in token.children if t.pos_ == "ADJ"]
                        if len(current_pairs) > 0:
                            doc_adjective_pairs.append(current_pairs)

                all_adjective_pairs.append(doc_adjective_pairs)

                if n % 1000 == 0:
                    log.info(f"Progress: {n / n_docs:.2%}")

            return all_adjective_pairs

        return None
        # raise ValueError(f"Unsupported type: {type(document)}")


class GensimProcessor:
    """
    Text processor using GenSim
    """

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
                    # allowed_tags=re.compile('(NN|JJ|RB)')
                    lemmatized_word = lemmatize(token)
                    # log.info(lemmatized_word[0])
                    # log.info(type(lemmatized_word[0]))
                    # log.info(dir(lemmatized_word[0]))

                    if lemmatized_word:
                        out += [lemmatized_word[0].decode("utf-8")]
                        # [lemmatized_word[0].split(b'/')[0].decode('utf-8')]
                else:
                    continue
            except Exception:
                log.error(f"Could not process: {token}")
                self.errors += 1

        self.processed += 1
        if self.processed % 1000 == 0:
            log.info(f"Processed: {self.processed / self.n_docs:.2%} Errors: {self.errors}")

        return out


def process_remove_regex(x, regex):
    if type(x) is not str:
        return None

    try:
        x = x.replace("\\", "")
        return regex.sub("", x).strip()
    except Exception as e:
        log.error(e)

    return x


@utils.timed
def remove_copyright(df):
    """
    Removes copyright claim strings by regular expressions
    """
    abstracts = df[ABSTRACTS]
    for expression in COPYRIGHT_REGEX:
        log.info(f"Processing Regular Expression: '{expression}'")
        regex = re.compile(expression, re.IGNORECASE + re.MULTILINE)
        replace = partial(process_remove_regex, regex=regex)
        abstracts = abstracts.apply(replace)

    df[ABSTRACTS] = abstracts


@utils.timed
def process_base_authors(df):
    authid, authname, authaffil, auth_givenname, auth_surname = extract_basic_author_info(df)
    df[AUTHOR_ID] = authid
    df[AUTHOR_NAME] = authname
    df[AUTHOR_AFFIL] = authaffil
    df["author:name:given"] = auth_givenname
    df["author:name:sur"] = auth_surname
    log.info(f"Got {len(authid.explode().unique())} unique authors")


@utils.timed
def fix_journal_names(df):
    """
    Some journal names in the database are misspelled or not consistent
    """
    df["prism:publicationName"] = df["prism:publicationName"] \
        .apply(lambda x: x.replace("&amp;", "and")) \
        .apply(lambda x: x.title()) \
        .apply(lambda x: x.replace("Affilia - Journal Of Women And Social Work", "Affilia")) \
        .apply(lambda x: x.replace("The Social Service Review", "Social Service Review")) \
        .apply(lambda x: x.replace("Social Work (United States)", "Social Work"))


@utils.timed
def process_base_affilations(df):
    affids, affilname, aff_city, aff_country = extract_basic_affiliation_info(df)
    df[AFFILIATION_ID] = affids
    df[AFFILIATION_NAME] = affilname
    df[AFFILIATION_CITY] = aff_city
    df[AFFILIATION_COUNTRY] = aff_country
    log.info(f"Got {len(affids.explode().unique())} unique affiliations")
    log.info(f"Got {len(aff_country.explode().unique())} affiliation countries")


def preprocess_provided_keywords(x):
    """
    Preprocess keywords provided by authors
    """
    if type(x) is not str:
        return []
    else:
        return [y.strip() for y in x.split("|")]


def extract_keywords_from_abstract(max_features, n_keywords, df, col_name="dc:description:tokenized") -> pd.Series:
    """
    Extracting keywords using TF-IDF

    :param max_features: max number of vectorized words
    :param n_keywords: number of keywords to extract
    :param df: dataframe, must contain tokenized documents
    :param col_name: name of the column with tokenized abstracts
    :return: 
    """
    nan = df[col_name].isna()
    documents = df[col_name][~nan].tolist()
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(documents)
    x = X.todense()

    log.info(f"Memory: {x.nbytes / 1024 / 1024} MB")

    # top extracted keywords
    top_keywords_index = np.argsort(x, axis=1)[:, ::-1][:, :n_keywords]
    doc_keywords = []
    remap = {index: k for k, index in vectorizer.vocabulary_.items()}
    for n, values in enumerate(top_keywords_index):
        doc = []
        if documents[n] != "":
            for i in values.tolist()[0]:
                doc.append(str(remap[i]))
        doc_keywords.append(doc)

    all_keywords = vectorizer.vocabulary_.keys()
    doc_keywords = pd.Series(doc_keywords, index=df[~nan].index)
    return doc_keywords, all_keywords


@utils.timed
def tokenize_abstracts_spacy(df):
    tokenizer = SpacyTokenizer()
    nan = df[ABSTRACTS].isna()
    abstracts = df[ABSTRACTS][~nan].tolist()

    tokenized = tokenizer.tokenize(abstracts)

    return pd.Series(tokenized, index=df.index[~nan])


@utils.timed
def extract_adjectives_spacy(df):
    tokenizer = SpacyTokenizer()
    nan = df[ABSTRACTS].isna()
    abstracts = df[ABSTRACTS][~nan].tolist()
    results = tokenizer.adjectives(abstracts)
    return pd.Series(results, index=df.index[~nan])


def cast_dates(df):
    df["prism:coverDate_D"] = df["prism:coverDate"].round("D")


def extract_basic_affiliation_info(df):
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


@utils.timed
def extract_basic_author_info(df):
    """

    """
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


@utils.timed
def merge_dataframes(paths) -> pd.DataFrame:
    """
    Create initial dataframe from individual downloaded dataframes. Duplicate dc:identifiers are being dropped.
    
    :param paths: path to pickled dataframe files
    :return:
    """
    df = pd.DataFrame()
    n_rows_dropped = 0
    n_files_dropped = 0

    for file in paths:
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


@utils.timed
def extract_author_info(df, all_authors):
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


@utils.timed
def extract_affiliation_info(df, all_affiliations):
    """
    Counting occurences here is much faster then grouping the dataframe.
    """
    affil_map = OrderedDict()

    drop = df[AFFILIATION_ID].isna()
    log.info(f"Dropping {drop.sum()}")
    df = df[~drop]
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
    Command line interface for preprocessing
    """
    utils.configure_logging()


@cli.command("spacy")
@click.option("--model", "-m", "model", type=str, default="en_core_web_sm")
@click.option("--no-progress", "disable_progress", type=bool, is_flag=True, default=False)
def preprocess_spacy(model, disable_progress):
    """
    Spacy preprocessing
    """
    log.info("Loading dataframe")
    df = pd.read_pickle(JOURNALS_DF)
    drop = df[ABSTRACTS].isna()
    log.info(f"Dropping {drop.sum()}")
    df = df[~drop]

    nlp = spacy.load(model)

    documents = nlp.pipe(df[ABSTRACTS], batch_size=1)

    for ident, document in tqdm.tqdm(zip(df.index, documents), total=len(df), disable=disable_progress):
        path = join(SPACY_DIR, f"{ident}.spacy")
        # log.info(f"Writing to {path}")
        document.to_disk(path)


@cli.command("affiliations")
def preprocess_affiliations():
    """
    Extract affiliation information
    """
    journals_db = pd.read_pickle(JOURNALS_DF)
    affil_ids = journals_db[AFFILIATION_ID]
    affil_ids = affil_ids[~affil_ids.isna()]
    affil_ids = affil_ids.explode().value_counts().index
    log.info(f"Unique Affiliations: {len(affil_ids)}")

    log.info("Creating Database")
    affil_map = extract_affiliation_info(journals_db, all_affiliations=affil_ids)
    affil_db = pd.DataFrame()
    author_ids = affil_map.keys()
    author_data = affil_map.values()

    affil_db[GEPHI_ID] = pd.Series(author_ids)
    affil_db[GEPHI_LABEL] = pd.Series(k["name"] for k in author_data)  # for gephi
    affil_db["affiliation:country"] = pd.Series(k["country"] for k in author_data)
    affil_db["affiliation:city"] = pd.Series(k["city"] for k in author_data)
    affil_db["affiliation:count"] = pd.Series(k["count"] for k in author_data)

    affil_db.set_index(GEPHI_ID, inplace=True)
    log.info("Saving Database")
    affil_db.to_pickle(AFFILIATIONS_DF)
    log.info("Exporting Nodelist")
    affil_db.to_csv(AFFILIATIONS_DF_CSV, sep=";", quoting=csv.QUOTE_ALL)
    log.info("Done")


@cli.command("authors")
def preprocess_authors():
    """
    Extract author information
    """
    log.info("Loading dataframe")
    journals_db = pd.read_pickle(JOURNALS_DF)

    author_ids = journals_db[AUTHOR_ID]
    author_ids = author_ids[~author_ids.isna()]
    all_authors = author_ids.explode().value_counts().index
    log.info(f"Unique Authors: {len(all_authors)}")

    log.info("Creating Database")
    t1 = time.perf_counter()
    author_map = extract_author_info(journals_db, author_ids.explode().value_counts().index)
    authors_db = pd.DataFrame()

    author_ids = author_map.keys()
    author_data = author_map.values()

    authors_db[GEPHI_ID] = pd.Series(author_ids)
    authors_db[GEPHI_LABEL] = pd.Series(k["name"] for k in author_data)  # for gephi
    authors_db[AUTHOR_NAME] = pd.Series(k["name"] for k in author_data)  # for gephi

    authors_db["author:affiliations"] = pd.Series(k["affiliations"] for k in author_data)
    aff_counts = [Counter(k["affiliations"]) for k in author_data]
    authors_db["author:affiliations:unique"] = pd.Series(list(counter.keys()) for counter in aff_counts)
    authors_db["author:affiliations:counts"] = pd.Series(list(counter.values()) for counter in aff_counts)
    authors_db["author:affiliations:top"] = pd.Series(k["affiliations:top"] for k in author_data)
    authors_db["author:count"] = pd.Series(k["count"] for k in author_data)

    t2 = time.perf_counter()
    log.info(f"Time: {t2 - t1:.2} s")
    authors_db.set_index(GEPHI_ID, inplace=True)
    log.info("Saving Database")
    authors_db.to_pickle(AUTHORS_DF)
    log.info("Exporting Nodelist")
    authors_db.to_csv(AUTHORS_DF_CSV, sep=";", quoting=csv.QUOTE_ALL)

    log.info("Done")


@cli.command("journals")
@click.option("-m", "--max-features", "max_features", type=int, default=5000,
              help="Vocabulary size for TF-IDF (5000)")
@click.option("-k", "--n-keywords", "n_keywords", type=int, default=20,
              help="Number of keywords to extract (20)")
@click.option("-t", "--tokenize", "tokenize", is_flag=True, type=bool, default=False,
              help="Activate tokenization with SpaCy and preprocessing for GenSim. Takes some time.")
@click.option("-p", "--pos-tag", "pos_tag", is_flag=True, type=bool, default=False,
              help="Activate Part of Speech (PoS) Tagging with SpaCy. Takes some time.")
def preprocess_journals(max_features, n_keywords, tokenize, pos_tag):
    """
    Process dataframes of downloaded journals.
    """
    setup_nltk_cache()

    log.info("Loading dataframes")
    files = [join(JOURNALS_DIR, f) for f in listdir(JOURNALS_DIR) if f.endswith(".pkl")]
    df = merge_dataframes(files)

    log.info(f"Documents: {len(df)}")
    log.info(f"Empty Abstracts: {df[ABSTRACTS].isna().sum()}")
    log.info(f"Abstracts: {(~df[ABSTRACTS].isna()).sum()}")

    log.info("Converting dates")
    cast_dates(df)

    log.info("Fixing misspelled journal names")
    fix_journal_names(df)

    log.info("Removing Copyright Claims")
    remove_copyright(df)

    log.info("Extracting Authors")
    process_base_authors(df)

    log.info("Extracting affiliations")
    process_base_affilations(df)

    log.info("Preprocessing Author Keywords")
    df["authkeywords:preprocessed"] = df["authkeywords"].apply(preprocess_provided_keywords)

    if tokenize:
        log.info("Tokenizing abstracts using spacy.")
        tokenized = tokenize_abstracts_spacy(df)
        df["dc:description:tokenized"] = tokenized

        if n_keywords > 0:
            log.info("Extracting keywords")

            doc_keywords, all_keywords = extract_keywords_from_abstract(
                max_features, n_keywords, df)
            df["dc:description:keywords"] = doc_keywords

        # gensim preprocessing
        log.info("Gensim Preprocessing")
        df["dc:description:tokenized:gensim"] = gensim_preprocess(df)

    if pos_tag:
        log.info("Extracting adjectives + nouns from abstracts")
        tokenized = extract_adjectives_spacy(df)
        df["dc:description:framing"] = tokenized

    df.to_csv(JOURNALS_DF_CSV)
    df.to_pickle(JOURNALS_DF)
    log.info("Done.")


if __name__ == "__main__":
    cli()
