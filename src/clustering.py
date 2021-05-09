#!/usr/bin/env python
"""
Scripts for calculating embeddings and clustering acticles based on abstracts
"""
import utils
utils.install_elsapy_workarounds()

import click
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

# own imports
import utils as utils
from constants import *

log = logging.getLogger(__name__)


@click.group()
def cli():
    """
    Calculate embeddings and cluster articles based on abstracts
    """
    utils.configure_logging()


@cli.command()
@click.option("--file", "-f", "file", help="Dataframe to use", default=JOURNALS_DF)
@click.option("--topics", "-t", "topics", help="Number of clusters", default=20)
@click.option("--tfidf-dim", "-d1", "tfidf_dim", help="TF-IDF Dimensionality", default=5000)
@click.option("--pca-dim", "-d2", "pca_dimm", help="PCA Dimensionality", default=20)
@click.option("--seed", "-s", "seed", help="Random Seed", default=42)
@click.option("--perplexity", "-p", "perplexity", help="Perplexity for TSNE", default=100)
def tfidf(file, topics=20, tfidf_dim=5000, pca_dimm=20, seed=42, perplexity=100):
    """
    Vectorize texts with TF-IDF, use PCA and T-SNE to reduce dimensionality.

    Based on arxiv-literature-clustering
    https://www.kaggle.com/maksimeren/arxiv-literature-clustering
    """
    log.info(f"Loading")
    df = pd.read_pickle(file)

    todrop = df["dc:description:tokenized"].isna()
    log.info(f"Dropping {todrop.sum()} rows")
    df = df[~todrop]

    log.info(f"Vectorizing")
    vectorizer = TfidfVectorizer(max_features=tfidf_dim)
    X = vectorizer.fit_transform(df["dc:description:tokenized"])

    log.info(f"Clustering")
    kmeans = KMeans(n_clusters=topics, random_state=seed)
    y_pred = kmeans.fit_predict(X)

    log.info(f"Calculating PCA Embedding")
    pca = PCA(n_components=pca_dimm, random_state=seed)
    X_embedded_pca = pca.fit_transform(X.toarray())
    log.info(f"Variance explained by PCA: {sum(pca.explained_variance_ratio_)}")

    log.info(f"Calculating TSNE Embedding")
    tsne = TSNE(verbose=0, perplexity=perplexity, random_state=seed)
    X_embedded_tsne = tsne.fit_transform(X_embedded_pca)

    log.info(f"Saving embeddings to {TFIDF_EMBEDDING}")
    with open(TFIDF_EMBEDDING, "wb") as f:
        pickle.dump(X_embedded_tsne, f)

    log.info(f"Saving clustering to {TFIDF_CLUSTERING}")
    with open(TFIDF_CLUSTERING, "wb") as f:
        pickle.dump(y_pred, f)


if __name__ == '__main__':
    cli()



