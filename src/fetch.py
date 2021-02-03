#!/usr/bin/env python
"""
Scripts for fetching data from the scopus API
"""
import utils
utils.install_elsapy_workarounds()

from requests import HTTPError
import yaml
from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
from elsapy.elsdoc import AbsDoc
from os.path import exists
import sys
import click
import pandas as pd
import os
import logging
import json

# own imports
import utils as utils
from constants import *

log = logging.getLogger(__name__)


def fetch_journal(client, issn, view="COMPLETE") -> pd.DataFrame():
    query = f"ISSN({issn})"
    doc_srch = ElsSearch(query, 'scopus')

    # elsapy does not support setting a custom view.
    # setting view to COMPLETE will also fetch abstracts
    doc_srch._uri = doc_srch._uri + f"&view={view}"

    doc_srch.execute(client, get_all=False)
    log.info(f"doc_srch has {doc_srch.tot_num_res} results.")

    if not doc_srch.hasAllResults():
        doc_srch.execute(client, get_all=True)

    return doc_srch.results_df


def fetch_and_write_journal(client, journal, skip_existing):
    name = journal["name"]
    issns = journal["issn"]
    for issn in issns:

        out_path_pickle = join(JOURNALS_DIR, f"{issn}.pkl")
        out_path_csv = join(JOURNALS_DIR, f"{issn}.csv")

        if skip_existing and (exists(out_path_pickle) or exists(out_path_csv)):
            log.info(f"Skipping {issn}")
            continue

        log.info(f"Loading publications for journal '{name}' ({issn})")

        try:
            df = fetch_journal(client, issn=issn)
            log.info(f"Writing to {out_path_pickle}")
            df.to_pickle(out_path_pickle)
            log.info(f"Writing to {out_path_csv}")
            df.to_csv(out_path_csv)
        except HTTPError as e:
            if "AUTHORIZATION_ERROR" in str(e):
                log.fatal("Authorization Failed.")
                sys.exit(1)
            else:
                log.exception(e)
                sys.exit(2)
        except Exception as e:
            log.exception(e)
            sys.exit(3)


def fetch_document(client, uri) -> dict:
    log.info(f"Fetching {uri}")
    scp_doc = AbsDoc(uri=uri)
    if scp_doc.read(client):
        log.info(f"Title: {scp_doc.title}")
        return scp_doc.data
    else:
        log.error("Read document failed.")
        return None


def fetch_and_write_document(client, pub, skip_existing=True) -> bool:
    issn = pub["prism:issn"]
    scp_id = pub["dc:identifier"].replace("SCOPUS_ID:", "")
    uri = pub["prism:url"]

    d = join(DOCUMENTS_DIR, str(issn))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{scp_id}.json")

    if skip_existing and os.path.exists(path):
        log.info("Exists")
        return True

    data = fetch_document(client, uri)
    if data is None:
        log.error("Could not fetch")
        return False

    issn = pub["prism:issn"]
    d = join(DOCUMENTS_DIR, str(issn))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{scp_id}.json")
    with open(path, "w") as fi:
        json.dump(data, fi, indent=4)

    return True


@click.command("documents")
def fetch_docs():
    """
    Fetch journals
    """
    utils.configure_logging()
    files = os.listdir(JOURNALS_DIR)
    files.sort()
    files = [f for f in files if f.endswith(".csv")]
    log.info(f"Found {len(files)} journals")

    files = [join(JOURNALS_DIR, f) for f in files]

    config = utils.get_elsa_config()
    client = ElsClient(config['apikey'], local_dir=CACHE_DIR)

    df = pd.concat([pd.read_csv(p) for p in files])
    log.info(f"Fetching {len(df)} documents")

    count = 0

    for f in files:
        df = pd.read_csv(f)
        for index, pub in df.iterrows():
            if count > 100:
                break

            fetch_and_write_document(client, pub, skip_existing=False)
            count += 1


@click.group()
def cli():
    """
    Command line interface for downloading from the scopus API
    """
    utils.configure_logging()


@cli.command("journals")
@click.option("--skip-existing/--no-skip-existing", default=True)
@click.option("-i", "--issn", default=None)
def fetch_journals(skip_existing, issn):
    """
    Fetch journals
    """
    config = utils.get_elsa_config()
    client = ElsClient(config['apikey'], local_dir=CACHE_DIR)

    if issn is None:
        with open(RESOURCE_JOURNALS, "r") as f:
            journals = yaml.load(f, Loader=yaml.SafeLoader)["Journals"]
    else:
        journals = [{"name": issn, "issn": [issn]}]

    for journal in journals:
        fetch_and_write_journal(client, journal, skip_existing)


if __name__ == "__main__":
    cli()
