"""

"""
from requests import HTTPError

import src.workarounds
src.workarounds.install()

import yaml
import logging
from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
from os.path import join
import pandas as pd
import sys

import src.utils as utils
import src.constants as const

log = logging.getLogger(__name__)


def fetch_journal_data(client, issn, view="COMPLETE") -> pd.DataFrame():
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


def main():
    config = utils.get_elsa_config()
    client = ElsClient(config['apikey'], local_dir=const.CACHE_DIR)

    with open(const.RESOURCE_JOURNALS, "r") as f:
        journals = yaml.load(f)["Journals"]

    for journal in journals:
        name = journal['name']
        issns = journal['issn']
        for issn in issns:
            log.info(f"Loading publications for journal '{name}' ({issn})")

            try:
                df = fetch_journal_data(client, issn=issn)
                out_path = join(const.JOURNALS_DIR, f"{issn}.pkl")
                log.info(f"Writing to {out_path}")
                df.to_pickle(out_path)

                out_path = join(const.JOURNALS_DIR, f"{issn}.csv")
                log.info(f"Writing to {out_path}")
                df.to_csv(out_path)
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


if __name__ == "__main__":
    utils.configure_logging()
    main()
