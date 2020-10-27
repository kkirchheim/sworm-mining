"""
Run after fetch_journals
"""
import src.workarounds
src.workarounds.install()

from elsapy.elsclient import ElsClient
from elsapy.elsdoc import AbsDoc
import pandas as pd
import os
from os.path import join
import logging
import json

import src.utils as utils
import src.constants as const

log = logging.getLogger(__name__)


def fetch_document(client, uri) -> dict:
    log.info(f"Fetching {uri}")
    scp_doc = AbsDoc(uri=uri)
    if scp_doc.read(client):
        log.info(f"Title: {scp_doc.title}")
        return scp_doc.data
    else:
        log.error("Read document failed.")
        return None


def fetch_and_write(client, pub, skip_existing=True) -> bool:
    issn = pub["prism:issn"]
    scp_id = pub["dc:identifier"].replace("SCOPUS_ID:", "")
    uri = pub["prism:url"]

    d = join(const.DOCUMENTS_DIR, str(issn))
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
    d = join(const.DOCUMENTS_DIR, str(issn))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{scp_id}.json")
    with open(path, "w") as fi:
        json.dump(data, fi, indent=4)

    return True


def main():
    utils.configure_logging()
    files = os.listdir(const.JOURNALS_DIR)
    files.sort()
    files = [f for f in files if f.endswith(".csv")]
    log.info(f"Found {len(files)} journals")

    files = [join(const.JOURNALS_DIR, f) for f in files]

    config = utils.get_elsa_config()
    client = ElsClient(config['apikey'], local_dir=const.CACHE_DIR)

    df = pd.concat([pd.read_csv(p) for p in files])
    log.info(f"Fetching {len(df)} documents")

    count = 0

    for f in files:
        df = pd.read_csv(f)
        for index, pub in df.iterrows():
            if count > 100:
                break

            fetch_and_write(client, pub, skip_existing=False)
            count += 1


if __name__ == "__main__":
    main()


