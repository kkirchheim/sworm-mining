"""

"""
from os.path import join, dirname, abspath

PROJECT_ROOT = join(abspath(dirname(__file__)), "..")

# DIRECTORIES
RESOURCES_DIR = join(PROJECT_ROOT, "resources")
DATA_DIR = join(PROJECT_ROOT, "data")
LOG_DIR = join(DATA_DIR, "logs")
CACHE_DIR = join(DATA_DIR, "cache")
JOURNALS_DIR = join(DATA_DIR, "journals")
DOCUMENTS_DIR = join(DATA_DIR, "documents")
ARTIFACTS_DIR = join(DATA_DIR, "artifacts")
MODELS_DIR = join(DATA_DIR, "models")

# FILES
CONFIG_FILE = join(DATA_DIR, "config.json")
RESOURCE_JOURNALS = join(RESOURCES_DIR, "journals.yaml")

JOURNALS_DF = join(ARTIFACTS_DIR, "journals.pkl")
JOURNALS_DF_CSV = join(ARTIFACTS_DIR, "journals.csv")

AUTHORS_DF = join(ARTIFACTS_DIR, "authors.pkl")
AUTHORS_DF_CSV = join(ARTIFACTS_DIR, "authors.csv")

AFFILIATIONS_DF = join(ARTIFACTS_DIR, "affiliations.pkl")
AFFILIATIONS_DF_CSV = join(ARTIFACTS_DIR, "affiliations.csv")

# COLUMN NAMES
AFFILIATION_COUNTRY = "affiliation:country"
AFFILIATION_CITY = "affiliation:city"
AFFILIATION_NAME = "affiliation:name"
AFFILIATION_ID = "affiliation:id"
GEPHI_LABEL = "Label"
GEPHI_ID = "Id"
AUTHOR_AFFIL = "author:affil"
ABSTRACTS = "dc:description"
IDENTIFIER = "dc:identifier"
AUTHOR_NAME = "author:name"
AUTHOR_ID = "author:id"
