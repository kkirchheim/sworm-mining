"""
Constants used in thus project
"""
from os.path import join, dirname, abspath

PROJECT_ROOT = abspath(join(dirname(__file__), ".."))

# DIRECTORIES
RESOURCES_DIR = join(PROJECT_ROOT, "resources")
DATA_DIR = join(PROJECT_ROOT, "data")
LOG_DIR = join(DATA_DIR, "logs")
CACHE_DIR = join(DATA_DIR, "cache")
JOURNALS_DIR = join(DATA_DIR, "journals")
DOCUMENTS_DIR = join(DATA_DIR, "documents")
ARTIFACTS_DIR = join(DATA_DIR, "artifacts")
BOKEH_DIR = join(ARTIFACTS_DIR, "bokeh")
MODELS_DIR = join(DATA_DIR, "models")
SPACY_DIR = join(DATA_DIR, "spacy")

# FILES
CONFIG_FILE = join(DATA_DIR, "config.json")
RESOURCE_JOURNALS = join(RESOURCES_DIR, "journals.yaml")

JOURNALS_DF = join(ARTIFACTS_DIR, "journals.pkl")
JOURNALS_DF_CSV = join(ARTIFACTS_DIR, "journals.csv")

AUTHORS_DF = join(ARTIFACTS_DIR, "authors.pkl")
AUTHORS_DF_CSV = join(ARTIFACTS_DIR, "authors.csv")

AFFILIATIONS_DF = join(ARTIFACTS_DIR, "affiliations.pkl")
AFFILIATIONS_DF_CSV = join(ARTIFACTS_DIR, "affiliations.csv")

TFIDF_EMBEDDING = join(BOKEH_DIR, "X-embedding-tfidf.pkl")
TFIDF_CLUSTERING = join(BOKEH_DIR, "y-pred-tfidf.pkl")

CLEAN_DF_STM = join(ARTIFACTS_DIR, "bereinigtes_df_scopus_13_04_2021.csv")
CLEAN_TOPICS_STM = join(ARTIFACTS_DIR, "theta_stm_scopus_13_04_2021.csv")

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

# list with regular expressions for copyright watermarks.
# most of them could be aggregated into single expressions ...
# however, good enough.
COPYRIGHT_REGEX = [
    r"©, © Taylor  &  Francis Group, LLC.\s*",
    r"© \d\d\d\d, Taylor & Francis Group, LLC.\s*",
    r"© \d\d\d\d, © \d\d\d\d The Author\(s\). Published by Informa UK Limited, trading as Taylor  &  Francis Group.\s*",
    r"© \d\d\d\d, © \d\d\d\d Informa UK Limited, trading as Taylor  &  Francis Group.\s*",
    r"© \d\d\d\d, © \d\d\d\d Informa UK Limited, trading as Taylor  &  Francis Group.\s*",
    r"© \d\d\d\d Informa UK Limited, trading as Taylor\s+&\s+Francis Group\.\s*",
    r"© \d\d\d\d Taylor and Francis Group, LLC\.\s*",
    r"© \d\d\d\d, © \d\d\d\d Taylor  &  Francis\.\s*",
    r"© \d\d\d\d Taylor\s+&\s+Francis\.\s*",
    r"© \d\d\d\d, © \d\d\d\d Taylor\s+&\s+Francis Group, LLC.\s*",
    r"© \d\d\d\d Taylor\s+&\s+Francis Group, LLC.\s*",
    r"© \d\d\d\d, Copyright © Taylor  &  Francis Group, LLC.\s*",
    r"© \d\d\d\d, Taylor  &  Francis Group, LLC.\s*",
    r"© \d\d\d\d Copyright Taylor and Francis Group, LLC.",
    r"© \d\d\d\d, Copyright © Taylor and Francis Group, LLC.\s*",
    r"© Taylor and Francis Group, LLC.\s*",
    r"Copyright © Taylor  &  Francis Group, LLC.\s*",
    r"©, Taylor  &  Francis Group, LLC.\s*",
    r"©Taylor  &  Francis Group, LLC.\s*",
    r"© Taylor  &  Francis Group, LLC.\s*",
    r"Copyright © Taylor\s+&\s+Francis\s*",
    r"© \d\d\d\d Copyright Taylor & Francis",
    r"© Taylor\s+&\s+Francis\s*",

    r"© \d\d\d\d This article is not subject to U\.S\. copyright law\.\s*",
    r"©, This article not subject to U\.S\. copyright law\.\s*",

    r"Published by Oxford University Press on behalf of\.al Workers.\s",
    r"© Oxford University Press \d\d\d\d.\s*",
    r"© \d\d\d\d, Oxford University Press.\s*",
    r"© \d\d\d\d Oxford University Press.\s*",

    # © 1974 by the National Association of Social Workers, Inc.
    r"Copyright \d\d\d\d by the National Association of Social Workers."  # 2 lines
    "© \d\d\d\d by the National Association of Social Workers, Inc.\s*",

    r"©\d\d\d\d, National Association of Social Workers.",
    r"\d\d\d\d National Association of Social Workers\d\d\d\d\s*",  # Typo
    r"© \d\d\d\d by the National Association of Social Workers, Inc.\s*",
    r"© \d\d\d\d[,]* by the National Association of Social Workers, Inc.\s*",
    r"© \d\d\d\d[,]* by the National Association of Social Workers.\s*",
    r"© \d\d\d\d The National Association of Social Workers, Inc.\s*",
    r"© \d\d\d\d the National Association of Social Workers, Inc.\s*",
    r"© \d\d\d\d National Association of SocialWorkers.\s*",  # typo
    r"© © \d\d\d\d[,]* National Association of Social Workers.\s*",
    r"© \d\d\d\d[,]* National Association of Social Workers\.\s*",
    r"© \d\d\d\d[,]* National Association of Social Workers, Inc\.\s*",
    r"\d\d\d\d National Association of Social Workers, Inc\.\s*",
    r"© \d\d\d\d The National Association of Social Workers.",
    r"\d\d\d\d National Association of Social Workers\.\s*",
    r"© National Assotiation of Scial Workers\.\s*",
    r"by the National Association of Social Workers.",
    r"\d\d\d\d National Association of Social Workers.",
    r"© \d\d\d\d by National Association of Social Workers, Inc.",
    r"© \d\d\d\d National Association of Workers.",
    r"© \d\d\d\d, Notional Associotion of Social Workers, Inc.",  # typo
    r"© \d\d\d\d, Notional Association of Social Workers, Inc.",
    r"© National Association of Social Workers, Inc.",

    r"© \d\d\d\d by the Council on Social Work Education, Inc.\s*",
    r"© \d\d\d\d, Council on Social Work Education, Inc.\s*",
    r"© by the council on social work education, Inc.\s*",
    r"© \d\d\d\d, © \d\d\d\d Council on Social Work Education.\s*",
    r"Copyright © \d\d\d\d Council on Social Work Education.\s*",
    r"© Council on Social Work Education.\s*",
    r"Council on Social Work Education, Inc.\s*",

    r"Article copies available for a fee from The Haworth Document.*",
    r"© \d\d\d\d, The Haworth Press, Inc.",
    r"© Copyright \(c\) by The Haworth Press, Inc.\s*",
    r"© \d\d\d\d by the Haworth Press, Inc.\s*",  # ?
    r"© \d\d\d\d by The Hawonh Press, Inc.\s*",  # typo
    r"© \d\d\d\d by The Haworth Press, Inc.\s*",
    r"© \d\d\d\d The Haworth Press, Inc.\s*",
    r"© \d\d\d\d by The Haworth Press,\s*",
    r"© \d\d\d\d, The Haworth Press[.]*\s*",
    r"Copyright © by The Haworth Press, Inc[\.]+\s*",
    r"© \d\d\d\d by The Haworth Press.\s*",
    r"© by The Haworth Press, Inc[\.]+\s*",
    r"© \d\d\d\d, by The Haworth Press, Inc.\s*",
    r"y The Haworth Press, Inc\.\s*",
    r"© \d\d\d\d The Haworth Press, Inc.\s*"


    r"© \d\d\d\d John Wiley\s+&\s+Sons Ltd\s*",
    r"© \d\d\d\d John Wiley\s*",

    r"© \d\d\d\d Sage Publications Los Angeles.\s*",
    r"© \d\d\d\d Sage Publications, Inc.\s*",
    r"© \d\d\d\d, Sage Publications\.\s*",
    r"© \d\d\d\d Sage Publications.\s*",
    r"© \d\d\d\d[,]* SAGE Publications.\s*",
    r"© \d\d\d\d Sage Publications Los Angeles."
    r"© SAGE Publications \d\d\d\d.\s*",
    r"© SAGE Publications.\s*",
    r"Copyright ©\d\d\d\d Sage Publications Los Angeles.\s*",
    r"age Publications Los Angeles, London.\s*",  # typo
    r"age Publications Los Angeles.\s*",  # typo
    r" age Publications London.\s*",  # typo
    r" age Publications.\s*",  # typo

    r"© \d\d\d\d The Author; Journal compilation © \d\d\d\d Blackwell Publishing Ltd.",
    r"© \d\d\d\d Blackwell Publishing Ltd.",
    r"© \d\d\d\d Blackwell Science Ltd.",

    r"Published by Oxford University Press on behalf of The British Association of Social Workers\.\s*",
    r"© \d\d\d\d British Association of Social Workers.\s*",
    r"© The Author\(s\) \d\d\d\d Reprints and permission: .*",
    r"©\s*\d\d\d\d Council on Social Work Education.\s*",
    r"©\s*\d\d\d\d by The University of Chicago.\s*",
    r"© \d\d\d\d, by The University of Chicago.\s*",

    r"©\s*\d\d\d\d Alliance for Strong Families and Communities\.\s*",
    r"Alliance for Strong Families and Communities.\s*",
    r"©\s*\d\d\d\d Alliance for Children and Families.\s*",
    r"Alliance for Children and Families.\s*",

    r"Child\s+&\s+Family Social Work published by John Wiley\s+[&|And]\s+Sons Ltd[\.]*\s*",
    r"Child and Family Social Work published by John Wiley  [&|And]  Sons Ltd.\s*",
    r"John Wiley And Sons Ltd.\s*",
    r"&  Sons Ltd.\s*",
    r"©\s*\d\d\d\d, Western Michigan University. All rights reserved.\s*",
    r"Western Michigan University. All rights reserved.\s*",

    r"© \d\d\d\d Human Sciences Press.",
    r"Human Sciences Press, Inc.\s*",

    r"© \d\d\d\d Singapore General Hospital.\s*",

    r"© \d\d\d\d Springer-Verlag Berlin Heidelberg.\s*",
    r"© \d\d\d\d, Springer Science\+Business Media New York.\s*",
    r"© \d\d\d\d, Springer Science\+Business Media, LLC, part of Springer Nature.\s*",
    r"© \d\d\d\d, Springer Science\+Business Media, LLC.\s*",
    r"© \d\d\d\d Springer Science\+Business Media, LLC.\s*",
    r"© Springer Science\+Business Media, LLC \d\d\d\d.\s*",
    r"© \d\d\d\d Springer Science\+Business Media, Inc.\s*",
    r"© \d\d\d\d Springer Science\+Business Media New York.\s*",

    r"© \d\d\d\d by the Society for Social Work and Research.\s*",

    r"© \d\d\d\d, Social Service Review.\s*",

    r"This is a U.S. Government work and not under copyright protection in the US; "
    "foreign copyright protection may apply. ",

    r"Journal of Social Service Research .*",

    r"© \d\d\d\d The Author\(s\). All rights reserved.\s*",
    r"All rights reserved\.\s*",

    r"© IASSW, ICSW, IFSW \d\d\d\d.\s*",
    r"© \d\d\d\d BMJ Publishing Group, Inc.\s*",

    r"© \d\d\d\d © The Author \d\d\d\d\.\s*",
    r"© \d\d\d\d, © The Author\(s\) \d\d\d\d\.\s*",
    r"© \d\d\d\d, The Author\(s\) \d\d\d\d\.\s*",
    r"© \d\d\d\d The Author \d\d\d\d.\s*",
    r"© \d\d\d\d The Author\(s\) \d\d\d\d\.\s*",
    r"© The Author \d\d\d\d\.\s*",
    r"© \d\d\d\d The Author\.\s*",
    r"© \d\d\d\d The Author\(s\)\.\s*",
    r"© The Author\(s\) \d\d\d\d\.\s*",
    r"© \d\d\d\d, The Author\(s\)\.\s*",

    r"© \d\d\d\d, Group for the Advancement of Psychodynamics and Psychotherapy in Social Work\s*",
    r"© Journal of Sociology and Social Welfare, \d\d\d\d.",

    r"© \d\d\d\d, Australian Association of Social Workers.\s*",
    r"© \d\d\d\d Australian Association of Social Workers.\s*",
    r"© \d\d\d\d[,]* Copyright Australian Association of Social Workers.\s*",

    r"© \d\d\d\d, © \d\d\d\d GAPS.\s*",
    r"© \d\d\d\d Copyright GAPS"
    r"© \d\d\d\d GAPS.\s*",
    r"Copyright © \d\d\d\d..\s*",
    r"Copyright \d\d\d\d\s*",
    r"Copyright \d\d\d\d\s*",
    r" Copyright",
    r" Inc.",
    r"Printed in U\.S\.A\.",

    r"ABSTRACT.\s*",
    r"Summary:\s*",
    r"Objective:\s*",
    r"Objectives:\s*",
    r"Purpose:\s*",
    r"Abstract:\s*",
    r"Group, LLC.\s*",
    r"Copyright © Crown copyright.\s*",
    r"Ó \d\d\d\d\s*",
    r"© \d\d\d\d\s*",
    r"\[\s*",
    r"\]\s*",

    r"^,\s*",  # clean up
    r"^\d\d\d\d[\.,]+\s*",  # clean up
    r"^:\s*",  # clean up
    r"^[\.,]\s*",  # clean up
    r"©\s*[,\.]*\s*",  # clean up
    r" \.$",  # clean up
]
