"""
We add data as a module in order to avoid reloading it from disk on each new request.

See: https://discourse.bokeh.org/t/pre-loading-data-in-bokeh-server/4542/2
"""
import logging
from os.path import join, abspath, dirname
import pandas as pd

log = logging.getLogger(__name__)

data_dir = join(abspath(join(dirname(__file__), "..")), "data", "artifacts", "bokeh")
path = join(data_dir, "data.pkl")
log.info(f"Loading data from {path}")
df = pd.read_pickle(path)

path = join(data_dir, "topic-list.pkl")
topic_list = list(pd.read_pickle(path)["topic"])

path = join(data_dir, "journal-list.pkl")
journal_list = list(pd.read_pickle(path)["journal"])

country_list = list(df["country"].unique())