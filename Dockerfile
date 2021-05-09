FROM python:3.9.4-buster

RUN useradd -ms /bin/bash bokeh
USER bokeh

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

COPY demo ./demo
COPY data/artifacts/bokeh/data.pkl data/artifacts/bokeh/
COPY data/artifacts/bokeh/topic-list.pkl data/artifacts/bokeh/
COPY data/artifacts/bokeh/journal-list.pkl data/artifacts/bokeh/

ENV PATH="${PATH}:/home/bokeh/.local/bin"
CMD [ "bokeh", "serve", "demo"]

