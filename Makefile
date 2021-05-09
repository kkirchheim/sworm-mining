#
all: fetch preprocess clustering

fetch:
	python src/fetch.py journals
	python src/fetch.py documents

preprocess:
	python src/preprocess.py journals
	python src/preprocess.py authors
	python src/preprocess.py affiliations

clustering:
	python src/clusering.py tfidf

docker-build:
	docker build -t docker.kondas.de/sworm-bokeh .

docker-push:
	docker push docker.kondas.de/sworm-bokeh

