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
