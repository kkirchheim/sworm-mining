#
all: journals documents

journals:
	python src/fetch_journals.py

documents:
	python src/fetch_documents.python
