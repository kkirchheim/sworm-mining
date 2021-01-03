# Scopus Mining
Scraping data from Elseviers scopus publication database.  

Running on Python 3.7, because gensim is not available for newer versions at the time of writing.

## Setup
```shell script
conda env import --name scopus 
conda activate scopus
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Spacy
You will have to download the english spacy model:
```sh
python -m spacy download en
```

### MALLET
You might have to install the MAchine Learning for LanguagE Toolkit (`mallet`) for LDA. 

```
git clone https://github.com/mimno/Mallet.git
cd Mallet
ant
```

The binary will then be located in `bin/mallet`.

## Usefull links
Documentation:
[Elsevier API doc](https://dev.elsevier.com/technical_documentation.html)

Search for Journal ISSNs: 
[Scimago Journal & Country Rank](https://www.scimagojr.com/journalsearch.php)
