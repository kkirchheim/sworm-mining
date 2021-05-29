<h1 style="text-align: center"> Social Work Research Map </h1>

Creating a Social Work Research Map by Mining data from Elseviers scopus publication database.  

This code contains several command line interfaces used to fetch data from scopus,  preprocess files and create models. 

Running on Python 3.8.5, because gensim is not available for newer versions at the time of writing.

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


## Command Line Interfaces  

### fetch.py
Downloads data from the scopus API. 
You will need a configuration file with api credentials in `data/config.json`
```sh
python src/fetch.py journals --help
```

### preprocess.py
Scripts for preprocessing.
```sh
python src/preprocess.py journals --help
```

### model.py

### bokeh_prepare.py

### gephi.py
