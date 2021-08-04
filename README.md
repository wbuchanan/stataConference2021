# Beyond n-grams, tf-idf, and word indicators for text: Leveraging the Python API for vector embeddings
This talk will share strategies that Stata users can use to get more informative word, sentence, and document vector embeddings of text in their data. While indicator and bag-of-words strategies can be useful for some types of text analytics, they lack the richness of the semantic relationships between words that provide meaning and structure to language. Vector space embeddings attempt to preserve these relationships and in doing so can provide more robust numerical representations of text data that can be used for subsequent analysis. I will share strategies for using existing tools from the Python ecosystem with Stata to leverage the advances in NLP in your Stata workflow.

# Prep
If you want to be able to follow along with some of the examples from the talk, you'll need some Python packages
installed on your machine.  You can use either of the code snippets below to install the necessary packages.
I happen to use Anaconda, but if you use pip the result should be the same.able

## Package installation
Feel free to use pip or conda depending on how your Python instance is set up/installed.

```
## Installing spaCy using pip
$ pip install -U pip setuptools wheel
# Use this line if you have no intention to train models
$ pip install -U spacy
# Or to install using conda:
$ conda install -c conda-forge spacy
# Use this line instead if you want to be able to train models
$ pip install -U spacy[transformers,lookups]
# If you want to add CUDA support, add it as an option like this:
# where the ### following cuda is the version number (e.g., 102 = CUDA 10.2)
$ pip install -U spacy[cuda111,transformers,lookups]
# This is necessary before using spaCy and downloads the pretrained model
$ python -m spacy download en_core_web_sm
# For the accuracy optimized pre-trained model use this line instead
$ python -m spacy download en_core_web_trf

## Installing transformers
# If TensorFlow 2.0 and/or PyTorch are already installed
$ pip install transformers
# For CPU support via PyTorch:
$ pip install transformers[torch]
# For CPU support via TensorFlow
$ pip install transformers[tf-cpu]
# To install with Flax
$ pip install transformers[flax]
# To install via conda
$ conda install -c huggingface transformers

# I'd also recommend installing another library to make it
# a bit easier to work with transformer based models
pip install simplerepresentations
```

## Initial Loading/Config
Doing this at the start of your script will front load a lot of the stuff that takes a bit of time to load.  One reason is that the spaCy model is a bit large and needs to get loaded in memory.  The other reason is that transformers will download the model that you need at run time, so if you do this now it should be relatively smooth sailing moving forward.  

_Note: If you are following along using the Jupyter Notebook, be aware that some of the tasks are extremely memory intensive and running the notebook locally could cause your computer to start swapping memory.  If you don't have a lot of RAM available, you should be fine following along using the Stata script which doesn't have the same memory overhead as the notebook._

```
import spacy
import torch
torch.manual_seed(0)
import pandas as pd
import json
import requests
from transformers import BertTokenizer, BertModel
from simplerepresentations import RepresentationModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
nlp = spacy.load('en_core_web_lg')
```

This step will take a little bit to complete as the pre-trained model gets downloaded, but will take care of some of the overhead that isn't directly related to any computational tasks discussed in the talk.
