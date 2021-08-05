// Start the Python interpreter from Stata
python

# Loads all of the modules needed in this script
import json
import requests
import pandas as pd
import spacy
from sfi import ValueLabel, Data, SFIToolkit
# Torch is Facebook's Deep Learning Library and it runs reasonably on a CPU only machine
import torch

torch.manual_seed(0)

# For examples using BERT see the jupyter notebook in the repository
# This will load the tokenizers and models using the BERT architecture
# from transformers import BertTokenizer, BertModel

# This will initialize the tokenizer and download the pretrained model parameters
# You can also use 'bert-large-cased' if you are using Stata SE or Stata MP.
# 'bert-large-cased' will produce 1,024 dimensional vectors, while 
# 'bert-base-cased' will return only 768 dimensional vectors.  
# If you really need something more expressive, there are other pre-trained models available that will return 
# > 2,000 dimension vectors (e.g., GPTNeo, xlm-mlm-en-2048, alberta-xxlarge-v1/2)
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)

# We'll also load up the model for spaCy at this time
nlp = spacy.load('en_core_web_lg')

# List of the URLs containing the data set
files = [ "https://raw.githubusercontent.com/DenisPeskov/2020_acl_diplomacy/master/data/test.jsonl",
"https://raw.githubusercontent.com/DenisPeskov/2020_acl_diplomacy/master/data/train.jsonl",
"https://raw.githubusercontent.com/DenisPeskov/2020_acl_diplomacy/master/data/validation.jsonl" ]

# Function to handle dropping "variables" that prevent pandas from
# reading the JSON object
def normalizer(obs: dict, drop: list) -> pd.DataFrame:
    # Loop over the "variables" to drop
    for i in drop:
        # Remove it from the dictionary object
        del obs[i]
    # Returns the Pandas dataframe
    return pd.DataFrame.from_dict(obs)

# Object to store each of the data frames
data = []

# Loop over each of the files from the URLs above
for i in files:
    # Get the raw content from the GitHub location
    content = requests.get(i).content
    # Split the JSON objects by new lines, pass each individual line to json.loads,
    # pass the json.loads value to the normalizer function, and
    # append the result to the data object defined outside of the loop
    # You should be able to add '_ = ' at the start of the next line to suppress the output
    [ data.append(normalizer(json.loads(i), [ "players", "game_id" ])) for i in content.decode('utf-8').splitlines() ]

# Define a couple data mappings for later use
labmap = { True: 1, False: 0, 'NOANNOTATION': -1 }
cntrys = { 'austria': 0, 'england': 1, 'france': 2, 'germany': 3, 'italy': 4, 'russia': 5, 'turkey': 6 }
seasons = { 'Fall': 0, 'Winter': 1, 'Spring': 2 }

# Combine each of the data frames for each game into one large dataset
dataset = pd.concat(data, axis = 0, join = 'inner', ignore_index = True, sort = False)

# Change data types of a couple columns
dataset['game_score'] = dataset['game_score'].astype('int')
dataset['sender_labels'] = dataset['sender_labels'].astype('int')
dataset['absolute_message_index'] = dataset['absolute_message_index'].astype('int')
dataset['relative_message_index'] = dataset['relative_message_index'].astype('int')
dataset['game_score_delta'] = dataset['game_score_delta'].astype('int')
dataset['years'] = dataset['years'].astype('int')

# Recodes text labels to numeric values
dataset.replace({'receiver_labels': labmap, 'speakers': cntrys, 'receivers': cntrys, 'seasons': seasons}, inplace = True)

# Creates an indicator for when the receiver correctly identifies the truthfulness of the message
dataset['correct'] = (dataset['sender_labels'] == dataset['receiver_labels']).astype('int')

# Get the number of tokens per message using spaCy's tokenizer
dataset['tokens'] = dataset['messages'].apply(lambda x: len(nlp(x)))

# Expand's data by token
dataset['token'] = dataset['messages'].apply(lambda x: nlp(x))

# Now the data set can be expanded by unique tokens
dataset = dataset.explode('token')

# Make sure the token variable is cast as a string
# If you don't do this you'll get an error saying that Stata couldn't store the 
# string value in the current Stata dataset
dataset['token'] = dataset['token'].astype('str')

# Then add ID's for each token (these values should also use zero-based indexing)
dataset['tokenid'] = dataset.groupby('messages').cumcount()

# Get the names of the variables
varnms = dataset.columns

# Sets the number of observations based on the messages column
Data.setObsTotal(len(dataset['messages']))

# Create the variables in Stata
for var in varnms:
    if var not in [ 'messages', 'token' ]:
        Data.addVarLong(var)
    else:
        Data.addVarStrL(var)

# Now push the data into Stata
Data.store(var = None, obs = None, val = dataset.values.tolist())

# Create mapping of value labels to variables
vallabmap = { 'sender_labels' : labmap, 'receiver_labels': labmap, 'seasons': seasons, 'speakers': cntrys, 'receivers': cntrys }

# Loop over the dictionary containing the value label mappings
for varnm, vallabs in vallabmap.items():

    # Start the string that defines the value labels
    ValueLabel.createLabel(varnm)
    
    # Now iterate over the value label mappings
    # Again if you want to suppress the output add '_ = ' at the start of the next line
    [ ValueLabel.setLabelValue(varnm, value, str(label)) for label, value in vallabs.items() ]

    # Now this string can be used to define the value labels in Stata
    ValueLabel.setVarValueLabel(varnm, varnm)
    
# Since we know the length of the vector in advance, we can create all of the 
# variables that we want, so we'll create variables for individual word vectors
# Again if you want to suppress the output add '_ = ' at the start of the next line
[ Data.addVarDouble('wembed' + str(i)) for i in range(1, 301) ]

# Gets all of the messages
for ob, token in enumerate(dataset['token'].tolist()):
    # Gets the spaCy embeddings
    embed = nlp(token)
    # Stores the word vector for this word
    # Again if you want to suppress the output add '_ = ' at the start of the next line
    [ Data.storeAt("wembed" + str(dim + 1), ob, embed.vector[dim]) for dim in range(0, len(embed.vector)) ]

# You can now fit a model to the data:
SFIToolkit.stata("logit correct i.speakers i.seasons i.years i.game_score wembed1-wembed300")

# These results are fairly noisy, so maybe there would be better luck using document vectors
SFIToolkit.stata("drop token tokenid wembed*")
SFIToolkit.stata("duplicates drop")

# Now use the same process used above, but using document vectors
[ Data.addVarDouble('docembed' + str(i)) for i in range(1, 301) ]
for ob, token in enumerate(dataset['messages'].tolist()):
    # Gets the spaCy embeddings
    embed = nlp(token)
    # Stores the word vector for this word
    # Again if you want to suppress the output add '_ = ' at the start of the next line
    [ Data.storeAt("docembed" + str(dim + 1), ob, embed.vector[dim]) for dim in range(0, len(embed.vector)) ]

# This model fits the data a bit better than the previous model and is also 
# noticably faster.
SFIToolkit.stata("logit correct i.speakers i.seasons i.years i.game_score docembed1-docembed300")

end

// Now you can start using the embeddings for additional models
