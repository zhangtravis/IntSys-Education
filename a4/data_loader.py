import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchtext import data
import pandas as pd
import re

import nltk
# word tokenization
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Lemmatization
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# Here in this file, you should define functions to try out different encodings.
# Some options:
#   1) Bag of words. You don't need any library for this, it's simple enough to
#       implement on your own.
#   2) Word embeddings. You can use spacy or Word2Vec (or others, but these are good
#       starting points). Spacy will give you better embeddings, since they are 
#       already defined. Word2Vec will create embeddings based on your dataset.

## Document the choices you make, including if you pre-process words/tokens by 
# stemming, using POS (parts of speech info) or anything else. 

## Create your own files to define Logistic regression/ Neural networks to try
# out the performace of different ML algorithms. It'll be up to you to evaluate
# the performance.

class SentimentDataset(Dataset):
    """SentimentDataset [summary]
    
    [extended_summary]
    
    :param path_to_data: Path to dataset directory
    :type path_to_data: str
    """
    def __init__(self, path_to_data, transform_fn=None):
        ## TODO: Initialise the dataset given the path to the dataset directory.
        ## You may want to include other parameters, totally your choice.

        inputData = pd.read_csv(path_to_data)
        self.data = self.preprocessData(inputData)

        self.transform = transform_fn

    def preprocessData(self, data):

        def clean(data):
            data = re.sub('[^A-Za-z" "]+', '',
                            data)  # Removes all special characters and numericals leaving the alphabets and the quotation marks("")
            data = re.sub('[""]+', '', data)  # removes the quotation marks("")
            return data

        def token_stem_stop(string):
            stem_rew = " "
            tokens = word_tokenize(string)  # word tokenization
            for word in tokens:
                if word.lower() not in stop_words:  # removing stop words
                    stem_word = ps.stem(word)  # stemming
                    stem_rew = stem_rew + " " + stem_word
            return str(stem_rew)

        def classDesignation(score):
            if score <= 0.2:
                return 'Very Negative'
            elif score <= 0.4:
                return 'Negative'
            elif score <= 0.6:
                return 'Neutral'
            elif score <= 0.8:
                return 'Positive'
            else:
                return 'Very Positive'

        data['phrase'] = data['phrase'].apply(clean)
        data['tokstem'] = data['phrase'].apply(token_stem_stop)
        data['class'] = data['label'].apply(classDesignation)

    def __len__(self):
        """__len__ [summary]
        
        [extended_summary]
        """
        ## TODO: Returns the length of the dataset.
        return len(self.data)

    def __getitem__(self, index):
        """__getitem__ [summary]
        
        [extended_summary]
        
        :param index: [description]
        :type index: [type]
        """
        ## TODO: This returns only ONE sample from the dataset, for a given index.
        ## The returned sample should be a tuple (x, y) where x is your input 
        ## vector and y is your label
        ## Before returning your sample, you should check if there is a transform
        ## sepcified, and pply that transform to your sample
        # Eg:
        # if self.transform:
        #   sample = self.transform(sample)
        ## Remember to convert the x and y into torch tensors.

        sample = self.data[index]
        # sample = torch.tensor(sample, dtype=torch.float)

        if self.transform:
            sample = self.transform(sample)

        # Return x and y in sample
        return sample[2], sample[1]

def get_data_loaders(path_to_train, path_to_val, path_to_test,
                     batch_size=32, transform_fn=None):
    """
      You know the drill by now.
      """
    # First we create the dataset given the path to the .csv file
    train_dataset = SentimentDataset(path_to_train, transform_fn=transform_fn)
    val_dataset = SentimentDataset(path_to_val, transform_fn=transform_fn)
    test_dataset = SentimentDataset(path_to_test, transform_fn=transform_fn)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Testing purposes
    train_loader, val_loader, test_data = get_data_loaders('data/train.csv', 'data/val.csv', 'data/test.csv')

    for batch_index, (x, y) in enumerate(train_loader):
        print(f"Batch {batch_index}")
        print(f"X: {x}")
        print(f"Y: {y}")

        if batch_index == 1:
            break