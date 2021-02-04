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
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Lemmatization
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# stopwords
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))   
    
def preprocessData(data):

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
            return 0
        elif score <= 0.4:
            return 1
        elif score <= 0.6:
            return 2
        elif score <= 0.8:
            return 3
        else:
            return 4

    data['phrase'] = data['phrase'].apply(clean)
    data['tokstem'] = data['phrase'].apply(token_stem_stop)
    data['class'] = data['label'].apply(classDesignation)

    return data

if __name__ == '__main__':
    td = pd.read_csv('data/train.csv')
    testd = pd.read_csv('data/test.csv')
    vd = pd.read_csv('data/val.csv')
    preprocessData(td).to_csv("data/pt_data.csv", encoding='utf-8', index=False)
    preprocessData(testd).to_csv("data/ptest_data.csv", encoding='utf-8', index=False)
    preprocessData(vd).to_csv("data/pval_data.csv", encoding='utf-8', index=False)
