#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import glob
import time
import nltk
import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from textblob import TextBlob
from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from dateutil import parser

dataPath = r"C:\Users\domdt\Desktop\MPS Data Analytics\Predictive Analytics\Team Project\Data"

#This is to combine all the input files. The data folder should only contain input files
def dataCombiner(dataPath):
    os.chdir(dataPath)
    csvFileList = [f for f in os.listdir(dataPath) if f.endswith('.csv')]
    concatDF = pd.concat(map(pd.read_csv, csvFileList))
    return concatDF


concatDF = dataCombiner(dataPath)

# Convert all columnst to UTF-8 encoding

concatDF.columns=[col.encode('utf-8', 'replace').decode('utf-8') for col in concatDF.columns]

# Remove NA values from the data frame
def dataCleanserRemoveNA(concatDF):
    #Remove NA values
    concatDF = concatDF. replace(np. nan,'',regex=True)
    print(len(concatDF['id']))
    return concatDF

concatDF = dataCleanserRemoveNA(concatDF)

#combine columns title and body
concatDF['combinedText'] = concatDF['title']+' '+concatDF['body']

# Convert Text to lowercase and remove special characters from the text
def dataCleanserRemoveSpecialCharacters(concatDF):
    # lower case and remove special characters\whitespaces
    concatDF['combinedText'] =concatDF['combinedText'].map(lambda x: x.lower())
    concatDF['combinedText'].replace(r'[^a-zA-Z\s]','',regex=True, inplace = True)
    return concatDF

concatDF = dataCleanserRemoveSpecialCharacters(concatDF)

# This is to get rid of words that we feel have to be removed. To remove more words, simply add to the list "WordListToDelete"
def dataCleanserManualTextRules(concatDF):
    wordListToDelete = ['comment','\n',"post removed","bitcoin","coin","crypto",'deleted','removed']
    for word in wordListToDelete:
        concatDF['combinedText'].replace(word,'',regex=True, inplace = True)
    print(concatDF['combinedText'])
#     concatDF.to_csv('temp.csv')
    return concatDF


concatDF = dataCleanserManualTextRules(concatDF)

# Remove stop words
def dataCleanserRemoveStopWords(concatDF):
    stop_words = nltk.corpus.stopwords.words('english')
    rowList = []
    for row in concatDF['combinedText']:
        temp =row.rstrip()
        temp = temp.strip()
        tokens = nltk.word_tokenize(str(temp))
        filtered_tokens = [token for token in tokens if token not in stop_words]
        doc = ' '.join(filtered_tokens)
        rowList.append(doc)
    concatDF['combinedText'] = rowList
    return concatDF

concatDF = dataCleanserRemoveStopWords(concatDF)

# Lemmatize and stem the data
def dataCleanserLemmatizerAndStemmer(concatDF):
    lemmatizer = WordNetLemmatizer()
    ps = PorterStemmer()
    ls = LancasterStemmer()
    rowList = []
    for row in concatDF['combinedText']:
        tokens = nltk.word_tokenize(str(row))
        tokens = [lemmatizer.lemmatize(i) for i in tokens]
#         tokens = [ps.stem(i) for i in tokens]
#         tokens = [ls.stem(i) for i in tokens]
        # re-create document from filtered tokens
        doc = ' '.join(tokens)
        rowList.append(doc)
    concatDF['combinedText'] = rowList
    return concatDF

concatDF = dataCleanserLemmatizerAndStemmer(concatDF)

# Calculate Sentiment Using TextBlob
def sentimentTextBlob(concatDF):
    polarityList = []
    sentimentList =[]
    for row in concatDF['combinedText']:
        polatityCalculated = TextBlob(row).sentiment.polarity
        polarityList.append(polatityCalculated)
        if polatityCalculated < 0:
            sentimentList.append("Negative")
        elif polatityCalculated == 0:
            sentimentList.append("Neutral")
        else:
            sentimentList.append("Positive")
    concatDF['sentimentTextBlob'] = sentimentList
    concatDF['polarityTextBlob'] = polarityList
    return concatDF

concatDF = sentimentTextBlob(concatDF)

# Calculate Sentiment Using AFINN
def sentimentAFINN(concatDF):
    afn = Afinn(emoticons=True)
    polarityList = []
    sentimentList =[]
    for row in concatDF['combinedText']:
        polatityCalculated = afn.score(row)
        polarityList.append(polatityCalculated)
        if polatityCalculated < 0:
            sentimentList.append("Negative")
        elif polatityCalculated == 0:
            sentimentList.append("Neutral")
        else:
            sentimentList.append("Positive")
    concatDF['sentimentAFINN'] = sentimentList
    concatDF['polarityAFINN'] = polarityList
    return concatDF

concatDF = sentimentAFINN(concatDF)

# Calculate Sentiment Using Vader
def sentimentVader(concatDF):
    analyzer = SentimentIntensityAnalyzer()
    polarityList = []
    sentimentList =[]
    for row in concatDF['combinedText']:
        scores = analyzer.polarity_scores(row)
        polatityCalculated = scores['compound']
        polarityList.append(polatityCalculated)
        if polatityCalculated < 0:
            sentimentList.append("Negative")
        elif polatityCalculated == 0:
            sentimentList.append("Neutral")
        else:
            sentimentList.append("Positive")
    concatDF['sentimentVader'] = sentimentList
    concatDF['polarityVader'] = polarityList
    return concatDF

concatDF = sentimentVader(concatDF)


# In[33]:
concatDF["score"] = concatDF["score"].astype(int)
# concatDF["weightedScore"] = (concatDF["score"] - concatDF["score"].min()) / (concatDF["score"].max() - concatDF["score"].min())

# Create Weighted Score
concatDF["weightedScore"] =concatDF['score']/sum(concatDF['score'])
# concatDF["weightedScore"] = (concatDF["score"] - concatDF["score"].min()) / (concatDF["score"].max() - concatDF["score"].min())

# Created Weighted Polarity for sentiment calculated using different packages
concatDF["weightedpolarityTextBlob"] = concatDF["weightedScore"] * concatDF["polarityTextBlob"]

concatDF["weightedpolarityAFINN"] = concatDF["weightedScore"] * concatDF["polarityAFINN"]

concatDF["weightedpolarityVader"] = concatDF["weightedScore"] * concatDF["polarityVader"]

#Convert format of datetype
concatDF['created'] = pd.to_datetime(concatDF['created'])
concatDF.dropna(subset=['created'], inplace=True)

concatDF['created'] = concatDF['created'].astype(str)

concatDF['created'] =concatDF['created'].map(lambda x: parser.parse(x))

concatDF['created'] = pd.to_datetime(concatDF['created']).dt.date

concatDF.to_csv('Final04202023_v1.csv')






