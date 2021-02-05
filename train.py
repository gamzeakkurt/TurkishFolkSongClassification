import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from FeatureExtraction import *
from ResamplingData import *
from Preprocessing import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#read csv file
data=pd.read_csv("Data\data.csv")

#data shape
print("Data Shape: ", data.shape)

#head of data
print("Data Heads: ")
print(data.head())

#unique categories
print("Unique data labels: ", data.region.unique())

#find the count of all unique values in index
print("Number of unique labels: ")
print(data['region'].value_counts())

#missing values
print("Number of null number")
print(data.isnull().sum())

#plot orginal data labels
x=data['region'].value_counts().values
plot=sns.barplot(["Sivas","Erzurum","Erzincan"],x)
plot.set(xlabel='Region', ylabel='Number of Data')
plt.show()
#plt.savefig("OrginalNumberOfData.png")

#labels are categorized
#0=Erzincan, 1=Erzurum, 2=Sivas
data['labels'] = pd.factorize(data.region)[0]

#Resample Data
data=Resample(data)
print("After resampling data shape: ",data.shape)

#check label counts
print("Resampling Data: ")
print(data.region.value_counts())

#plot sampling data labels
x=data['region'].value_counts().values
plot=sns.barplot(["Sivas","Erzurum","Erzincan"],x)
plot.set(xlabel='Region', ylabel='Number of Data')
plt.show()
#plt.savefig("SamplingNumberOfData.png")


#Preprocessing

#remove digits,new lines and punctuations.
#Data converts uppercase to lowercase 
data=clean_data(data)

#clean turkish stop words from data   
clean_messages = []
for message in data['turku_text']:
    clean_messages.append(text_to_wordlist(
        message, remove_stopwords=True, return_list=False))


#find frequent words using Bag Of Words method
#get feature names and values
data_features,data_features_name=BagOfWords(clean_messages)
