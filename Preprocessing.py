import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
#turkish stop words
stops = set(stopwords.words('turkish'))

def clean_data(data):
    #convert text values to lowercase
    data["turku_text"] = data["turku_text"].str.lower()

    #remove punctuations 
    data["turku_text"] = data['turku_text'].str.replace('[^\w\s]','')

    #remove new lines
    data['turku_text']=data['turku_text'].replace('\n',' ',regex=True)

    #remove digits from data frames

    data['turku_text'] = data['turku_text'].str.replace('\d+', '')

    return data
    #remove non standard characters and stop words from data

def text_to_wordlist(text, remove_stopwords=False, return_list=False):

    exc_letters_pattern = '[^a-zçğışöü]'
    # replace special letters
    special_letters = {'î':'i', 'â': 'a'}
    
    for sp_let, tr_let in special_letters.items():
        text = re.sub(sp_let, tr_let, text)
        
    #remove non-letters
    text = re.sub(exc_letters_pattern, ' ', text)
    
    #split
    wordlist = text.split()
    
    #remove stopwords
    if remove_stopwords:
        wordlist = [w for w in wordlist if w not in stops]
        
    if return_list:
        return wordlist
    else:
        return ' '.join(wordlist)
