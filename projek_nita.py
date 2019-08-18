# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

from googletrans import Translator
translator = Translator()

import tweepy
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud
import re
import numpy as np
import pandas as pd
import seaborn as sns

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from string import punctuation
#from tweepy import Stream
#from tweepy import OAuthHandler
#from tweepy.streaming import StreamListener


#akses data di twitter
ACCESS_TOKEN = '551448985-IfSf4ktX1NM9qvYSKmkKk6GnxpAXX72fvtROkvoP'
ACCESS_SECRET = 'uIkCP1bNucQlVG96w8smCmIgRvFCSP5pJypw5cr5ThITU'
CONSUMER_KEY = 'nSpjUQp1ibRvs3Sm9icezmUvv'
CONSUMER_SECRET = 'sCE7CJKRlXON4PL1TNnIBg1U7RktTvpH6VcIX6gfYEFpAxlOL8'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)

tweets = api.user_timeline('@JeniusConnect', count=500, tweet_mode='extended')
for t in tweets:
    print(t.full_text)
    print()

def list_tweets(user_id, count, prt=False):
    tweets = api.user_timeline(
        "@" + user_id, count=count, tweet_mode='extended')
    tw = []
    for t in tweets:
        tw.append(t.full_text)
        if prt:
            print(t.full_text)
            print()
    return tw

#cleaning data    
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"ya", " ", text)
    text = re.sub(r"di", "  ", text)
    text = re.sub(r"ini", " ", text)
    text = re.sub(r"ada", " ", text)
    text = re.sub(r"dan", " ", text)
    text = re.sub(r"yang", " ", text)
    text = re.sub(r"kamu", " ", text)
    text = _removeNonAscii(text)
    text = text.lower()
    return text

def clean_lst(lst):
    lst_baru = []
    for r in lst:
        lst_baru.append(clean_text(r))
    return lst_baru

def clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
    return lst
   

#text processing
stop_words = []
f = open('D:\Digital Talent Kominfo 2019\Statistika\Projek\stopwords (2).txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))
    
additional_stop_words = ['t', 'will']
stop_words += additional_stop_words

print(len(stop_words))

#score sentimen analisis
def sentiment_analyzer_scores(text, engl=True):
    if engl:
        trans = text
    else:
        trans = translator.translate(text).text
    score = analyser.polarity_scores(trans)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1

#sentiment_analyzer_scores(tw_namair[16])
def anl_tweets(lst, title='Tweets Sentiment', engl=True ):
    sents = []
    for tw in lst:
        try:
            st = sentiment_analyzer_scores(tw, engl)
            sents.append(st)
        except:
            sents.append(0)
    ax = sns.distplot(sents,kde=False,bins=3)
    ax.set(xlabel='Negative                 Neutral               Positive',
           ylabel='#Tweets',
          title="Tweets of @"+title)
    return sents

#wordcloud
def word_cloud(wd_list):
    stopwords = stop_words +list('stopwords')
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=21,
        colormap='jet',
        max_words=50,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear");

user_id = 'JeniusConnect'
count=500

data_jenius = {"raw" :pd.Series(list_tweets(user_id, count, True))}
tw_jenius = pd.DataFrame(data_jenius)
tw_jenius['raw'][3]

tw_jenius['clean_text']= clean_lst(tw_jenius['raw'])
tw_jenius['clean_text'][1]

tw_jenius['clean_vector']=clean_tweets(tw_jenius['clean_text'])
tw_jenius['clean_vector'][1]

sentiment_analyzer_scores(tw_jenius['clean_text'][3], True)

tw_jenius['sentiment']= pd.Series(anl_tweets(tw_jenius['clean_vector'], user_id, True))


#tw_jenius = list_tweets(user_id, count)
#tw_jenius = clean_tweets(tw_jenius)
##tw_jenius = clean_text(tw_jenius)
#tw_jenius_sent = anl_tweets(tw_jenius, user_id)
 
word_cloud(tw_jenius['clean_vector'])
word_cloud(tw_jenius['clean_vector'][tw_jenius['sentiment'] == 1])
word_cloud(tw_jenius['clean_vector'][tw_jenius['sentiment'] == -1])

 
