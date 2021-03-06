from flask import Flask, request, redirect, render_template, send_file, url_for, jsonify, Response
from flask_sqlalchemy import SQLAlchemy

import subprocess # show commandline output 
#from flask import jsonify # takes any data structure in python and converts it to valid json

import string
import tweepy
from tweepy import Cursor
from tweepy import TweepError 

import tempfile
import json
import sys
import pandas as pd 
import numpy as np
import csv
import datetime as DT
import codecs
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from colorama import Fore, Back, Style
import twitterClient
import TwitterProcessing
import urllib.parse
import io
import time
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os # The StringIO and cStringIO modules are gone. Instead, import the io module and use io.StringIO or io.BytesIO for text and data respectively.
from pandas.plotting import register_matplotlib_converters

# Topic Modelling
import pyLDAvis.sklearn
from wordcloud import WordCloud
from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import math

import sklearn
import unicodedata
import nltk 


client = twitterClient.twitterClient()

app = Flask(__name__)

@app.route('/')
def home():
   
    return render_template('home.html')

@app.route('/topics')
def topicModelling():
    return render_template('topics.html')
     
@app.route('/Vader')
def sentimentAnalysis():
    return render_template('sentimentAnalysis.html')

@app.route('/sentiment', methods=["GET", "POST"])
def clickSentimentButton():
    if request.method == "POST":
        return render_template('sentimentAnalysis.html') 
    # return redirect(url_for(sentimentAnalysis)) this isnt needed?

@app.route('/buttonPress', methods=["GET", "POST"])
def button():
    Button_Pressed = 0 # unused
    #tokens = tokenize()

    if request.method == "POST":
        return render_template('topics.html') # return render_template('topics.html', tokens = tokens) 
    return redirect(url_for(topicModelling))
    #return redirect(url_for('button')) # return url for the button function
    #return render_template('topics.html', tokens = tokens)


def input():
    # request.args.get for get request
    # reqiest.form.get for post reqiest 
    #if request.method == "POST":
    hashtag = request.form.get("hashtag")
    hashtag = str(hashtag)

    checkStartWithHash = hashtag.startswith('#')
    if checkStartWithHash == True:
        pass
    elif checkStartWithHash == False:
        hashtag = "#" + hashtag[0:]

    return hashtag

#@app.route('/inputCalculation', methods=["GET", "POST"]) ### just added #used to be just 'get'
def getTweets():
        today = DT.date.today()
        week_ago = today - DT.timedelta(days=7)
        #month_ago = today - DT.timedelta(days=30)

        count = 0
        tweets = []
        hashtag = input()
        print(f'hashtag is: {hashtag}')
        try:
            for idx, tweet in enumerate(tweepy.Cursor(client.search,q= hashtag ,count=100, lang="en").items()): # lang="en", since = week_ago).items()):
                tweets.append(tweet)
                count = count + 1
                if count == 401:
                    time.sleep(0.1)
                    count = 0
                if idx > 2000: #2500 seems like a good number, also 1600
                    break

            print(len(tweets))
                
            
        except:
            pass
        
        finally:
            print(f"Number of tweets collected is {len(tweets)}")
            return tweets


def vaderSentimentAnalysis(jsonTweets, bPrint, tweetProcessor):
    """
    Use Vader lexicons instead of a raw positive and negative word count.

    @param sTweetsFilename: name of input file containing a json formated tweet dump
    @param bPrint: whether to print the stream of tokens and sentiment.  Uses colorama to highlight sentiment words.
    @param tweetProcessor: TweetProcessing object, used to pre-process each tweet.

    @returns: list of tweets, in the format of [date, sentiment]
    """

    # this is the vader sentiment analyser, part of nltk
    sentAnalyser = SentimentIntensityAnalyzer()


    lSentiment_vader = []
    # open file and process tweets, one by one

    tweetURLs = []
    tokensl = []
    dSentimentScoresl = []

    for tweet in jsonTweets:
        try:
            tweetText = tweet.text
            tweetDate = tweet.created_at

            tweet_id = tweet.id_str
            tweet_screen_name = tweet.user.screen_name
            tweetURLs.append("https://twitter.com/" + str(tweet_screen_name) + "/status/" + tweet_id)
    

            # pre-process the tweet text
            lTokens = tweetProcessor.process(tweetText)

            # this computes the sentiment scores (called polarity score in nltk, but mean same thing essentially)
            # see lab sheet for what dSentimentScores holds
            dSentimentScores = sentAnalyser.polarity_scores(" ".join(lTokens))

            # save the date and sentiment of each tweet (used for time series)
            lSentiment_vader.append([pd.to_datetime(tweetDate), dSentimentScores['compound']])

            tokensl.append(lTokens)
            dSentimentScoresl.append(dSentimentScores)
          
        except KeyError as e:
            pass


    output = list(zip(tokensl, dSentimentScoresl, tweetURLs))


    return lSentiment_vader, tokensl, dSentimentScoresl, tweetURLs, output


def getStopwordsVader():
    from nltk.corpus import stopwords # for some reason this doesnt work as a global import
    stop_words = set(stopwords.words('english'))

    #add words that aren't in the NLTK stopwords list
    new_stopwords = ['https', 'rt', 'via']
    new_stopwords_list = stop_words.union(new_stopwords)

    #remove words that are in NLTK stopwords list
    not_stopwords = {'but'} 
    stopwords = set([word for word in new_stopwords_list if word not in not_stopwords])
    

    # convert english stopwords from set to list and add add punctuation stop words
    stopwords = list(stopwords) 
    return stopwords


def processAndVader():
    stopwords = getStopwordsVader()
    tweets = getTweets()
    tweetProcessor_vader = TwitterProcessing.TwitterProcessing(TweetTokenizer(), stopwords)
    lSentiment_vader = []
    lSentiment_vader, tokensl, dSentimentScoresl, tweetURLs, output = vaderSentimentAnalysis(tweets, True, tweetProcessor_vader)
    return lSentiment_vader, tokensl, dSentimentScoresl, tweetURLs, output, tweets


# processing used in LDA?
def process(text, tokeniser=TweetTokenizer(), stopwords=[]):
    """
    Perform the processing.  We used a a more simple version than week 4, but feel free to experiment.
    @param text: the text (tweet) to process
    @param tokeniser: tokeniser to use.
    @param stopwords: list of stopwords to use.
    @returns: list of (valid) tokens in text
    """

    text = text.lower()
    tokens = tokeniser.tokenize(text)

    return [tok for tok in tokens if tok not in stopwords and not tok.isdigit()]



def runVaderLDAAndCheckToday():
    """
    Runs Vader and LDA algorithm and check the time span of all the tweets
    """
    from nltk.tokenize import TweetTokenizer 
    
    lSentiment_vader, tokensl, dSentimentScoresl, tweetURLs, output, tweets = processAndVader()

    date_time = []
    for i in lSentiment_vader:
        date_time.append(i[0])


    import pytz
    local = pytz.timezone ("Australia/Melbourne") # https://stackoverflow.com/questions/79797/how-to-convert-local-time-string-to-utc

    today = DT.datetime.today()

    today_naive = DT.datetime.strptime (str(today), "%Y-%m-%d %H:%M:%S.%f")
    today_local_dt = local.localize(today_naive, is_dst=None)
    today_utc_dt = today_local_dt.astimezone(pytz.utc)

    today = today_utc_dt.day
    one_day = today_utc_dt - DT.timedelta(days=1)
    one_day = one_day.day
    two_day = today_utc_dt - DT.timedelta(days=2)
    two_day = two_day.day

    allTwoDays = []
    for i in date_time:
        day = i.day
        if day == today:
            allTwoDays.append(True)
        elif day == one_day:
            allTwoDays.append(True)
        elif day == two_day:
            allTwoDays.append(True)
        elif day != today or one_day or two_day:
            allTwoDays.append(False)
        else:
            allTwoDays.append(False)
            
    dfallTwoDays = pd.DataFrame(allTwoDays) 
   
    for index, row in dfallTwoDays.iterrows():
        if row[0] == False:
            allTwoDays = False
        else:
            allTwoDays = True

    
    # if all posts in the last 2 horurs
    now = DT.datetime.today()
    now_naive = DT.datetime.strptime (str(now), "%Y-%m-%d %H:%M:%S.%f")
    now_local_dt = local.localize(now_naive, is_dst=None)
    now_utc_dt = now_local_dt.astimezone(pytz.utc)

    this_hour = now_utc_dt
    this_hour = this_hour.hour

    one_hour = now_utc_dt - DT.timedelta(hours=1)
    one_hour = one_hour.hour

    two_hour = now_utc_dt - DT.timedelta(hours=2)
    two_hour = two_hour.hour
    
    
    # tweets from last few horus
    allLastFewHours = []
    for i in date_time:
        day = i.day
        hour = i.hour
        if hour == this_hour and day == today: 
            allLastFewHours.append(True)
        elif hour == one_hour and day == today: 
            allLastFewHours.append(True)
        elif hour == two_hour and day == today:
            allLastFewHours.append(True)
        else:
            allLastFewHours.append(False)
        
    dfAllLastFewHours = pd.DataFrame(allLastFewHours) 

    for index, row in dfAllLastFewHours.iterrows():
        if row[0] == False:
            allLastFewHours = False
            break
        else:
            allLastFewHours = True


    ################### LDA ################
    from nltk.corpus import stopwords #global imports arent working?
    from nltk.tokenize import TweetTokenizer # global imports arent working?
    # LDA parameters
    # number of topics to discover (default = 10)
    topicNum = 6 # default = 10
    # maximum number of words to display per topic (default = 10)
    # Answer to Exercise 1 (change from 10 to 15)
    wordNumToDisplay = 13 # default was 15
    # this is the number of features/words to used to describe our documents
    # please feel free to change to see effect
    featureNum = 300 # 300 words for features

    punct = list(string.punctuation)
    #stop_words = set(stopwords.words('english'))
    #stopwordList = set(stopwords.words('english'))# + punct + ['rt', 'via', '...', 'https', 'co']
    stopwordList = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"] + punct + ['rt', 'via', '...', 'https', 'co']
    tweetTokenizer = TweetTokenizer()

    lTweets = []
    for tweet in tweets:
        lTokens = process(text=tweet.text, tokeniser=tweetTokenizer, stopwords=stopwordList)
        lTweets.append(' '.join(lTokens))


    tfVectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=featureNum, stop_words= stopwordList) # 'english'
    tf = tfVectorizer.fit_transform(lTweets)
    # extract the names of the features (in our case, the words)
    tfFeatureNames = tfVectorizer.get_feature_names()

    # Run LDA (see documentation about what the arguments means)
    ldaModel = LatentDirichletAllocation(n_components =topicNum, max_iter=10, learning_method='online').fit(tf)
    
    
    return ldaModel, tfFeatureNames, allTwoDays, allLastFewHours, lSentiment_vader, tokensl, dSentimentScoresl, tweetURLs, output


    #return ldaModel, tfFeatureNames
    #return allTwoDays, allLastFewHours, lSentiment_vader, tokensl, dSentimentScoresl, tweetURLs, output




def display_topics(model, featureNames, numTopWords):
    """
    Prints out the most associated words for each feature.

    @param model: lda model.
    @param featureNames: list of strings, representing the list of features/words.
    @param numTopWords: number of words to print per topic.
    """
    topics = []
    words = []
    
    # print out the topic distributions
    for topicId, lTopicDist in enumerate(model.components_):
        topics.append(str("Topic %d:" % (topicId)))
        words.append(str(" ".join([featureNames[i] for i in lTopicDist.argsort()[:-numTopWords - 1:-1]])))
    
    return topics, words


def displayWordcloud(model, featureNames):
    """
    Displays the word cloud of the topic distributions, stored in model.

    @param model: lda model.
    @param featureNames: list of strings, representing the list of features/words.
    """

    # this normalises each row/topic to sum to one
    # use this normalisedComponents to display your wordclouds
    normalisedComponents = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    topicNum = len(model.components_)
    # number of wordclouds for each row
    plotColNum = 3
    # number of wordclouds for each column
    plotRowNum = int(math.ceil(topicNum / plotColNum))

    wordcloudURL = ""
    i = 1
    plt.clf()
    img = io.BytesIO()
    # only the last graph in the loop beccause plt.close() is off
    for topicId, lTopicDist in enumerate(normalisedComponents):
        lWordProb = {featureNames[i] : wordProb for i,wordProb in enumerate(lTopicDist)}
        wordcloud = WordCloud(background_color='black')
        wordcloud.fit_words(frequencies=lWordProb)
        plt.suptitle('Topic Clusters of ' + str(input()))
        plt.subplot(plotRowNum, plotColNum, topicId+1)
        plt.title('Topic %d:' % (topicId+1))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        img = io.BytesIO()
        plt.savefig(img, format='png') #png
        img.seek(0)
        imgURL = base64.b64encode(img.read()).decode('ascii')
        wordcloudURL = 'data:img/png;base64,{}'.format(imgURL)
        #plt.close()
        i = i + 1

    return wordcloudURL


# very good tutorial https://technovechno.com/creating-graphs-in-python-using-matplotlib-flask-framework-pythonanywhere/
#@app.route('/inputCalculation', methods=["GET, POST"]) ### just added # changed get to post

def build_graph(seriesName):
    plt.clf()
    img = io.BytesIO()
    #plt.plot(seriesName)
    # https://stackoverflow.com/questions/31345489/pyplot-change-color-of-line-if-data-is-less-than-zero

    # pos_series = seriesName.copy()
    # neg_series = seriesName.copy()

    # pos_series[pos_series <= 0] = np.nan # pos_series[pos_series <= 0] = np.nan
    # neg_series[neg_series > 0] = np.nan

    # #plt.style.use('fivethirtyeight')
    
    # plt.plot(pos_series, color='b')
    # plt.plot(neg_series, color='r')
    plt.plot(seriesName)
    plt.autoscale(False)   # ignores matplotlib trying to autoscale for anything after this command (therefore the zero line under wont be automatically in the plot)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xticks(rotation=20) #90
    plt.suptitle('Sentiment Analysis of ' + str(input()))
    #plt.suptitle('Sentiment Analysis')
    plt.ylabel('Sentiment - Negative < 0 > Positive')
    #plt.xlabel('Date Time')
    # xlabel doesnt work for some reason
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.read()).decode('ascii') # both ways work
    #graph_url = base64.b64encode(img.getvalue()).decode()  # both ways work
    #plt.close()



    return 'data:image/png;base64,{}'.format(graph_url)

# this function which runs all the other fuctions runs when request method is undertaken
@app.route('/inputCalculation', methods=["POST"]) # tihs belongs here (might need to add get?)
def create_figure():
    if request.method == "POST":
        try:
            import datetime as DT 
            
            register_matplotlib_converters()
            
            ldaModel, tfFeatureNames, today, allLastFewHours, lSentiment_vader, tokensl, dSentimentScoresl, tweetURLs, output = runVaderLDAAndCheckToday()
            
            if today == True and allLastFewHours == True:
                time = "5Min"
            elif today == False:
                time = "1D"
            elif today == True:
                time = "1H"
            
            # just to make sure a time is entered
            else: 
                time = "6H"
            
            series = pd.DataFrame(lSentiment_vader, columns=['date', 'sentiment'])
            series.date = pd.to_datetime(series.date, format='%Y-%m-%d %H:%M:%S', errors='ignore')#, errors='ignore')
            
            # tell pandas that the date column is the one we use for indexing (or x-axis)
            series.set_index('date', inplace=True)
            series[['sentiment']] = series[['sentiment']].apply(pd.to_numeric)
 
            newSeries = series.resample(time).sum()
            graph_url = build_graph(newSeries)

            # LDA words
            topics, words = display_topics(ldaModel, tfFeatureNames, 15)
            TopicModel = list(zip(topics, words))

            # LDA wordcloud
            WordCloudURL = displayWordcloud(ldaModel, tfFeatureNames)

            return render_template('result.html', graph = graph_url, tokensl = tokensl, dSentimentScoresl = dSentimentScoresl, tweetURLs = tweetURLs, output = output, WordCloudURL = WordCloudURL, TopicModel = TopicModel)
     
        except ValueError: 
            return render_template('error.html')
    

if __name__ == '__main__':
    app.run(debug=False)