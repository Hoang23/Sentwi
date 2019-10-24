from flask import Flask, request, redirect, render_template, send_file
import subprocess # show commandline output 
from flask import jsonify # takes any data structure in python and converts it to valid json
from flask import Response



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
import os


#os.chdir(sys.path[0]) # path[0], is the directory containing the script that was used to invoke the Python interpreter

# https://pythonise.com/feed/flask/python-before-after-request

# if you want the image to display in a page and not just by itself, - https://www.reddit.com/r/flask/comments/3uwv6a/af_how_do_i_use_flaskpython_to_create_and_display/

# construct twitter client
client = twitterClient.twitterClient()


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'images')

# Configure Flask app and the logo upload folder
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER






@app.route('/')
def home():
    return render_template('home.html')



# get vs post
# get - in flask, functions assume get requests unless explicitly stated
# post - post to a database usually, also doesnt show in url and history

# # @app.route("/post_field", methods=["POST"]) use decorator if we want to see an output

#@app.route('/inputCalculation', methods=["POST"])

def input():
    # request.args.get for get request
    # reqiest.forms.get for post reqiest
    
    
    hashtag = request.args.get("hashtag")
    hashtag = str(hashtag)

    

    checkStartWithHash = hashtag.startswith('#')
    if checkStartWithHash == True:
        pass
    elif checkStartWithHash == False:
        hashtag = "#" + hashtag[0:]

    return hashtag

# @app.route("/", methods=["POST"])
# def waitingScreen():
#     hashtag = input()
#     hashtag = f'Calculating sentiment for {hashtag}.'
#     alert = "Please wait a minute for the sentiment analysis. It may take a while depending on the query. Try a different query if so."
#     return render_template("home.html", hashtag = hashtag, alert = alert)


#app.config["IMAGE_UPLOADS"] = "/home/hoang-pc/Sync/Dropbox_insync/Files/Me/Programming/Python/Sentwi/Hosted/static/images"
# Create two constant. They direct to the app root folder and logo upload folder



def getTweets():
        
        today = DT.date.today()
        week_ago = today - DT.timedelta(days=7)
        count = 0
        tweets = []
        try:
            for idx, tweet in enumerate(tweepy.Cursor(client.search,q= input() ,count=100, lang="en", since = week_ago).items()):
                tweets.append(tweet)
                count = count + 1
                if count == 1000:
                    time.sleep(0.2)
                    count = 0
            
        except:
            pass
        
        finally:
            print(count)
            return tweets
    




# @app.route("/post_field", methods=["POST"]) # which ever output is being displayed to, put this decorator
# def display():

#     tweets = input()
#     return render_template('home.html', tweets = tweets)

# def displayAttributes():
#     #tweets = getTweets()
#     hashtag = input()

#     return render_template("home.html", hashtag = hashtag) # tweets = tweets,


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
    for tweet in jsonTweets:
        try:
            tweetText = tweet.text
            tweetDate = tweet.created_at

           
            # pre-process the tweet text
            lTokens = tweetProcessor.process(tweetText)

            # this computes the sentiment scores (called polarity score in nltk, but mean same thing essentially)
            # see lab sheet for what dSentimentScores holds
            dSentimentScores = sentAnalyser.polarity_scores(" ".join(lTokens))

            # save the date and sentiment of each tweet (used for time series)
            lSentiment_vader.append([pd.to_datetime(tweetDate), dSentimentScores['compound']])

            # if we are printing, we print the tokens then the sentiment scores.  Because we don't have the list
            # of positive and negative words, we cannot use colorama to label each token
            if bPrint:
                print(*lTokens, sep=', ')
                for cat,score in dSentimentScores.items():
                    print('{0}: {1}, '.format(cat, score), end='')
                print()

        except KeyError as e:
            pass


    return lSentiment_vader


def stopwords():
    from nltk.corpus import stopwords

    

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
    tweetProcessor_vader = TwitterProcessing.TwitterProcessing(TweetTokenizer(), stopwords())
    lSentiment_vader = []
    lSentiment_vader = vaderSentimentAnalysis(getTweets(), True, tweetProcessor_vader)
    return lSentiment_vader


def checkToday():
    #lSentiment_vader 
    
    lSentiment_vader = processAndVader()

    date_time = []
    for i in lSentiment_vader:
        date_time.append(i[0])
        
    today = DT.date.today()
    one_day = today - DT.timedelta(days=1) 
    one_day = one_day.day
    two_day = today - DT.timedelta(days=2) 
    two_day = two_day.day

    allToday = []
    for i in date_time:
        if i.day == one_day or two_day:
            allToday.append(True)
        else:
            allToday.append(False)
            
    dfAllToday = pd.DataFrame(allToday) 
   
    
    for index, row in dfAllToday.iterrows():
        if row[0] == False:
            return False
        
    return True
        
         


# @app.route('/inputCalculation')
# def graph():
#     return render_template("result.html")


#inputCalculation
#@app.route('/inputCalculation', methods=["GET"])  or @app.route('/plot.png')
# @app.route('/inputCalculation') #/inputCalculation
# def plot_png():
#     fig = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     #Response(output.getvalue(), mimetype='image/png')
#     return Response(output.getvalue(), mimetype='image/png')

def build_graph(x_coordinates, y_coordinates):
    img = io.BytesIO()
    plt.plot(x_coordinates, y_coordinates)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.show()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


@app.route('/inputCalculation', methods=["POST"]) # tihs belongs here
def create_figure():

      
    today = checkToday()
    if today == True:
        time = "1H"
    elif today == False:
        time = "1D"


    #fig = Figure()
    series = pd.DataFrame(processAndVader(), columns=['date', 'sentiment'])
    series.date = pd.to_datetime(series.date, format='%Y-%m-%d %H:%M:%S', errors='ignore')
    
   
    
    
    # tell pandas that the date column is the one we use for indexing (or x-axis)
    series.set_index('date', inplace=True)
    
    #series.set_index(['date'])
    #series.index = pd.to_datetime(series.index, unit='s')
#data.index = pd.to_datetime(data.index, unit='s')
    # pandas makes a guess at the type of the columns, but to make sure it doesn't get it wrong, we set the sentiment
    # column to floats
    series[['sentiment']] = series[['sentiment']].apply(pd.to_numeric)
    newSeries = series.resample('1H').sum()
    
    graph_url = build_graph(newSeries.index, newSeries.values)



    # newSeries.plot()
    # plt.suptitle('Sentiment Analysis')
    # plt.ylabel('Sentiment - Negative < 0 > Positive')
    # plt.xlabel('Time')
    
    #fig = plt.gcf() # get current figure
    
    return render_template('home.html', name = 'new_plot', graph = graph_url)

   

    # f = tempfile.TemporaryFile()
    # plt.savefig(f)    
    # img64 = base64.b64encode(f.read()).decode('UTF-8')
    # f.close()
    # return render_template('home.html', image_data=img64) # 



# @app.route('/inputCalculation')
# def graph():
#     return render_template("result.html")




# @app.route('/inputCalculation')
# def result2():
#     return render_template('result.html')










    # img = io.BytesIO()  # create the buffer
    # plt.savefig(img, format='png')  # save figure to the buffer
    # img.seek(0)  # rewind your buffer
    # plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) # base64 encode & URL-escape
    # return render_template('home.html', plot_url=plot_data) # https://stackoverflow.com/questions/44091516/passing-a-plot-from-matpotlib-to-a-flask-view
    

    #graph = plt.show()
    #return send_file()
    # return render_template("home.html", graph = graph) # https://stackoverflow.com/questions/20107414/passing-a-matplotlib-figure-to-html-flask








if __name__ == '__main__':
    app.run(debug=True)