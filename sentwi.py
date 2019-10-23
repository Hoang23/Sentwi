from flask import Flask, request, render_template, send_file
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

# if you want the image to display in a page and not just by itself, - https://www.reddit.com/r/flask/comments/3uwv6a/af_how_do_i_use_flaskpython_to_create_and_display/

# construct twitter client
client = twitterClient.twitterClient()


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')



# get vs post
# get - in flask, functions assume get requests unless explicitly stated
# post - post to a database usually, also doesnt show in url and history

# # @app.route("/post_field", methods=["POST"]) use decorator if we want to see an output


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


def getTweets():
        
        today = DT.date.today()
        week_ago = today - DT.timedelta(days=7)
        count = 0
        tweets = []
        try:
            for tweet in tweepy.Cursor(client.search,q= input() ,count=100, lang="en", since = week_ago).items():
                tweets.append(tweet)
                count = count + 1
                if count == 1000:
                    time.sleep(2)
                    count = 0
            
        except:
            pass
        
        finally:
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


# @app.route('/inputCalculation')
# def graph():
#     return render_template("result.html")


# #inputCalculation
# #@app.route('/inputCalculation', methods=["GET"])  or @app.route('/plot.png')
# @app.route('/inputCalculation') #/inputCalculation
# def plot_png():
#     fig = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     #Response(output.getvalue(), mimetype='image/png')
#     return Response(output.getvalue(), mimetype='image/png')




@app.route('/inputCalculation', methods=["GET"])
def create_figure():
    fig = Figure()
    series = pd.DataFrame(processAndVader(), columns=['date', 'sentiment'])
    # tell pandas that the date column is the one we use for indexing (or x-axis)
    series.set_index('date', inplace=True)
    # pandas makes a guess at the type of the columns, but to make sure it doesn't get it wrong, we set the sentiment
    # column to floats
    series[['sentiment']] = series[['sentiment']].apply(pd.to_numeric)

    # This step is not necessary, but pandas has a neat function that allows us to group the series at different
    # resultion.  The 'how=' part tells it how to group the instances.  In this example, it sames we want to group
    # by day, and add up all the sentiment scores for the same day and create a new time series called 'newSeries'
    # with this day resolution
    # play with this for different resolution, '1H' is by hour, '1M' is by minute etc 
    
    # Deprecated??: newSeries = series.resample('1D', how='sum') # default (1D) # 1M looks most informative? 1M is 1 month not minute??
    newSeries = series.resample('1D').sum()
    # this plots and shows the time series
    newSeries.plot()
    plt.suptitle('Sentiment Analysis for ' + str(input()))
    plt.ylabel('Sentiment - Negative < 0 > Positive')
    plt.xlabel('Time')
    
    fig = plt.gcf() # get current figure
    #fig.savefig('fig1.png')
    #return fig

    f = tempfile.TemporaryFile()
    plt.savefig(f)    
    img64 = base64.b64encode(f.read()).decode('UTF-8')
    f.close()
    return render_template('home.html', image_data=img64) # 

    #return fig




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