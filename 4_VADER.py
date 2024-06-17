'''Import useful libraries'''
import pandas as pd
import numpy as np

# import SentimentIntensityAnalyzer class from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

'''Function to print sentiment scores of the reviews.'''
def sentiment_scores(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # Define sentiment dictionary containing pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # compute compound score
    compound_score = sentiment_dict['compound']

    return compound_score

'''Function to print sentiment labels of the reviews.'''
def sentiment(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # Define sentiment dictionary
    sentiment_dict = sid_obj.polarity_scores(sentence)
    
    # Compute compound score
    compound_score = sentiment_dict['compound']

    if compound_score >= 0.05 :
        sentiment_label = "Positive"

    elif compound_score <= - 0.05 :
        sentiment_label = "Negative"

    else :
        sentiment_label = "Neutral"

    return sentiment_label

    
'''Apply pre-defined functions to the dataset'''
df = pd.read_csv("Scraped_reviews.csv")  
df["Sentiment"] = df['Review'].apply(sentiment)
df["Sentiment_score"] = df['Review']. apply(sentiment_scores)


# Display the results
print(df.head())

'''Save the results to a new CSV file'''
df.to_csv("output.csv", index=False)
