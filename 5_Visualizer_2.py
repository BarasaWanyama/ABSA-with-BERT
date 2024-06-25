
import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load dataset 
df= pd.read_csv('output.csv')


#Function to visualize sentiments
def visualize_sentiment(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot for sentiment scores
    sns.barplot(x='Aspect', y='Sentiment_score', data=df, palette='viridis', ax=ax1)
    ax1.set_title('Sentiment Scores by Aspect')
    ax1.set_xlabel('Aspect')
    ax1.set_ylabel('Sentiment Score')
    ax1.set_ylim(-1, 1)
    ax1.axhline(0, color='grey', linewidth=0.5)

    
    # Pie chart for positive vs negative sentiment counts
    df['sentiment'] = df['Sentiment_score'].apply(lambda x: 'positive' if x > 0 else 'negative')
    sentiment_counts = df['sentiment'].value_counts()
    ax2 = plt.subplot(1, 2, 2)
    ax2.pie(sentiment_counts, colors=['#66b3ff', '#ff6666'], labels=sentiment_counts.index, autopct='%1.1f%%')
    ax2.set_title('Distribution of Positive and Negative Sentiments')
    
    plt.tight_layout()
    plt.show()


    sentiment_counts.plot.pie(autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'], ax=ax2, labels=sentiment_counts.index)
    ax2.set_title('Distribution of Positive and Negative Sentiments')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



#Function to remove special characters fron text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits, but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

#Function to visalize aspects in word cloud
def word_cloud(df):
    # Clean the aspects
    aspects = df['Aspect'].astype(str).apply(clean_text)
    
    # Count the cleaned aspects
    aspect_counts = aspects.value_counts()
    
    # Create a dictionary for WordCloud
    word_freq = {aspect: count for aspect, count in aspect_counts.items()}
    
    # Create word cloud
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, collocations=False).generate_from_frequencies(word_freq)

    # Sort frequencies for annotation
    sorted_frequencies = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    max_words = 5  # Adjust the number of words to annotate

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')

    # Add annotations for top words
    for i, (word, freq) in enumerate(sorted_frequencies[:max_words]):
        plt.annotate(f"{word}: {freq}", 
                     xy=(0.7, 1 - (i + 1) * 0.1),
                     xycoords='axes fraction',
                     fontsize=12, 
                     color='red')

    plt.axis('off')
    plt.title("Word Cloud of Aspects")
    plt.tight_layout(pad=0)
    plt.show()

    # Print the top words and their frequencies for verification
    print("Top aspects and their frequencies:")
    for word, freq in sorted_frequencies[:10]:
        print(f"{word}: {freq}")
    # Print the top words and their frequencies for verification
    print("Top words and their frequencies:")
    for word, freq in sorted_frequencies[:10]:
        print(f"{word}: {freq}")    # Print the top words and their frequencies for verification
    
    print("Sample of original aspects:")
    print(df['Aspect'].head(10))

    print("\nSample of cleaned aspects:")
    print(df['Aspect'].astype(str).apply(clean_text).head(10))

#Visualize the results
word_cloud(df)
visualize_sentiment(df)
