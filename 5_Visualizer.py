import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

# Load dataset 
df= pd.read_csv('output.csv')

# Function to visualize word cloud
def word_cloud(df):

    # Combine all text data into a single string
    text = ' '.join(df['Aspect'].astype(str))

    # Create a word cloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Prevalent Product Aspects')
    plt.axis('off')
    plt.show()




# Function to visualize sentiment scores by aspect
def visualize_sentiment(df):
    # Bar plot for sentiment scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Aspect', y='Sentiment_score', data=df, palette='viridis')
    plt.title('Sentiment Scores by Aspect')
    plt.xlabel('Aspect')
    plt.ylabel('Sentiment Score')
    plt.ylim(-1, 1)  # Assuming sentiment scores range from -1 (negative) to 1 (positive)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.show()

    # Pie chart for positive vs negative sentiment counts
    df['sentiment'] = df['Sentiment_score'].apply(lambda x: 'positive' if x > 0 else 'negative')
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 8))
    sentiment_counts.plot.pie(autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'])
    plt.title('Distribution of Positive and Negative Sentiments')
    plt.ylabel('')
    plt.show()

# Visualize the results
visualize_sentiment(df)
word_cloud(df)
