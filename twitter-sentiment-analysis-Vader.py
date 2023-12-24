import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load your CSV file into a DataFrame
df = pd.read_csv("C:/Users/RubanVenkateshD/MBA/Thesis Data/twitter_data.csv")

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Get the English stopwords
stop_words = set(stopwords.words('english'))

# Initialize lists to store sentiment scores
negative_scores = []
neutral_scores = []
positive_scores = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    tweet = row["TweetText"]
    
    # Tokenize the tweet
    words = nltk.word_tokenize(tweet)

    # Remove stopwords
    tweet_words = [word for word in words if word.lower() not in stop_words]

    for i, word in enumerate(tweet_words):
        if word.startswith('@') and len(word) > 1:
            tweet_words[i] = "@user"
        elif word.startswith('http'):
            tweet_words[i] = "http"
        elif word.startswith('#'):
            tweet_words[i] = word.split('#', 1)[-1].strip()

    tweet_processed = " ".join(tweet_words)

    # Analyze sentiment using VADER
    sentiment = sia.polarity_scores(tweet_processed)

    # Extract VADER sentiment scores
    negative, neutral, positive = sentiment['neg'], sentiment['neu'], sentiment['pos']

    # Append sentiment scores to the respective lists
    negative_scores.append(negative)
    neutral_scores.append(neutral)
    positive_scores.append(positive)

# Add new columns for sentiment scores to the DataFrame
df["Negative (VADER)"] = negative_scores
df["Neutral (VADER)"] = neutral_scores
df["Positive (VADER)"] = positive_scores

# Save the DataFrame with sentiment scores to a new CSV file
df.to_csv("C:/Users/RubanVenkateshD/MBA/Thesis Data/twitter-sentiment-analysis-Vader.csv", index=False)
