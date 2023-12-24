import pandas as pd
from afinn import Afinn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords data
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load your CSV file into a DataFrame
df = pd.read_csv("C:/Users/RubanVenkateshD/MBA/Thesis Data/twitter_data.csv")

# Initialize the AFINN sentiment analyzer
afinn = Afinn()

# Get the English stopwords
stop_words = set(stopwords.words('english'))

# Initialize lists to store sentiment scores
afinn_scores = []
afinn_negative_scores = []
afinn_neutral_scores = []
afinn_positive_scores = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    tweet = row["TweetText"]
    
    # Tokenize the tweet
    words = word_tokenize(tweet)

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

    # Analyze sentiment using AFINN
    sentiment_score = afinn.score(tweet_processed)

    # Categorize AFINN scores into Negative, Neutral, and Positive
    if sentiment_score < -2:
        sentiment_category = "Negative"
    elif sentiment_score > 2:
        sentiment_category = "Positive"
    else:
        sentiment_category = "Neutral"

    # Append AFINN sentiment scores to the respective lists
    afinn_scores.append(sentiment_score)
    if sentiment_category == "Negative":
        afinn_negative_scores.append(1)
        afinn_neutral_scores.append(0)
        afinn_positive_scores.append(0)
    elif sentiment_category == "Neutral":
        afinn_negative_scores.append(0)
        afinn_neutral_scores.append(1)
        afinn_positive_scores.append(0)
    elif sentiment_category == "Positive":
        afinn_negative_scores.append(0)
        afinn_neutral_scores.append(0)
        afinn_positive_scores.append(1)

# Add new columns for AFINN sentiment scores to the DataFrame
df["AFINN Score"] = afinn_scores
df["Negative AFINN Score"] = afinn_negative_scores
df["Neutral AFINN Score"] = afinn_neutral_scores
df["Positive AFINN Score"] = afinn_positive_scores

# Save the DataFrame with sentiment scores to a new CSV file
df.to_csv("C:/Users/RubanVenkateshD/MBA/Thesis Data/twitter-sentiment-analysis-Affinn.csv", index=False)
