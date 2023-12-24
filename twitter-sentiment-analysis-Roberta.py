import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords data
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load your CSV file into a DataFrame
df = pd.read_csv("C:/Users/RubanVenkateshD/MBA/Thesis Data/twitter_data.csv")

# Initialize the sentiment analysis model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

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

    # Encode the processed tweet
    encoded_tweet = tokenizer(tweet_processed, return_tensors='pt')
    output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Extract sentiment scores
    negative, neutral, positive = scores

    # Append sentiment scores to the respective lists
    negative_scores.append(negative)
    neutral_scores.append(neutral)
    positive_scores.append(positive)

# Add new columns for sentiment scores to the DataFrame
df["Negative"] = negative_scores
df["Neutral"] = neutral_scores
df["Positive"] = positive_scores

# Save the DataFrame with sentiment scores to a new CSV file
df.to_csv("C:/Users/RubanVenkateshD/MBA/Thesis Data/twitter-sentiment-analysis-Roberta.csv", index=False)
