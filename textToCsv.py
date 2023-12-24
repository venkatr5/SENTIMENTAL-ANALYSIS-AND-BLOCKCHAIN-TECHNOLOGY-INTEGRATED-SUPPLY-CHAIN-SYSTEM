#this python code converts txt document to csv for machine learning
import re
import pandas as pd
# Specify the path to the file
file_path = "C:/Users/RubanVenkateshD/MBA/Thesis Data/Twitter Data Blockchain.txt"  # Replace with the actual file path
# Initialize a list to store the TwitterTweet values
tweet_texts = []
tweet_ids = []
professions = []
userIDs = []
# Flag to indicate when to capture the TwitterTweet text
capture_tweet = False
# Define the regular expression pattern
pattern1 = r"\d+\.\s+TweetID"
pattern2 = r"TwitterTweet"
pattern3 = r"Profession"
pattern4 = r"UserID"
# Open the file and read the lines
try:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Iterate through each line
    for line in lines:
        # Check if the line matches the pattern
        if re.search(pattern1, line):
            # Extract the TweetID value and store it in the list
            current_tweet_id = line.strip().split(":")[1].strip()
            tweet_ids.append(current_tweet_id)
        # Check if the line matches the pattern
        if re.search(pattern2, line):
            line = line.replace("TwitterTweet : ", "")
            tweet_texts.append(line)
        if re.search(pattern3, line):
            line = line.replace("Profession: ", "")
            professions.append(line)
        if re.search(pattern4, line):
            line = line.replace("UserID : ", "")
            userIDs.append(line)
    for userID in (userIDs):
        print(f"a1. {userID}")
except FileNotFoundError:
    print(f"File not found at path: {file_path}")
except Exception as e:
    print(f"An error occurred: {str(e)}")

# Create a dictionary with the lists as columns
data = {
    "TweetID": tweet_ids,
    "UserID": userIDs,
    "Profession": professions,
    "TweetText": tweet_texts
}

# Create a Pandas DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv("C:/Users/RubanVenkateshD/MBA/Thesis Data/twitter_data1.csv", index=False)




