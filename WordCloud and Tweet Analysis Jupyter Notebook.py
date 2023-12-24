#!/usr/bin/env python
# coding: utf-8

# In[2]:


# disable warning
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

df = pd.read_csv('C:/Users/RubanVenkateshD/MBA/Thesis Data/Roberta graphs.csv', header=None, index_col=[0])
df = df[[1,2]].reset_index(drop=True)
df.columns = ['text', 'sentiment']
df.head()


# In[4]:


df = df.drop(0)
df.head()


# In[5]:


import preprocess_kgptalkie as ps

df = ps.get_basic_features(df)


# In[6]:


# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# plot 2x4 grid histogram for each numerical feature
plt.figure(figsize=(20,10))

num_cols = df.select_dtypes(include='number').columns

for col in num_cols:
    plt.subplot(2,4, num_cols.get_loc(col)+1)

    # use sentiment as hue to see the distribution of each numerical feature
    # sns.distplot(df[col], label=col, color='red')
    # sns.histplot(x=col, hue='sentiment', data=df, color='green', bins=100, kde=True)
    sns.kdeplot(data=df, x=col, hue='sentiment', fill=True)


# In[7]:


df.to_csv("C:/Users/RubanVenkateshD/MBA/Thesis Data/Roberta_tweet_analysis.csv", index=False)


# In[8]:


df.head


# In[9]:


df['sentiment'].value_counts().plot(kind='pie', autopct='%1.0f%%')


# word cloud
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)


# In[10]:


# plot 2x2 grid word cloud for each sentiment
plt.figure(figsize=(40,20))

for index, col in enumerate(df['sentiment'].unique()):
    plt.subplot(2,2, index+1)
    # print(col)
    df1 = df[df['sentiment']==col]
    data = df1['text']
    wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=500, max_font_size=40, scale=5).generate(str(data))
    # fig = plt.figure(figsize=(15,15))
    # plt.axis('off')
    # disable ticks
    plt.xticks([])
    plt.yticks([])
    plt.imshow(wordcloud)
    plt.title(col, fontsize=40)
    
plt.show()
plt.tight_layout()


# In[ ]:




