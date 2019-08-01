import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load climate tweets.

df = pd.read_csv( 'data/climate_tweets.csv' )

# Make a new column to highlight retweets.

df['is_retweet'] = df['tweet'].apply(lambda x: x[:2]=='RT')
df['is_retweet'].sum()  # number of retweets

# Get number of unique retweets.

df.loc[df['is_retweet']].tweet.unique().size

# Find 10 most repeated tweets.

df.groupby(['tweet']).size().reset_index(name='counts')\
  .sort_values('counts', ascending=False).head(10)
  
# Count number of times each tweet appears.

counts = df.groupby(['tweet']).size()\
           .reset_index(name='counts')\
           .counts

# Define bins for histogram of counts.

my_bins = np.arange(0,counts.max()+2, 1)-0.5

# Plot histogram of tweet counts.

plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('copies of each tweet')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()

def find_retweeted(tweet):
    '''Extract the twitter handles of retweeted people'''
    return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_mentioned(tweet):
    '''Extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

def find_hashtags(tweet):
    '''Extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)   
    
# Two sample tweets.

my_tweet = 'RT @our_codingclub: Can @you find #all the #hashtags?'
my_other_tweet = 'Not a retweet. All views @my own'
  
  
# Create new columns for retweeted usernames, mentioned usernames and hashtags.

df['retweeted'] = df.tweet.apply(find_retweeted)
df['mentioned'] = df.tweet.apply(find_mentioned)
df['hashtags'] = df.tweet.apply(find_hashtags)

# Take the rows from the hashtag columns where there are actually hashtags.

hashtags_list_df = df.loc[
                       df.hashtags.apply(
                           lambda hashtags_list: hashtags_list !=[]
                       ),['hashtags']]
                       
# Create dataframe where each use of hashtag gets its own row.

flattened_hashtags_df = pd.DataFrame(
    [hashtag for hashtags_list in hashtags_list_df.hashtags
    for hashtag in hashtags_list],
    columns=['hashtag'])
    
# Number of unique hashtags.

flattened_hashtags_df['hashtag'].unique().size

# Count of appearances of each hashtag.

popular_hashtags = flattened_hashtags_df.groupby('hashtag').size()\
                                        .reset_index(name='counts')\
                                        .sort_values('counts', ascending=False)\
                                        .reset_index(drop=True)
                                        
# Number of times each hashtag appears.

counts = flattened_hashtags_df.groupby(['hashtag']).size()\
                              .reset_index(name='counts')\
                              .counts

# Define bins for histogram.

my_bins = np.arange(0,counts.max()+2, 5)-0.5

# Plot histogram of tweet counts.

plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('Nummber of appearances for hashtags')
plt.ylabel('Frequency')
plt.yscale('log', nonposy='clip')
plt.show()

# Get hashtags which appear at least this many times.

min_appearance = 10

# Find popular hashtags.

popular_hashtags_set = set(popular_hashtags[
                           popular_hashtags.counts>=min_appearance
                           ]['hashtag'])
                           
# Create new column with only the popular hashtags.

hashtags_list_df['popular_hashtags'] = hashtags_list_df.hashtags.apply(
            lambda hashtag_list: [hashtag for hashtag in hashtag_list
                                  if hashtag in popular_hashtags_set])

# Drop rows without any popular hashtag.

popular_hashtags_list_df = hashtags_list_df.loc[
            hashtags_list_df.popular_hashtags.apply(lambda hashtag_list: hashtag_list !=[])]
            
# Create new dataframe.

hashtag_vector_df = popular_hashtags_list_df.loc[:, ['popular_hashtags']]

# Create columns to encode presence of hashtags.

for hashtag in popular_hashtags_set:
    hashtag_vector_df['{}'.format(hashtag)] = hashtag_vector_df.popular_hashtags.apply(
        lambda hashtag_list: int(hashtag in hashtag_list))

hashtag_matrix = hashtag_vector_df.drop('popular_hashtags', axis=1)
 
# Calculate the correlation matrix.

correlations = hashtag_matrix.corr()

# Plot the correlation matrix.

plt.figure(figsize=(10,10))

sns.heatmap(correlations,
    cmap='RdBu',
    vmin=-1,
    vmax=1,
    square = True,
    cbar_kws={'label':'correlation'})
plt.show()

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')

def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    return tweet

my_stopwords = nltk.corpus.stopwords.words('english')

word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem

my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

# Clean a tweet.

def clean_tweet(tweet, bigrams=False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet

df['clean_tweet'] = df.tweet.apply(clean_tweet)

from sklearn.feature_extraction.text import CountVectorizer

# Create vectorizer object will to transform text to vector form.

vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

# Apply transformation.

tf = vectorizer.fit_transform(df['clean_tweet']).toarray()

# tf_feature_names tells us what word each column in the matric represents.

tf_feature_names = vectorizer.get_feature_names()

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

# Extract ten topics.

number_of_topics = 10

# Uncomment next line to use LDA for topic extraction.
#model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)

# Uncomment next line to use NNMF for topic extraction.
model = NMF(n_components=number_of_topics, random_state=0, alpha=.1, l1_ratio=.5)

model.fit(tf)

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

# Display the top words.
    
num_top_words = 10
display_topics(model, tf_feature_names, num_top_words)
                                                          