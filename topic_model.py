import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from read_data import get_text_for_topic
from clean_data import clean_text_string
from visual_data import visualize_word_cloud

print('Pandas version is : ', pd.__version__)

# import matplotlib.pyplot as plt

name_topics = ["globalization", "mahatma gandhi", "fake news", "women empowerment", "Palliative care", 
"Auschwitz concentration camp"]
# name_topics = ["globalization", "fake news", "hollywood"]

# topic_clean_text=[]
# for index, name_topic in enumerate(name_topics):
#     print(index, name_topic)
#     topic_text = get_text_for_topic(name_topic)
#     topic_clean_text.append(clean_text_string(topic_text))
#     print(topic_clean_text[index][0:100])

# This function can be called to visualise the word clloud of various topics
# visualize_word_cloud(name_topics, topic_clean_text)


# data = pd.DataFrame({'topics':name_topics, 'text':topic_clean_text})
# data.to_pickle("topic_data.pkl")
# print(data['topics'][0])
data = pd.read_pickle("topic_data.pkl")

# the corpus of the text is in data
# here we create a document term matrix: The most common tokenization technique is to 
# break down text into words. We can do this using scikit-learn's CountVectorizer, 
# where every row will represent a different document and every column will represent a different word

################
cv = CountVectorizer(stop_words='english')
# cv = CountVectorizer()
data_cv = cv.fit_transform(data.text)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data.topics
print(len(data_dtm))
# print(data_dtm)

# Find the top 20 (most common) words in each topic
data_dtm = data_dtm.transpose()
top_dict = {}
for c in data_dtm.columns:
    top = data_dtm[c].sort_values(ascending=False).head(20)
    top_dict[c]= list(zip(top.index, top.values))

# print(top_dict)

# Print the top 10 words from each text 
for topic, top_words in top_dict.items():
    print(topic)
    print(', '.join([word for word, count in top_words[0:9]]))
    print('---')

# If there are common words in all the topics which appear many times they can be removed.
# Look at the most common top words --> add them to the stop word list
from collections import Counter

# Let's first pull out the top 10 words for each topic
words = []
for topic in data_dtm.columns:
    top = [word for (word, count) in top_dict[topic]]
    for t in top:
        words.append(t)
        
print(words)

# sentiment analysis 
# Create quick lambda functions to find the polarity and subjectivity of each routine
# Terminal / Anaconda Navigator: conda install -c conda-forge textblob
from textblob import TextBlob


pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['text'].apply(pol)
data['subjectivity'] = data['text'].apply(sub)
print(data['polarity'])
print(data['subjectivity'])
# data
#  Let's plot the results
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 8]

for index, topics in enumerate(data.index):
    x = data.polarity.loc[topics]
    y = data.subjectivity.loc[topics]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data['topics'][index], fontsize=10)
    # plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()


# Sentiment of Routine Over Time
#


#########################
# read data
# clean_data
# explore_data
# TODO: word count scatter plot for all topics in one graph
# TODO: check any more stopwords to be added
# TODO: document term matrix
print('done')



if __name__ == "__main__":
    import doctest
    doctest.testmod()
