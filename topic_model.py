import numpy as np
import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
import pickle
# from gensim import matutils, models
# import scipy.sparse
# import matplotlib.pyplot as plt

from read_data import get_text_for_topic
from clean_data import clean_text_string
from visual_data import visualize_word_cloud
from document_term_matrix import document_term
from common_words import most_common_words
from sentiment_analysis import senti_analyse
from topic_distribution import topic_distrbut
from generate_text import generated_text
from summarization import text_summary

print('Pandas version is : ', pd.__version__)

name_topics = ["globalization", "mahatma gandhi", "fake news", "women empowerment", "Palliative care", 
"Auschwitz concentration camp"]
# name_topics = ["globalization", "fake news", "hollywood"]
# orig_text=[]
# topic_clean_text=[]
# for index, name_topic in enumerate(name_topics):
#     print(index, name_topic)
#     topic_text = get_text_for_topic(name_topic)
#     orig_text.append(topic_text)
#     topic_clean_text.append(clean_text_string(topic_text))
#     print(topic_clean_text[index][0:100])

# This function can be called to visualise the word cloud of various topics
# visualize_word_cloud(name_topics, topic_clean_text)

# data_orig = pd.DataFrame({'topics':name_topics, 'text':orig_text})
# data_orig.to_pickle("data_original.pkl")

data_orig=pd.read_pickle('data_original.pkl')

# data = pd.DataFrame({'topics':name_topics, 'text':topic_clean_text})
# data.to_pickle("topic_data.pkl")
# print(data['topics'][0])

# the corpus of the text is in data
data_corpus = pd.read_pickle("topic_data.pkl")

################
# The document term matrix is generated in the following module
# document_term(data_corpus)

# once the data is made into a document term matrix, no need to run the module again 
# as the result is pickled into the pkl file

# The document term matric is obtained from the pickled data.
data_dtm = pd.read_pickle("dtm_data.pkl")
# print(data_dtm)

# The most common words in the documents can be checked here and 
# do any processing like removing the most common words from all topics etc.

# most_common_words(data_dtm)


# sentiment analysis 
# senti_analyse(data_corpus)

# Sentiment of each essay can be divided into several parts and can be checked the 
# sentiments changing over time

# topic modeling
# tdm = data_dtm.transpose()
# topic_distribution=topic_distrbut(tdm)

# tdm = data_dtm

# print('topics are....')
# for entry in topic_distribution:
#     print(entry)

# improve it with including nouns only or nouns and adjectives

# Extract only one topic text
top1_text = data_orig.text.loc[0]
# top1_text = data_corpus.head[0]
print(top1_text[:200])
# print(data_corpus.topics)

# gen_tex = generated_text(top1_text)
# print(gen_text)


# text summaization
text_summary(top1_text)


print('done')



if __name__ == "__main__":
    import doctest
    doctest.testmod()
