import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from gensim import matutils, models
import scipy.sparse


from read_data import get_text_for_topic
from clean_data import clean_text_string
from visual_data import visualize_word_cloud
from document_term_matrix import document_term
from common_words import most_common_words

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

# This function can be called to visualise the word cloud of various topics
# visualize_word_cloud(name_topics, topic_clean_text)

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

most_common_words(data_dtm)


# sentiment analysis 
# Create quick lambda functions to find the polarity and subjectivity of each routine
# Terminal / Anaconda Navigator: conda install -c conda-forge textblob
from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data_corpus['polarity'] = data_corpus['text'].apply(pol)
data_corpus['subjectivity'] = data_corpus['text'].apply(sub)
print(data_corpus['polarity'])
print(data_corpus['subjectivity'])
# data
#  Let's plot the results
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [10, 8]

# for index, topics in enumerate(data.index):
#     x = data.polarity.loc[topics]
#     y = data.subjectivity.loc[topics]
#     plt.scatter(x, y, color='blue')
#     plt.text(x+.001, y+.001, data['topics'][index], fontsize=10)
#     # plt.xlim(-.01, .12) 
    
# plt.title('Sentiment Analysis', fontsize=20)
# plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
# plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

# plt.show()

# Sentiment of Routine Over Time
##########################

# topic modeling
# tdm = data_dtm.transpose()
tdm = data_dtm
print(tdm.head())

# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix

id2word = dict((v, k) for k, v in cv.vocabulary_.items())
# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, passes=10)
print(lda.print_topics())
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
