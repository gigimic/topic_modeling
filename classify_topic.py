import pandas as pd
from nltk import word_tokenize, pos_tag
import pickle
from gensim import matutils, models
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer

# Here we take only nouns from the text data to classify the data into 5 topics 
# we need to choose the number of topics in a trial and error method 

name_topics = ["globalization", "mahatma gandhi", "fake news", "women empowerment", "Palliative care", 
"Auschwitz concentration camp"]

data_corpus = pd.read_pickle("topic_data.pkl")

def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    # print(pos_tag(tokenized))
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)


# Apply the nouns function to the transcripts to filter only on nouns
data_nouns = pd.DataFrame(data_corpus.text.apply(nouns))
data_nouns.topics = data_corpus.topics

print(len(data_nouns))
# print(data_nouns.topics)

# Create a new document-term matrix using only nouns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.text)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.topics

#  Create the gensim corpus
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())

# Let's start with 3 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=3, id2word=id2wordn, passes=10)
print(ldan.print_topics())

# Let's take a look at which topics each transcript contains
corpus_transformed = ldan[corpusn]
topic_distribution=list(zip([a for [(a,b)] in corpus_transformed], data_dtmn.index))
# print(topic_distribution)
print('topics are....using nouns only')
for entry in topic_distribution:
    print(entry)


def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ' 
    tokenized = word_tokenize(text)
    # print(pos_tag(tokenized))
    all_nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns_adj)


# Apply the nouns_adj function to the transcripts to filter only on nouns and adjectives
data_nouns_adj = pd.DataFrame(data_corpus.text.apply(nouns_adj))
data_nouns_adj.topics = data_corpus.topics

print(len(data_nouns_adj))
# print(data_nouns.topics)

# Create a new document-term matrix using only nouns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cvnadj = CountVectorizer(stop_words=stop_words)
data_cvnadj = cvnadj.fit_transform(data_nouns_adj.text)
data_dtmnadj = pd.DataFrame(data_cvnadj.toarray(), columns=cvnadj.get_feature_names())
data_dtmnadj.index = data_nouns_adj.topics

#  Create the gensim corpus
corpusnadj = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmnadj.transpose()))

# Create the vocabulary dictionary
id2wordnadj = dict((v, k) for k, v in cvnadj.vocabulary_.items())

# Let's start with 3 topics
ldanadj = models.LdaModel(corpus=corpusnadj, num_topics=3, id2word=id2wordnadj, passes=10)
print(ldanadj.print_topics())

# Let's take a look at which topics each transcript contains
corpus_transformed = ldanadj[corpusnadj]
topic_distribution=list(zip([a for [(a,b)] in corpus_transformed], data_dtmnadj.index))
# print(topic_distribution)
print('topics are....using nouns and adjectives only')
for entry in topic_distribution:
    print(entry)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
