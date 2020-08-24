import pickle
from gensim import matutils, models
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer


def topic_distrbut(tdm):
    # We're going to put the term-document matrix into a new gensim format, 
    # from df --> sparse matrix --> gensim corpus
    sparse_counts = scipy.sparse.csr_matrix(tdm)
    corpus = matutils.Sparse2Corpus(sparse_counts)

    # Gensim also requires dictionary of the all terms and their respective 
    # location in the term-document matrix
    cv = pickle.load(open("cv.pkl", "rb"))
    id2word = dict((v, k) for k, v in cv.vocabulary_.items())

    # Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
    # we need to specify two other parameters as well - the number of topics and the number of passes

    lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, passes=10)
    print(lda.print_topics())

    #  Let's take a look at which topics each transcript contains
    dtm = tdm.transpose()
    corpus_transformed = lda[corpus]
    topic_distribution=list(zip([a for [(a,b)] in corpus_transformed], dtm.index))
    return topic_distribution



if __name__ == "__main__":
    import doctest
    doctest.testmod()
