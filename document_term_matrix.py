
# here we create a document term matrix: The most common tokenization technique is to 
# break down text into words. We can do this using scikit-learn's CountVectorizer, 
# where every row will represent a different document and every column will represent a different word

def document_term(text):
    cv = CountVectorizer(stop_words='english')
    # cv = CountVectorizer()
    data_cv = cv.fit_transform(data.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = data.topics
    print(len(data_dtm))
    # print(data_dtm)




if __name__ == "__main__":
    import doctest
    doctest.testmod()