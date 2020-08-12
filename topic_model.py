from read_data import get_text_for_topic
from clean_data import clean_text_string
from visual_data import visualize_word_cloud
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
# import matplotlib.pyplot as plt

# name_topics = ["globalization", "hollywood", "mahatma gandhi", "fake news", "women empowerment"]
name_topics = ["globalization", "fake news"]
topic_clean_text=[]
for index, name_topic in enumerate(name_topics):
    print(index, name_topic)
    topic_text = get_text_for_topic(name_topic)
    topic_clean_text.append(clean_text_string(topic_text))
    print(topic_clean_text[index][0:100])

    
# This function can be called to visualise the word clloud of various topics
# visualize_word_cloud(name_topics, topic_clean_text)

# document-term-matrix
# check this in the data cleaning module


# cv = CountVectorizer(stop_words='english')
cv = CountVectorizer()
data_cv = cv.fit_transform(topic_clean_text)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
# data_dtm.index = data_clean.index
print(data_dtm)

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
