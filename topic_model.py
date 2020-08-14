import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from read_data import get_text_for_topic
from clean_data import clean_text_string
from visual_data import visualize_word_cloud

print('Pandas version is : ', pd.__version__)

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


data = pd.DataFrame({'topics':name_topics, 'text':topic_clean_text})
# print(data['topics'][0])
# print(data.text.loc['fake news'])
# print(data['text'][1])

# TODO here the data frame has to be corrected. the column name should be 'text'

# pd.set_option('max_colwidth',150)

# data_df = pd.DataFrame.from_dict(data).transpose()
# data_df.columns = data['text']
# data_df = data_df.sort_index()
# print(len(data_df))

# # data_df.text.loc['globalization']

# document-term-matrix
# check this in the data cleaning module


cv = CountVectorizer(stop_words='english')
# cv = CountVectorizer()
data_cv = cv.fit_transform(data.text)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data.topics
print(len(data_dtm))
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
