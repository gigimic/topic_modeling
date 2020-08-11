from read_data import get_text_for_topic
from clean_data import clean_text_string
from visual_data import visualize_word_cloud

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
