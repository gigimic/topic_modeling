import sys
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_word_cloud(name_topics, text):
    for i in range(len(name_topics)):
        plt.subplot(3, 2, i+1)
        wordcloud = WordCloud(width=480, height=480, margin=0).generate(text[i])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
    plt.show()

if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    data_corpus = pd.read_pickle("topic_data.pkl")
    text = data_corpus['text']
    name_topics = ["globalization", "mahatma gandhi", "fake news", "women empowerment", "Palliative care", 
    "Auschwitz concentration camp"]
    visualize_word_cloud(name_topics, text)
    #testing different cases
    # wordcloud = WordCloud(width=480, height=480, margin=0).generate(text[0])
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.margins(x=0, y=0)
    # # plt.show()
    # plt.savefig("wordcloud_global.jpg", dpi=150)

