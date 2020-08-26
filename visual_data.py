import sys
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np


def visualize_word_cloud(name_topics, text):
    for i in range(len(name_topics)):
        plt.subplot(1, 2, i+1)
        wordcloud = WordCloud(width=480, height=480, margin=0).generate(text[i])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
    plt.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
