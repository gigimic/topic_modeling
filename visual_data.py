import sys
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

def visualize_word_cloud(i: int, text):
    plt.subplot(1, 2, i+1)
    wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
