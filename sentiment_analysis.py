# sentiment analysis 

from textblob import TextBlob
import matplotlib.pyplot as plt

# Create quick lambda functions to find the polarity and subjectivity of each routine
# Terminal / Anaconda Navigator: conda install -c conda-forge textblob

def senti_analyse(data_corpus):
    pol = lambda x: TextBlob(x).sentiment.polarity
    sub = lambda x: TextBlob(x).sentiment.subjectivity

    data_corpus['polarity'] = data_corpus['text'].apply(pol)
    data_corpus['subjectivity'] = data_corpus['text'].apply(sub)
    print(data_corpus['polarity'])
    print(data_corpus['subjectivity'])

    # polarity and subjectivity are plotted here
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