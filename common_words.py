import pandas as pd 
from collections import Counter

def most_common_words(data_dtm):
    data_dtm = data_dtm.transpose()
    top_dict = {}
    for c in data_dtm.columns:
        top = data_dtm[c].sort_values(ascending=False).head(20)
        top_dict[c]= list(zip(top.index, top.values))
        print('printing top words...')
        print(top_dict[c])

# print(top_dict)

# Print the top 10 words from each text 
    for topic, top_words in top_dict.items():
        print(topic)
        print(', '.join([word for word, count in top_words[0:9]]))
        print('---')

    # If there are common words in all the topics which appear many times they can be removed.
    # Look at the most common top words --> add them to the stop word list
    

    # Let's first pull out the top 10 words for each topic
    words = []
    for topic in data_dtm.columns:
        top = [word for (word, count) in top_dict[topic]]
        for t in top:
            words.append(t)
            
    print(words)

if __name__ == "__main__":
    import doctest
    doctest.testmod()