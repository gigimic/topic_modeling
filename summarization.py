import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
import numpy as np
import pandas as pd

'''Text summarization refers to the technique of shortening long pieces of text. 
The intention is to create a coherent and fluent summary having only the main points outlined in the document.
Automatic text summarization is a common problem in machine learning and natural language processing (NLP).'''

def text_summary(text):
    # Tokenizing the text 
    stopWords = set(stopwords.words("english")) 
    words = word_tokenize(text) 
    print(len(words))

    # Creating a frequency table to keep the  
    # score of each word 
    
    freqTable = dict() 
    for word in words: 
        word = word.lower() 
        if word in stopWords: 
            continue
        if word in freqTable: 
            freqTable[word] += 1
        else: 
            freqTable[word] = 1
    print(len(freqTable))    
    # Creating a dictionary to keep the score 
    # of each sentence 
    sentences = sent_tokenize(text) 
    sentenceValue = dict() 
    
    for sentence in sentences: 
        for word, freq in freqTable.items(): 
            if word in sentence.lower(): 
                if sentence in sentenceValue: 
                    sentenceValue[sentence] += freq 
                else: 
                    sentenceValue[sentence] = freq 
   
    sumValues = 0
    for sentence in sentenceValue: 
        sumValues += sentenceValue[sentence] 
    
    # Average value of a sentence from the original text 
    
    average = int(sumValues / len(sentenceValue)) 
    print(average)

      
    # Storing sentences into our summary. 
    summary = '' 
    for sentence in sentences: 
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.6 * average)): 
            summary += " " + sentence 
    print(summary) 



if __name__ == "__main__":
    import doctest
    data_orig=pd.read_pickle('data_original.pkl')
    top1_text = data_orig.text.loc[0]
    text_summary(top1_text)
    # doctest.testmod()
