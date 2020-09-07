# text analysis 

data collected from various sources
cleaned the data: removed punctuations, special characters, numbers and kept only the text
cleaned data was saved as a pkl file (pickle file - binary): corpus data

text vectorization: It is the method to transform text into numerical vectors
Most popular methods are 'Bag of words', or TF-IDF
'Bag of words' provides the frequency of words in each document
TF-IDF provides the normalized frequency
TF-IDF algorithm (Term Frequency-Inverse Document Frequency)
TF – shows the frequency of the term in the text, as compared with the total number of the words in the text. It is the ratio of the number of times the term appears in the text to the total number of terms in the text.
IDF – is the inverse frequency of terms in the text. It simply displays the importance of each term. It is calculated as a logarithm of the number of texts divided by the number of texts containing this term.
document term matrix was generated using sklearn - countvectorizer
countvectorizer use the method 'Bag of words'


word cloud was used to visualize the most common words
if there are common words in all the text, they can be removed. 

sentiment analysis was done with polarity (positive and negative) and subjectivity (facts and opinions).
this gives an idea of which text gives positive news and facts and which one gives negative news and opinions

topic modelling was done to classify the text into different topics. The number of topics were selected by trial and error method.

text generation was also attempted.

entry point is topic_model