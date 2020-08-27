# text analysis 

data collected from various sources
cleaned the data: removed punctuations, special characters, numbers and kept only the text
cleaned data was saved as a pkl file (pickle file - binary): corpus data

document term matrix was generated using sklearn - countervectorizer
word cloud was used to visualize the most common words
if there are common words in all the text, they can be removed. 

sentiment analysis was done with polarity (positive and negative) and subjectivity (facts and opinions).
this gives an idea of which text gives positive news and facts and which one gives negative news and opinions

topic modelling was done to classify the text into different topics. The number of topics were selected by trial and error method.

text generation was also attempted.

entry point is topic_model