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
this gives an idea of which text gives positive news and facts and which one gives negative news and opinions.
The sentiment function of textblob returns two properties, polarity, and subjectivity.
Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement. Subjective sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. Subjectivity is also a float which lies in the range of [0,1].

Sentiment analysis models focus on polarity (positive, negative, neutral) but also on feelings and emotions (angry, happy, sad, etc), and even on intentions (e.g. interested v. not interested).
A rule-based system uses a set of given rules to help identify subjectivity, polarity, or the subject of an opinion.
The rule-based system works like the following:

    Defines two lists of polarized words (e.g. negative words such as bad, worst, ugly, etc and positive words such as good, best, beautiful, etc).
    Counts the number of positive and negative words that appear in a given text.
    If the number of positive word appearances is greater than the number of negative word appearances, the system returns a positive sentiment, and vice versa. If the numbers are even, the system will return a neutral sentiment.

In automatic approach, the task is considered a classification problem with categories of positive, nagatice or neutral.
Here a training (tagged) set is used for feature extraction and fed into the machine learning algorithms

The feature vector generated using the feature extraction is used in the model to classify the text and tags them.

textblob uses Naive Bayes and Decision Tree classifiers.


topic modelling was done to classify the text into different topics. The number of topics were selected by trial and error method.

text generation was also attempted.

entry point is topic_model