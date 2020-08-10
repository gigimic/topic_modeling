# Web scraping, pickle imports
import requests
from bs4 import BeautifulSoup
# import pickle
import wikipedia
# print(wikipedia.__version__)

# print(wikipedia.summary("Wikipedia"))


def get_text_for_topic(name_topic):
    topic1 = wikipedia.page(name_topic)
    print(topic1.title)
    print(topic1.url)
    print(len(topic1.content))
    # print(topic1.content[0:500])
    return topic1.content

