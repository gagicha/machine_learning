
# coding: utf-8

# In[ ]:

import nltk
import re
import pandas as pd
from bs4 import BeautifulSoup
data=open('RawDataSet.txt')
soup = BeautifulSoup(data,'html.parser')#expects string or bytes like object
[s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title','font','None'])]
visible_text = soup.getText()
list=visible_text.splitlines()
url=[]
content=[]
for line in list:
        a,b,*rest=line.split(': u')
        url.append(a)
        content.append(b)

stopwords = nltk.corpus.stopwords.words('english')
# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if token.lower() not in stopwords:
                filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if token.lower() not in stopwords:
                filtered_tokens.append(token)
    return filtered_tokens


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in content:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print( 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(content) 
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
get_ipython().magic('time km.fit(tfidf_matrix)')
clusters = km.labels_.tolist()

text_cluster = {'content': content, 'cluster': clusters, 'url': url }
frame = pd.DataFrame(text_cluster, index = [clusters] , columns = [ 'cluster', 'url'])

frame['cluster']
frame['url']

from __future__ import print_function
#from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import numpy
print("Top url per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d url:" % i, end='')
    print()
    
    for ind in order_centroids[i, :10]:
        print(' %s' % url[ind], end=',') 
        distance = order_centroids[i,ind]
        print (distance)
        print()
    
print()
print()

