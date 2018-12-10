import numpy as np
import pandas as pd
import sys
import re
import string
import random
import json
import os
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

stemmer = SnowballStemmer("english")
lemmatizer = nltk.stem.WordNetLemmatizer()

d = pd.read_json('NewYorkTimesClean.jsonl', lines=True)
dff = pd.DataFrame(data=d, columns=['headline', 'keywords', 'lead_paragraph', 'section'])
df = dff.sample(frac=0.2, replace=True)
df = df.fillna('nullval')
df = df[~df.lead_paragraph.str.contains('nullval')]
df = df[~df.index.duplicated()]

# Clean up data.
regex1 = re.compile('[%s]' % re.escape(string.punctuation))
stop = stopwords.words('english')
df['section'] = df.section.apply(lambda x : str.lower(x))
df['section'] = df.section.apply(lambda x : re.sub(r'[^\w\s]+', '', x))
df['lead_paragraph'] = df.lead_paragraph.fillna('nullval').apply(lambda x : str.lower(x))
df['lead_paragraph'] = df.lead_paragraph.apply(lambda x : re.sub(r'[^\w\s]+', '', x))
df['lead_paragraph'] = df.lead_paragraph.apply(lambda x : re.sub(r'\b\d+(?:\.\d+)?\s+', '', x))
df['lead_paragraph'] = df.lead_paragraph.apply(lambda x : re.sub("[^a-zA-Z]",  " ",  str(x)))

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in df['lead_paragraph']:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
#print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=10000,
                                 min_df=0.005, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(df['lead_paragraph'].values.tolist()) #fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)

# KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

joblib.dump(km, 'kmeans-cluster.pkl')
km = joblib.load('kmeans-cluster.pkl')
clusters = km.labels_.tolist()

section_list = df['section'].values.tolist()
section_cat = { 'section': section_list, 'main_paragraph': df['lead_paragraph'], 'cluster': clusters}

frame = pd.DataFrame(section_cat, index = [clusters] , columns = ['section', 'main_paragraph', 'cluster'])

print("Top terms per cluster:\n")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    # replace 6 with n words per cluster
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print()
    print()

    print("Cluster %d section:" % i, end='')
    for section in frame.loc[i]['section'].values.tolist():
        print(' %s,' % section, end='')
    print()
    print()

print()
print()

# My attempt at plotting this as visual data but couldn't get it to do anything.

tfidf_matrix_reduced = TruncatedSVD(n_components=num_clusters, random_state=0).fit_transform(tfidf_matrix)
tfidf_matrix_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(tfidf_matrix_reduced)

fig = plt.figure(figsize = (10, 10))
ax = plt.axes()
plt.scatter(tfidf_matrix_embedded[:, 0], tfidf_matrix_embedded[:, 1], marker = ".", c = km.labels_)
ax.legend()
plt.show()
