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
import matplotlib.patches as mpatches

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

# Instantiate parameters for vectorizer
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

# Dump into pickle file
joblib.dump(km, 'kmeans-cluster.pkl')
km = joblib.load('kmeans-cluster.pkl')

clusters = km.labels_.tolist()
section_list = df['section'].values.tolist()
section_cat = { 'section': section_list, 'main_paragraph': df['lead_paragraph'], 'cluster': clusters}
frame = pd.DataFrame(section_cat, index = [clusters] , columns = ['section', 'main_paragraph', 'cluster'])

print("Top terms per cluster:\n")
# Sort the centers of clusters by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    # replace 6 with n words per cluster
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print('\n')

    print("Cluster %d section:" % i, end='')
    for section in frame.loc[i]['section'].values.tolist():
        print(' %s,' % section, end='')
    print('\n')

print('\n')

# Visualization
tfidf_matrix_reduced = TruncatedSVD(n_components=num_clusters, random_state=0).fit_transform(tfidf_matrix)
tfidf_matrix_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(tfidf_matrix_reduced)

fig = plt.figure(figsize = (10, 10))
ax = plt.axes()

xs, ys = tfidf_matrix_embedded[:, 0], tfidf_matrix_embedded[:, 1]
dfd = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=section_list))
groups = dfd.groupby('label')

# Colors per cluster
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

# Clusters formed by most frequent terms
cluster_names = {0: 'Top terms for cluster #0: years, new, time, week, like, president',
                 1: 'Top terms for cluster #1: january, loving, family, beloved, wife, friends',
                 2: 'Top terms for cluster #2: new, york, new, city, york, new',
                 3: 'Top terms for cluster #3: said, states, unit, unit, officials, mr',
                 4: 'Top terms for cluster #4: editor, letter, jan, dec, pages, susan'}
for name, group in groups:
    ax.plot(group.x, group.y, marker='.', linestyle='',
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',
        which='both',
        left='off',
        top='off',
        labelleft='off')

ax.legend(numpoints=1)
plt.savefig('kmeans_visual.png', dpi=200)
plt.show()
