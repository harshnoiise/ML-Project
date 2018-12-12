import numpy as np
import sys
from nltk.corpus import stopwords
import re
import string
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import StandardScaler
import random
import collections
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from nltk import pos_tag
from nltk.stem import PorterStemmer
from collections import Counter
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

# Extract randomized sample selection
d = pd.read_json('NewYorkTimesClean.jsonl', lines=True)
dff = pd.DataFrame(data=d, columns=['headline', 'keywords', 'lead_paragraph', 'section'])
df = dff.sample(frac=0.2, replace=True)
df = df.fillna('nullval') # or just: TextData df['lead_paragraph'].fillna('some value')
df = df[~df.lead_paragraph.str.contains('nullval')]
df = df[~df.headline.str.contains('nullval')]
df = df[~df.section.str.contains('nullval')]
df['section_id'] = df.section.factorize()[0]

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=17, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.lead_paragraph).toarray()
labels = df.section_id.values.tolist()

y = labels
X = features

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Clustering the data with Random Forest Classifier
clf = RandomForestClassifier(n_estimators=1000, random_state=0)
clf.fit(X_train, y_train)
y_pred= clf.predict(X_test)
print(classification_report(y_test,y_pred))
acc = accuracy_score(y_test, y_pred)
print('Random Forest accuracy = ' + str(acc * 100) + '%')
