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
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Extract randomized sample selection
d = pd.read_json('newyorktimes_filtered.jsonl', lines=True)
dff = pd.DataFrame(data=d, columns=['headline', 'keywords', 'lead_paragraph', 'section'])
df = dff.sample(frac=0.2, replace=True)
df = df.fillna('nullval') # or just: TextData df['lead_paragraph'].fillna('some value')
df = df[~df.lead_paragraph.str.contains('nullval')]

# Clean up data.
regex1 = re.compile('[%s]' % re.escape(string.punctuation))
stop = stopwords.words('english')
# Remove stop words
def f(x):
    return [w for w in x if w not in stop]

df['section'] = df.section.apply(lambda x : str.lower(x))
df['section'] = df.section.apply(lambda x : re.sub(r'[^\w\s]+', '', x))

df['lead_paragraph'] = df.lead_paragraph.fillna('nullval').apply(lambda x : str.lower(x))
df['lead_paragraph'] = df.lead_paragraph.apply(lambda x : re.sub(r'[^\w\s]+', '', x))
df['lead_paragraph'] = df.lead_paragraph.apply(lambda x : re.sub(r'\b\d+(?:\.\d+)?\s+', '', x))
df['lead_paragraph'] = df.lead_paragraph.apply(lambda x : re.sub("[^a-zA-Z]",  " ",  str(x)))
df['lead_paragraph'] = df['lead_paragraph'].str.split().apply(f).str.join(' ')
#df['lead_paragraph'] = df['lead_paragraph'].fillna('nullval').apply(lambda x : word_tokenize(str(x)))

new = df[['section', 'lead_paragraph']]

data_features = new.iloc[:,1]
data_labels = new.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(data_features,
                                                   data_labels,
                                                   test_size=.2, random_state=1234)

################################################################
################ TF-IDF ########################################

# Use CountVectorizer to count the number of occurences for each term.
cvec = CountVectorizer(stop_words='english', min_df=.0025, max_df=.1, ngram_range=(1,2))
cvec.fit(data_features)

# Top 20 most common terms.
cvec_counts = cvec.transform(data_features)
occurences = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
df_counts = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occurences})
#print(df_counts.sort_values(by='occurrences', ascending=False).head(20))

# Tf-IDF; using TfidfTransformer to calculate weights for each term.
tfidf_trans = TfidfTransformer()
tfidf_trans_weights = tfidf_trans.fit_transform(cvec_counts)

# Top 20 terms using average TF-IDF weight.
weights = np.asarray(tfidf_trans_weights.mean(axis=0)).ravel().tolist()
df_weights = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
#print(df_weights.sort_values(by='weight', ascending=False).head(20))

################################################################
################################################################

################################################################
################ Multinomial Naive Bayes #######################

# tokenize train and test text data
vect = CountVectorizer()
X_train_tokens = vect.fit_transform(X_train)
X_test_tokens = vect.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_tokens, y_train)
y_pred_nb = nb.predict(X_test_tokens)
# Precision, Accuracy, and Recall model_eval_calculations for NB and SVM.
def model_eval_calculations(y_test, y_pred_nb):
    print('Precision:', precision_score(y_test, y_pred_nb, average='weighted'))
    print('Accuracy:', accuracy_score(y_test, y_pred_nb))
    print('Recall:', recall_score(y_test, y_pred_nb, average='weighted'))

# Print NB Precision, Accuracy, and Recall
model_eval_calculations(y_test, y_pred_nb)
