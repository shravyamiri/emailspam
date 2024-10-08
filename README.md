import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

import re
import string
import random
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch


from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import LSTM, Embedding, BatchNormalization, Dense, Dropout, Bidirectional, Flatten, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score


import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sms_data = pd.read_csv('/content/spam.csv', encoding = 'latin-1')
sms_data.dropna(how="any", inplace=True, axis=1)
sms_data.columns = ['label', 'message']
sms_data.head()

sms_data.shape

shuffled_data = sms_data.sample(frac = 1, random_state = 42)

random_index = np.random.randint(0, len(shuffled_data)-5)
for _, row in shuffled_data[['label', 'message']][random_index: random_index + 5].iterrows():
  target = row['label']
  text = row['message']
  print(f"Target: {target}", "(real message)" if target == 'ham' else "(Spam messege)")
  print(f"Text:\n{text}\n")
  print("---\n")


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

shuffled_data['label'] = encoder.fit_transform(shuffled_data['label'])
shuffled_data['label'].value_counts()

shuffled_data.duplicated().sum()

shuffled_data.drop_duplicates(inplace = True, keep = 'first')
shuffled_data.shape


sms_data['len_messages'] = sms_data.message.apply(len)
sms_data.head()

plt.figure(figsize=(12, 8))

sms_data[sms_data.label =='ham'].len_messages.plot(bins = 35,
                                                            kind = 'hist',
                                                            color = 'blue',
                                       label = 'Ham messages', alpha = 0.6)
sms_data[sms_data.label == 'spam'].len_messages.plot(kind = 'hist',
                                                             color = 'red',
                                       label = 'Spam messages', alpha = 0.6)
plt.legend()
plt.xlabel("Message Length")

balance_counts = sms_data.groupby('label')['label'].agg('count').values

data = {
    'Category': ['ham', 'spam'],
    'Count': [balance_counts[0], balance_counts[1]]
}

df = pd.DataFrame(data)

plt.figure(figsize = (10, 6))
barplot = sns.barplot(x = 'Category', y = 'Count', data = df)

for index, row in df.iterrows():
    barplot.text(index, row.Count, row.Count, color = 'black', ha = "center")

plt.title('Dataset distribution by target', fontsize = 32, fontname = 'Times New Roman')
plt.show()

ham_df = sms_data[sms_data['label'] == 'ham']['len_messages'].value_counts().sort_index()
spam_df = sms_data[sms_data['label'] == 'spam']['len_messages'].value_counts().sort_index()

ham_data = pd.DataFrame({'len_messages': ham_df.index, 'count': ham_df.values, 'label': 'ham'})
spam_data = pd.DataFrame({'len_messages': spam_df.index, 'count': spam_df.values, 'label': 'spam'})

combined_df = pd.concat([ham_data, spam_data])

plt.figure(figsize = (10, 6))
sns.lineplot(data = combined_df, x = 'len_messages', y = 'count', hue = 'label')

plt.title('Data Roles in Different Fields', fontsize = 32, fontname = 'Times New Roman')
plt.xlim(0, 70)
plt.show()

wc = WordCloud(
    background_color='white',
    max_words=200,
)
wc.generate(' '.join(text for text in sms_data.loc[sms_data['label'] == 'ham', 'message']))
plt.figure(figsize=(18,10))
plt.title('Top words for HAM messages',
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()

wc = WordCloud(
    background_color='white',
    max_words=200,
)
wc.generate(' '.join(text for text in sms_data.loc[sms_data['label'] == 'spam', 'message']))
plt.figure(figsize=(18,10))
plt.title('Top words for SPAM messages',
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()

from nltk.corpus import stopwords

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers
    Remove all stopwords
    Returns a list of the cleaned text
    """
    text = str(mess).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']

    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    text = re.sub('\n', '', nopunc)
    text = re.sub('\w*\d\w*', '', text)

    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = text_process(text)
    # Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))

    return text

import nltk
nltk.download('stopwords')
shuffled_data['cleaned_message'] = shuffled_data['message'].apply(preprocess_data)
shuffled_data.head()

from sklearn.model_selection import train_test_split
# split data first
X = shuffled_data.cleaned_message
y = shuffled_data.label
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify = y)
print(X_train.shape, y_train.shape)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_dtm)
tfidf_transformer.transform(X_train_dtm)

svc = SVC(kernel = 'sigmoid', gamma = 1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
gnb = GaussianNB()
bnb = BernoulliNB()
dtc = DecisionTreeClassifier(max_depth = 5)
lrc = LogisticRegression(solver = 'liblinear', penalty = 'l1')
rfc = RandomForestClassifier(n_estimators = 50, random_state = 2)
abc = AdaBoostClassifier(n_estimators = 50, random_state = 2)
bc = BaggingClassifier(n_estimators = 50, random_state = 2)
etc = ExtraTreesClassifier(n_estimators = 50, random_state = 2)
gbdt = GradientBoostingClassifier(n_estimators = 50,random_state = 2)
xgb = XGBClassifier(n_estimators = 50,random_state = 2)

clfs = {
    'SVC' : svc,
    'KN' : knc,
    'NB': mnb,
    'gnb': gnb,
    'bnb': bnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}

def train_and_evaluate_classifiers(classifiers, X_train, y_train, X_test, y_test):

    # Convert sparse matrices to dense, if needed
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()

    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    results = {}

    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)

        # Predictions for training and test sets
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Calculate accuracies and precision
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average = 'weighted')  # Handle multi-class if needed

        # Store the results
        results[clf_name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision
        }

    return results

results = train_and_evaluate_classifiers(clfs, X_train_dtm, y_train, X_test_dtm, y_test)

# To view the results
for clf_name, metrics in results.items():
    print(f"{clf_name} -> Train Accuracy: {metrics['train_accuracy']:.4f}, "
          f"Test Accuracy: {metrics['test_accuracy']:.4f}, "
          f"Precision: {metrics['precision']:.4f}")

performance_df1 = pd.DataFrame(results).T.sort_values('test_accuracy',ascending = False)
performance_df1

plt.figure(figsize = (10, 6))
sns.barplot(x = performance_df1.index, y = performance_df1['test_accuracy'],
            palette = 'viridis')

# Add titles and labels
plt.title('Model Performance Based on Test Accuracy', fontsize = 16)
plt.xlabel('Classifiers', fontsize = 12)
plt.ylabel('Test Accuracy', fontsize = 12)

# Rotate x-axis labels for better readability
plt.xticks(rotation = 45, ha = 'right')

# Show plot
plt.tight_layout()
plt.show()


x_axes = ['Ham', 'Spam']
y_axes =  ['Spam', 'Ham']

def conf_matrix(z, x = x_axes, y = y_axes):
    z = np.flip(z, 0)

    plt.figure(figsize = (10, 8))
    ax = sns.heatmap(z, annot = True, fmt = 'd', cmap = 'viridis',
                     xticklabels = x, yticklabels = y, cbar = True)

    ax.set_title('Confusion matrix', fontsize = 16, fontweight = 'bold')
    ax.set_xlabel('Predicted value')
    ax.set_ylabel('Real value')

    plt.show()

from sklearn.metrics import confusion_matrix
mnb = MultinomialNB()
mnb.fit(X_train_dtm, y_train)

# Predictions for training and test sets
y_test_pred = mnb.predict(X_test_dtm)

print(accuracy_score(y_test, y_test_pred))

conf_matrix(confusion_matrix(y_test, y_test_pred))

from sklearn.ensemble import VotingClassifier

mnb = MultinomialNB()
lrc = LogisticRegression(solver = 'liblinear', penalty = 'l1')
etc = ExtraTreesClassifier(n_estimators = 50, random_state=2)

voting = VotingClassifier(estimators=[('lrc', lrc), ('nb', mnb), ('et', etc)],voting = 'soft')

voting.fit(X_train_dtm, y_train)

y_test_pred = voting.predict(X_test_dtm)

print('Accuracy', accuracy_score(y_test, y_test_pred))
print("Precision", precision_score(y_test, y_test_pred))

from sklearn.ensemble import StackingClassifier

estimators = [('lrc', lrc), ('nb', mnb), ('et', etc)]
final_estimator = XGBClassifier(n_estimators = 50,random_state = 2)

clf = StackingClassifier(estimators = estimators, final_estimator = final_estimator)
clf.fit(X_train_dtm, y_train)

y_test_pred = clf.predict(X_test_dtm)

print('Accuracy', accuracy_score(y_test, y_test_pred))
print("Precision", precision_score(y_test, y_test_pred))

import xgboost as xgb
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('bow', CountVectorizer()),
    ('tfid', TfidfTransformer()),
    ('model', xgb.XGBClassifier(
        learning_rate = 0.1,
        max_depth = 7,
        n_estimators = 80,
        use_label_encoder = False,
        eval_metric = 'auc',
    ))
])

# Fit the pipeline with the data
pipe.fit(X_train, y_train)

y_pred_class = pipe.predict(X_test)
y_pred_train = pipe.predict(X_train)

print('Train: {}'.format(accuracy_score(y_train, y_pred_train)))
print('Test: {}'.format(accuracy_score(y_test, y_pred_class)))

conf_matrix(confusion_matrix(y_test, y_pred_class))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word2index = tokenizer.word_index

len(word2index)+1

max_len = round(sum([len(i.split()) for i in X_train]) / len(X_train))
max_len
