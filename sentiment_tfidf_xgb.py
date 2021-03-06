#Twitter preprocessing and cleaning

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
from nltk.stem.porter import *
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


warnings.filterwarnings("ignore")

# Reading data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Appendig the train and test data
combi = train.append(test, ignore_index = True)

# Removing the twitter handels

def remove_pattern(input_txt, pattern):
	r = re.findall(pattern, input_txt)

	for i in r:
		input_txt = re.sub(i, '', input_txt)

	return input_txt

# Creating the new column tidy_tweet
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")

# Removing puncuations, Numbers and special characters
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

# Removing short words(like hmm, yaa okay etc etc)
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# Tokenization = splitting the datasets
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())

# Stemming = stripping the suffixes from the word
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) #stemming

# stitching this token back together

for i in range(len(tokenized_tweet)):
	tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet

# Story generation and visualisation from lists

all_words = ' '.join([text for text in combi['tidy_tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
#plt.show()


# Words in non racist/sexist tweets

normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label']==0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
#plt.show()

# Racist and sexist tweets

negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
#plt.show()


# Understanding the impact of hash tags in tweets 
# Function to collect hashtags

def hashtag_extract(x):
	hashtags = []

	# loop over the words in the tweet
	for i in x:
		ht = re.findall(r"#(\w+)",i) # for words staring woth '#'
		hashtags.append(ht)

	return hashtags

# Extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label']==0])

# Extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

# Unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative, [])

# Non-racist/sexist tweets
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})

# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
#plt.show()

# Racist/sexists tweets

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
#plt.show()


tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=2000, stop_words='english')

# TF-IDF feature matrix

tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])

# Building models using TF-IDF features
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train['label'], random_state=42, test_size=0.2)

lreg = XGBClassifier(learning_rate =0.01,
 n_estimators=1000,
 max_depth=15,
 min_child_weight=1,
 gamma=0,
 subsample=0.9,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict(xvalid_tfidf)
#prediction_int = prediction[:,1] >= 0.25
prediction_int = prediction.astype(np.int)
#prediction_int = prediction.astype(np.int)

score2 = f1_score(yvalid, prediction_int)

print(score2)


'''
test_pred1 = lreg.predict_proba(test_tfidf)
test_pred_int1 = test_pred1[:,1] >= 0.25
test_pred_int1 = test_pred_int1.astype(np.int)
test['label'] = test_pred_int1
submission = test[['id','label']]
submission.to_csv('sub.csv', index=False) # writing data to a CSV file



from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfVectorizer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)),])
text_clf.fit(xtrain_tfidf, ytrain)  
predicted = text_clf.predict(xvalid_tfidf)
print(np.mean(predicted == twenty_test.target))
'''