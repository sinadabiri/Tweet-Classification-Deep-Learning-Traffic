import os
import csv
from sklearn.feature_extraction.text import CountVectorizer

import re
os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\TI-Data Collection')

# This file apply some sort of quick calculation for counting and then adjusting the number of tweets from each label
# Also apply more processing such as re-labeling specifically label 2 tweets.

# Purely-TI-keywords: around 60 keywrods. Comes originally from the All-TI-Keywords and observation from collected
# TI tweets. These are the words that
with open('Purely-TI-Keywords.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    keywords = []
    for row in reader:
        keywords.append(row[0])

keywords = list(set(keywords))

with open('stopwords.txt', 'r', newline='') as f:
    reader = csv.reader(f)
    stopwords = []
    for row in reader:
        for i in row:
            A = re.sub('"', '', i)
            A = re.sub(' ', '', A)
            stopwords.append(A)
with open('2-Labeled_DataCollection-TI_IUs_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    TI_Tweets = []
    for row in reader:
        TI_Tweets.append(row)

with open('1-Remain-Labeled_DataCollection-TI_IUs_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        TI_Tweets.append(row)

count = [0, 0, 0, 0]
for i in range(len(TI_Tweets)):
    if TI_Tweets[i][0] == '-1':
        count[0] += 1
    if TI_Tweets[i][0] == '1':
        count[1] += 1
    if TI_Tweets[i][0] == '2':
        count[2] += 1
    if TI_Tweets[i][0] == '3':
        count[3] += 1

with open('3-Mixed-Labeled_DataCollection-TI_IUs_Tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in TI_Tweets:
        writer.writerow(tweet)

# Relabeling the label 2 again.

TrafficCondition_words = ['open', 'opened', 'reopen', 'reopened', 're-open',
                          'opening', 'heavy', 'slow', 'jammed', 'clear', 'cleared', 'clearance']


vectorizer = CountVectorizer(min_df=1, stop_words=stopwords, ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()

for i in range(len(TI_Tweets)):
    if TI_Tweets[i][0] == '2':
        wordSplit = analyze(TI_Tweets[i][2])
        if any(word in analyze(TI_Tweets[i][2]) for word in keywords):
            TI_Tweets[i][0] = '2'
        elif any(word in analyze(TI_Tweets[i][2]) for word in TrafficCondition_words):
            TI_Tweets[i][0] = '4'
        else:
            TI_Tweets[i][0] = '5'


with open('1-Remain-Labeled_DataCollection-TI_IUs_Tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in TI_Tweets:
        writer.writerow(tweet)
