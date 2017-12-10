import os
import csv
from sklearn.feature_extraction.text import CountVectorizer

# This file do some pre-labeling stuff to speed up labeling process. But, at the end, all tweets' labels double-checked

os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\NTI Data Collection')

with open('Reduced_TI-Keyword.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    keywords = []
    for row in reader:
        keywords.append(row[0])

with open('Filter_DataCollection-NTI_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    NTI_Tweets = []
    for row in reader:
        NTI_Tweets.append(row)

TrafficCondition_words = ['open', 'opened', 'reopen', 'reopened', 're-open',
                          'opening', 'heavy', 'slow', 'jammed']
Clear_words = ['clear', 'cleared', 'clearance']

vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()

for i in range(len(NTI_Tweets)):
    if len(analyze(NTI_Tweets[i][2])) < 4:
        NTI_Tweets[i][0] = '-1'
        continue
    A = [word in analyze(NTI_Tweets[i][2]) for word in keywords]
    if A.count('True') == 1:
        NTI_Tweets[i][0] = '3'
    elif A.count('True') > 1:
        NTI_Tweets[i][0] = '2'
    elif A.count('True') == 0:
        NTI_Tweets[i][0] = '1'

with open('Labeled_DataCollection-NTI_Tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in NTI_Tweets:
        writer.writerow(tweet)
