import os
import csv
from sklearn.feature_extraction.text import CountVectorizer

# This file do some pre-labeling stuff to speed up labeling process. But, at the end, all tweets' labels double-chechedl
os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\TI-Data Collection')


# Keywords: all traffic-related words. Both for label 2 (incident) and label 3 (traffic condition)
with open('All-TI-Keywords-Weights.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    keywords = []
    for row in reader:
        keywords.append(row[0])

with open('DataCollection-TI_IUs_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    TI_Tweets = []
    for row in reader:
        TI_Tweets.append(row)

# Words related to traffic condition to be labeled as 3.
TrafficCondition_words = ['open', 'opened', 'reopen', 'reopened', 're-open',
                          'opening', 'heavy', 'slow', 'jammed']
# words related to clear to be labeled as 3.
Clear_words = ['clear', 'cleared', 'clearance']

vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()
A = 'ELLIOTT HIGHWAY from STEESE EXPRESSWAY/HIGHWAY to MILEPOST '
if any(word in analyze(A) for word in TrafficCondition_words):
    print('yes')
for i in range(len(TI_Tweets)):
    if any(word in analyze(TI_Tweets[i][2]) for word in keywords):
        if any(word in analyze(TI_Tweets[i][2]) for word in Clear_words):
            TI_Tweets[i][0] = '3'
        elif any(word in analyze(TI_Tweets[i][2]) for word in TrafficCondition_words):
            TI_Tweets[i][0] = '4'
        else:
            TI_Tweets[i][0] = '2'
    else:
        TI_Tweets[i][0] = '1'

with open('Remain-Labeled_DataCollection-TI_IUs_Tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in TI_Tweets:
        writer.writerow(tweet)
