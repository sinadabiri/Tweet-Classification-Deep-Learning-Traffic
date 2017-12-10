import numpy as np
import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors

# Parse data. In this case there is no difference between 2-class and 3-class dataset
filename = '../LSTM-CNN code/1_TrainingSet_2Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    def clean(sentence):
        # cleaning process to remove any punctuation, parentheses, question marks. This leaves only alphanumeric characters.
        remove_special_chars = re.compile("[^A-Za-z0-9 ]+")
        return re.sub(remove_special_chars, "", sentence.lower())
    reader = csv.reader(f)
    training_tweets = []
    for tweet in reader:
        tweet[2] = clean(tweet[2])
        training_tweets.append(tweet)

# Find the distribution of number of words in tweets
vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()

# word2vec model from twitter based on 400 m tweets. Final results is for 3039345 m words with 400-dimension vector
Word2Vec_model_T = KeyedVectors.load_word2vec_format('word2vec_twitter_model.bin', binary=True, encoding='latin-1')
# word2vec model from pre-trained Google model with 1 m words and 300-dimension vector
Word2Vec_model_G = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

numberofwords_T = []  # for twitter
numberofwords_G = []  # for Google
for tweet in training_tweets:
    words_seq = analyze(tweet[2])
    count = 0
    for word in words_seq:
        try:
            a = Word2Vec_model_T[word]
            count += 1
        except KeyError:
            pass
    numberofwords_T.append(count)

for tweet in training_tweets:
    words_seq = analyze(tweet[2])
    count = 0
    for word in words_seq:
        try:
            a = Word2Vec_model_G[word]
            count += 1
        except KeyError:
            pass
    numberofwords_G.append(count)
# Histogram of number of words per tweet
labels = [['Google_word2vec', numberofwords_G], ['Twitter_word2vec', numberofwords_T]]
for i, wor2vec in enumerate(labels):
    plt.figure(i)
    plt.rcParams['font.family'] = ['serif']  # default is sans-serif
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    bin = np.linspace(np.min(wor2vec[1]), np.max(wor2vec[1]), np.max(wor2vec[1]) - np.min(wor2vec[1]) + 1)
    plt.hist(wor2vec[1], bins=bin, histtype='bar', color='dimgray')
    #plt.hist(numberofwords_G, bins=np.linspace(0, 22, 23), histtype='step', color='r', label='Google word2vec')
    #plt.legend(prop={'size': 10})
    plt.xticks(np.arange(0, max(max(numberofwords_G), max(numberofwords_T)) + 1, 1.0))
    plt.ylabel('Frequency')
    plt.xlabel('Number of words in a tweet')
    filename = 'Histogram_' + wor2vec[0] +'.png'
    plt.savefig(filename, dpi=1200)
    plt.show()


# Summary statistics on number of words in each tweet.
print('Mean of number of words for Google is {} and Twitter is {}: '.
      format(np.mean(numberofwords_G), np.mean(numberofwords_T)))
print('Median of number of words for Google is {} and Twitter is {}: '.
      format(np.median(numberofwords_G), np.median(numberofwords_T)))
print('90 quantile of number of words for Google is {} and Twitter is {}: '.
      format(np.percentile(numberofwords_G, 90), np.percentile(numberofwords_T, 90)))
print('Max of number of words for Google is {} and Twitter is {}: '.
      format(np.max(numberofwords_G), np.max(numberofwords_T)))
print('Min of number of words for Google is {} and Twitter is {}: '.
      format(np.min(numberofwords_G), np.min(numberofwords_T)))

# Find average number of missing words in word2vec models.
num_miss_words = []
for tweet in training_tweets:
    words_seq = analyze(tweet[1])
    count = 0
    for word in words_seq:
        try:
            Word2Vec_model_G[word]  # or Word2Vec_model_T[word]
        except KeyError:
            count += 1
    num_miss_words.append(count)
ave_num_miss_words = sum(num_miss_words) / len(num_miss_words)
print('Average number of missing words: ', ave_num_miss_words)

