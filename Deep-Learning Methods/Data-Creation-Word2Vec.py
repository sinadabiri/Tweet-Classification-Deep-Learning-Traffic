import numpy as np
import os
import logging
from gensim.models.keyedvectors import KeyedVectors
import warnings
import csv
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import random
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class data_creation_word2vec():
    @staticmethod
    def clean(sentence):
        # cleaning process to remove any punctuation, parentheses, question marks. This leaves only alphanumeric characters.
        remove_special_chars = re.compile("[^A-Za-z0-9 ]+")
        return re.sub(remove_special_chars, "", sentence.lower())

    def __init__(self, filename_train, filename_test, word2vec_type, classdataset):
        self.classdataset = classdataset
        self.word2vec_type = word2vec_type  # 'Google' or 'Twitter', 'random'
        # Parse train data
        filename = '../LSTM-CNN code/' + filename_train
        with open(filename, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            training_tweets = []
            for tweet in reader:
                tweet[2] = data_creation_word2vec.clean(tweet[2])
                training_tweets.append(tweet)
        self.training_tweets = training_tweets
        # Parse test data
        filename = '../LSTM-CNN code/' + filename_test
        with open(filename, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            test_tweets = []
            for tweet in reader:
                tweet[2] = data_creation_word2vec.clean(tweet[2])
                test_tweets.append(tweet)
        self.test_tweets = test_tweets

        if word2vec_type == 'Twitter':
            # word2vec model from twitter based on 400 m tweets. Final results is for 3039345 m words with
            # 400-dimension vector
            self.Word2Vec_model = KeyedVectors.load_word2vec_format('word2vec_twitter_model.bin', binary=True,
                                                                    encoding='latin-1')
        if word2vec_type == 'Google' or word2vec_type == 'random':
            # word2vec model from pre-trained Google model with 1 m words and 300-dimension vector
            self.Word2Vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def data_creation(self):
        if self.word2vec_type == "Twitter" or self.word2vec_type == 'Google':
            vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
            analyze = vectorizer.build_analyzer()
            # Create Train_Y (i.e., labels)
            Train_Y = [int(tweet[0]) for tweet in self.training_tweets]

            # Create (Lxd) word embedding matrix corresponding to each tweet. L: number of words in tweet,
            # d: word vector dimension
            d = self.Word2Vec_model.vector_size
            if d == 300:  # means we are using Google word2vec
                L = 12  # 90 percentile value of number of words in a tweet based on Google
            else:
                L = 13  # 90 percentile value of number of words in a tweet based on Twitter

            Train_X = np.zeros((len(self.training_tweets), L, d), dtype=np.float32)
            for i in range(len(self.training_tweets)):
                words_seq = analyze(self.training_tweets[i][2])
                index = 0
                for word in words_seq:
                    if index < L:
                        try:
                            Train_X[i, index, :] = self.Word2Vec_model[word]
                            index += 1
                        except KeyError:
                            pass
                    else:
                        break


            Test_Y_ori = [int(tweet[0]) for tweet in self.test_tweets]
            Test_X = np.zeros((len(self.test_tweets), L, d), dtype=np.float32)
            for i in range(len(self.test_tweets)):
                words_seq = analyze(self.test_tweets[i][2])
                index = 0
                for word in words_seq:
                    if index < L:
                        try:
                            Test_X[i, index, :] = self.Word2Vec_model[word]
                            index += 1
                        except KeyError:
                            pass
                    else:
                        break
            filename = '1_Word2Vec_' + self.word2vec_type + '_' + self.classdataset + '.pickle'
            with open(filename, 'wb') as f:
                pickle.dump([Train_X, Test_X, Train_Y, Test_Y_ori], f)
            print("{} word2vec matrix has been created as the input layer".format(self.word2vec_type))

        elif self.word2vec_type == 'random':
            # Create randomized word embedding matrix for test and train data
            L = 12  # the same as Google
            d = 300  # the same as Google
            Train_X = np.zeros((len(self.training_tweets), L, d), dtype=np.float32)
            max_val = np.amax(self.Word2Vec_model.syn0)
            min_val = np.amin(self.Word2Vec_model.syn0)
            for i in range(len(self.training_tweets)):
                Train_X[i, :, :] = min_val + (max_val - min_val) * np.random.rand(L, d)

            Test_X = np.zeros((len(self.test_tweets), L, d), dtype=np.float32)
            for i in range(len(self.test_tweets)):
                Test_X[i, :, :] = min_val + (max_val - min_val) * np.random.rand(L, d)

            Test_Y_ori = [int(tweet[0]) for tweet in self.test_tweets]
            Train_Y = [int(tweet[0]) for tweet in self.training_tweets]
            filename = '1_Word2Vec_' + self.word2vec_type + '_' + self.classdataset + '.pickle'
            with open(filename, 'wb') as f:
                pickle.dump([Train_X, Test_X, Train_Y, Test_Y_ori], f)

            print("{} word2vec matrix has been created as the input layer".format(self.word2vec_type))

if __name__ == '__main__':
    word2vec = data_creation_word2vec(filename_train='1_TrainingSet_2Class.csv', filename_test='1_TestSet_2Class.csv',
                                           word2vec_type='Google', classdataset='2class')

    data = word2vec.data_creation()

















