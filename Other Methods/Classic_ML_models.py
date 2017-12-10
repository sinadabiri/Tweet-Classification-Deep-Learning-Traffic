import numpy as np
import pickle
import os
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\LSTM-CNN code')

filename = '../LSTM-CNN code/OnlyFirstSet_Carvalho_2class.pickle'
with open(filename, mode='rb') as f:
    Train_X, Test_X, Train_Y, Test_Y = pickle.load(f, encoding='latin1')  # Also can use the encoding 'iso-8859-1'


# SVM classification, rbf
SVM_rbf = SVC(kernel='rbf')
SVM_rbf.fit(Train_X[:1000, :], Train_Y[:1000])
Prediction = SVM_rbf.predict(Test_X)
print('Accuracy score by SVM rbf: ', accuracy_score(Test_Y, Prediction) * 100)
#print(classification_report(Test_Y, Prediction))

# SVM classification, linear
SVM_linear = SVC(kernel='linear')
SVM_linear.fit(Train_X, Train_Y)
Prediction = SVM_linear.predict(Test_X)
print('Accuracy score by SVM linear: ', accuracy_score(Test_Y, Prediction) * 100)
#print(classification_report(Test_Y, Prediction))

# Naive Bayes
Naive = BernoulliNB()
Naive.fit(Train_X, Train_Y)
Prediction = Naive.predict(Test_X)
print('Accuracy score by Naive Bayes: ', accuracy_score(Test_Y, Prediction) * 100)
#print(classification_report(Test_Y, Prediction))