import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Reshape
import pickle
from keras.optimizers import Adam
import csv
from sklearn.metrics import confusion_matrix
import time
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
start_time = time.clock()

# Parse data
filename = '../LSTM-CNN code/1_Word2Vec_Twitter_3class.pickle'
with open(filename, mode='rb') as f:
    Train_X, Test_X, Train_Y, Test_Y_ori = pickle.load(f, encoding='latin1')  # Also can use the encoding 'iso-8859-1'

# Construct CNN model.
NoClass = len(list(set((Train_Y))))

Train_Y = keras.utils.to_categorical(Train_Y, num_classes=NoClass)
Test_Y = keras.utils.to_categorical(Test_Y_ori, num_classes=NoClass)

L = len(Train_X[0, :, 0])
d = len(Train_X[0, 0, :])

# Model and Compile
model = Sequential()
model.add(Reshape((L, d, 1), input_shape=(L, d)))

model.add(Conv2D(100, (2, d), strides=(1, 1), padding='valid', activation='relu', use_bias=True))

# model.add(Conv2D(100, (2, 1), strides=(1, 1), padding='valid', activation='relu', use_bias=True))

output = model.output_shape
model.add(MaxPooling2D(pool_size=(output[1], output[2])))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(NoClass, activation='softmax'))

# model.add(Dense(NoClass, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
# Optimizer and loss function
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ====================================================================================================
# Training, check point and save the best model
filepath = "weights.best2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
hist = model.fit(Train_X, Train_Y, epochs=20, batch_size=64, shuffle=False,
                            validation_data=(Test_X, Test_Y), callbacks=callbacks_list)

# Print the maximum acc_val and its corresponding epoch
index = np.argmax(hist.history['val_acc'])
print('\n')
print('The optimal epoch size: {}, The value of high accuracy {}'.format(hist.epoch[index], np.max(hist.history['val_acc'])))
print('\n')
print('Computation Time', time.clock() - start_time, "seconds")
print('\n')

# Save the history accuracy results.
with open('Accuracy-History-CNN-2class.pickle', 'wb') as f:
    pickle.dump([hist.epoch, hist.history['acc'], hist.history['val_acc']], f)

# =============================================================================================================
# load weights, predict based on the best model, compute accuracy, precision, recall, f-score, confusion matrix.
model.load_weights("weights.best2.hdf5")
# Compile model (required to make predictions)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Computer confusion matrix, precision, recall.
pred = model.predict(Test_X, batch_size=64)
y_pred = np.argmax(pred, axis=1)
Test_Y_ori = [int(item) for item in Test_Y_ori]
print('Test Accuracy %: ', len(np.where(y_pred == np.array(Test_Y_ori))[0])/len(Test_Y_ori) * 100)
print('\n')
print('Confusin matrix: ', confusion_matrix(Test_Y_ori, y_pred))
print('\n')
print(classification_report(Test_Y_ori, y_pred, digits=3))
# =====================================================================================================

# Qualitative assesment. Exploring miss-classified tweets.
# Exploring the tweets that classified wrongly
with open('1_TestSet_3Class.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    test_tweets = []
    for tweet in reader:
        test_tweets.append(tweet)

wrong_tweets_corpus = []

for i in range(len(Test_Y_ori)):
    wrong_tweet = []
    if y_pred[i] != Test_Y_ori[i]:
        wrong_tweet.append(pred[i])
        wrong_tweet.append(y_pred[i])
        wrong_tweet.append(test_tweets[i][0])
        wrong_tweet.append(test_tweets[i][1])
        wrong_tweet.append(test_tweets[i][2])
        wrong_tweets_corpus.append(wrong_tweet)

with open('Paper_wrong_classified_tweets_Twitter.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['probability', 'Predict', 'True', 'id', 'Text'])
    for tweet in wrong_tweets_corpus:
        writer.writerow(tweet)
