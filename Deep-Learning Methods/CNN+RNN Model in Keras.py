import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Conv1D, Flatten, MaxPooling2D, Lambda, Reshape
import pickle
from keras.optimizers import Adam
import os
from sklearn.metrics import confusion_matrix
import time
from keras.layers import LSTM
from keras import regularizers
from keras.backend import mean
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

start_time = time.clock()

# Parse data
filename = '../LSTM-CNN code/1_Word2Vec_Twitter_3class.pickle'
with open(filename, mode='rb') as f:
    Train_X, Test_X, Train_Y, Test_Y_ori = pickle.load(f, encoding='latin1')  # Also can use the encoding 'iso-8859-1'

# Construct and compile the LSTM model
NoClass = len(list(set(Train_Y)))

Train_Y = keras.utils.to_categorical(Train_Y, num_classes=NoClass)
Test_Y = keras.utils.to_categorical(Test_Y_ori, num_classes=NoClass)

L = len(Train_X[0, :, 0])
d = len(Train_X[0, 0, :])

model = Sequential()
model.add(Reshape((L, d, 1), input_shape=(L, d)))
model.add(Conv2D(100, (2, d), strides=(1, 1), padding='valid', activation='relu', use_bias=True))
output = model.output_shape
model.add(Reshape((output[1], output[3])))
model.add(Dropout(.25))
model.add(LSTM(100, return_sequences=False, activation='tanh', recurrent_activation='hard_sigmoid'))
model.add(Dropout(.50))
model.add(Dense(NoClass, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

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
