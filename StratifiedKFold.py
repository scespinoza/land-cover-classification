# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 21:06:01 2019

@author: secan
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from indian_pines import IndianPines
from LULC_CNN_KERAS import LULC_CNN
from DataAugmentation import rotate

# Input
P = 5
B = 220
NCLASSES = 16
# Folds
K = 3
# Random Forest
N_ESTIMATORS = 100
# SVM
C = 50
COEF = 0.01

BATCH_SIZE = 16
EPOCHS = 50

data_augmentation = False
angle = 180

def class_accuracy(y_true, y_pred):
    class_acc = []
    for c in range(NCLASSES):
        
        c_idx = (y_true == c)
        
        acc = np.mean(y_true[c_idx] == y_pred[c_idx])
        class_acc.append(acc)
        
    return class_acc


def load_data():
    ip = IndianPines()
    X, y = ip.get_dataset()

    return X, y, ip.class_df
   

    
X, y, class_df = load_data()

mask = (y != 0)
X = X[mask.ravel()]
y = y[mask.ravel()] - 1

if data_augmentation:
    X, y = rotate(X, y, 45)

X_vol = X.copy()    
X = X.reshape(X.shape[0], np.prod(X.shape[1:]))

enc = OneHotEncoder(sparse=False)
y_enc = enc.fit_transform(y)

skf = StratifiedKFold(n_splits=K, shuffle=True)

f = 1

accuracy_dict = {'rf': [], 'svm': [], 'cnn': []}
time_dict = {'rf': [], 'svm': [], 'cnn': []}
class_accuracy_dict = {'rf': [], 'svm': [], 'cnn': []}

for train_index, test_index in skf.split(X, y):
    
    
    
    
    print('#' * 50)
    print('# Fold {}'.format(f))
    print('#' * 50)
    #########################################################
    # Random Forest
    #########################################################
    print('Random Forest Classifier:')
    
    start_time = time.time()
    
    rfc = RandomForestClassifier(n_estimators=N_ESTIMATORS, verbose=1, n_jobs=-1)
    
    end_time = time.time()
    
    time_dict['rf'].append(end_time - start_time)
    rfc.fit(X_vol[train_index, 2, 2, :], y[train_index].ravel())
    y_pred = rfc.predict(X_vol[test_index, 2, 2, :])
    rf_acc = accuracy_score(y[test_index].ravel(), y_pred)
    rf_class_acc = class_accuracy(y[test_index].ravel(), y_pred)
    print('Random Forest Accuracy: ', rf_acc)
    accuracy_dict['rf'].append(rf_acc)
    
    for i in range(NCLASSES):
        print('{} Accuracy: {}'.format(class_df['class_name'][i + 1], rf_class_acc[i]))
        
    class_accuracy_dict['rf'].append(rf_class_acc)
    
    print('Finished Training Random Forest. Elapsed Time: {:.2f}s'.format(end_time - start_time))
    
    #########################################################
    # Support Vector Machine
    #########################################################
    
    print('-' * 50)
    print('Support Vector Machine: ')
    print('-' * 50)
    
    svc = SVC(C=C, coef0=COEF)
    start_time = time.time()
    svc.fit(X_vol[train_index, 2, 2, :], y[train_index].ravel())
    end_time = time.time()
    y_pred = svc.predict(X_vol[test_index, 2, 2, :])
    svm_acc = accuracy_score(y[test_index].ravel(), y_pred)
    svm_class_acc = class_accuracy(y[test_index].ravel(), y_pred)
    
    print('Support Vector Machine Accuracy: ', svm_acc)
    
    
    time_dict['svm'].append(end_time - start_time)
    accuracy_dict['svm'].append(svm_acc)
    class_accuracy_dict['svm'].append(svm_class_acc)
    
    
    for i in range(NCLASSES):
        print('{} Accuracy: {}'.format(class_df['class_name'][i + 1], svm_class_acc[i]))
        
    print('Finished Training SVM. Elapsed Time: {:.2f}s'.format(end_time - start_time))
    #########################################################
    # Convolutional Neural Network
    #########################################################
    
    print('-' * 50)
    print('Convolutional Neural Net: ')
    print('-' * 50)
    
    
    logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    print('Saving history to:{}'.format(logdir))
    cnn = LULC_CNN(num_classes=NCLASSES)
    start_time = time.time()
    print('Start Training.')
    train_history = cnn.fit(X_vol[train_index], y_enc[train_index], 
                            validation_data = (X_vol[test_index], y_enc[test_index]),
                            batch_size=BATCH_SIZE, epochs=EPOCHS, 
                            callbacks=[tensorboard_callback], verbose=1)
    end_time=time.time()
    print("Average test loss: ", np.average(train_history.history['loss']))
    y_hat = cnn.predict(X_vol[test_index])
    y_pred = np.argmax(y_hat, axis=1)
    
    cnn_acc = accuracy_score(y[test_index].ravel(), y_pred)
    cnn_class_acc = class_accuracy(y[test_index].ravel(), y_pred)
    
    print('Convolutional Neural Net Accuracy: ', cnn_acc)
    for i in range(NCLASSES):
        print('{} Accuracy: {}'.format(class_df['class_name'][i + 1], cnn_class_acc[i]))
    
    time_dict['cnn'].append(end_time - start_time)
    accuracy_dict['cnn'].append(cnn_acc)
    class_accuracy_dict['cnn'].append(cnn_class_acc)
    
    cnn.save('models/cnn_{}.h5'.format(f))
    print('Finished Training CNN. Elapsed Time: {:.2f}s'.format(end_time - start_time))
    
    f += 1
    
    

    
pd.DataFrame(accuracy_dict).to_csv('output/fold_accuracy.csv')

class_accuracy = {}
for model in class_accuracy_dict:
    class_accuracy[model] = np.array(class_accuracy_dict[model]).mean(0)

pd.DataFrame(class_accuracy).to_csv('output/class_accuracy.csv')
pd.DataFrame(time_dict).to_csv('output/fold_time.csv')
    
    