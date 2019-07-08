# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 23:42:17 2019

@author: secan
"""
import numpy as np
from indian_pines import IndianPines
from LULC_CNN_KERAS import LULC_CNN
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from DataAugmentation import rotate



P = 5
B = 220
N = 17
data_augmentation = False

cnn = LULC_CNN(num_classes=16)


X, y = IndianPines().get_dataset()

mask = (y != 0)
X = X[mask.ravel()]
y = y[mask.ravel()] - 1

def class_accuracy(y_true, y_pred):
    class_acc = []
    for c in range(16):
        
        c_idx = (y_true == c)
        
        acc = np.mean(y_true[c_idx] == y_pred[c_idx])
        class_acc.append(acc)
        
    return class_acc


if data_augmentation:
    X, y = rotate(X, y, 90)
    print('Finished Augmentation: {} samples'.format(X.shape[0]))
    
    
X_train, X_test, y_train, y_test = train_test_split(X.reshape(X.shape[0], 5 * 5 * 220), y)
m_train, m_test = X_train.shape[0], X_test.shape[0]
X_train = X_train.reshape(m_train, P, P, B)
X_test = X_test.reshape(m_test, P, P, B)

encoder = OneHotEncoder(sparse=False)
y_enc = encoder.fit_transform(y_train)
    
cnn.fit(X_train, y_enc, batch_size=16, epochs=50)

y_hat = cnn.predict(X_test)
y_pred = np.argmax(y_hat, axis=1)

print(class_accuracy(y, y_pred))