# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:58:00 2019

@author: secan
"""
#from LULC_CNN_TF import LULC_CNN as CNN
from indian_pines import IndianPines
#from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math



def rotate(X, y, angle=180):
    imgs = X.copy()
    X_aug = X.copy()
    y_aug = y.copy()
    mask = [y != 0]
    with tf.Session() as sess:
        
        for i in range(1, 360 // angle):
            print('Rotating {}° degrees'.format(i * angle))
            rotated_X = sess.run(tf.contrib.image.rotate(imgs[mask], math.radians(i * angle)))
            X_aug = np.append(X_aug, rotated_X, axis=0)
            y_aug = np.append(y_aug, y[mask], axis=0)
            
            
    return X_aug, y_aug

    
    


if __name__ == '__main__':
    
    ip = IndianPines()

    img = ip.get_patch(70, 80)
    X_aug, y_aug = rotate(img[np.newaxis, :], np.array([1]), 45)
    fig, ax = plt.subplots(2, 4)
    
    for i, axi in enumerate(ax.flatten()):
    
        axi.imshow(X_aug[i][:,  :, [29, 19, 9]])
        axi.set_title('{}° rotation'.format(i * 45))
        axi.set_axis_off()
    
    fig.suptitle('Data Augmentation')
    fig.savefig('figures/data-augmentation.png')
            










"""
enc = OneHotEncoder(sparse=False)

y_enc = enc.fit_transform(y)

X_try, y_try = X[:32], y_enc[:32]



p = 5
b = 220
n_classes = 17
cnn = CNN([p, p, b], n_classes, batch_size=3)
cnn.build_graph()
cnn.train(X_try, y_try)

"""