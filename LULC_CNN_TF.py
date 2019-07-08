# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 21:41:31 2019

@author: secan
"""

import tensorflow as tf
import numpy as np


def _weights(shape, name):
    init = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(init, name=name)

def _bias(shape, name):
    init=tf.constant(0.1, shape=shape)
    return tf.Variable(init, name=name)

def conv2d(x, W, b, strides=1, padding='SAME'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, stride=2):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')

def batch_normalization(x, phase):
    return tf.layers.batch_normalization(x, center=True, scale=True, training=phase)

def fully_connected(x, W, b):
    return tf.add(tf.matmul(x, W), b)


class LULC_CNN:
    
    def __init__(self, input_shape, n_classes, learning_rate=0.01, keep_prob=0.2, epochs=50, batch_size=16):
        
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.epochs = epochs
        self.batch_size = batch_size
        self.build_graph()
        
    def forward_pass(self):
        
        
        bn1 = batch_normalization(self.x, self.phase)
        
        with tf.name_scope('conv1'):
            _W = _weights([3, 3, self.input_shape[-1], 32], name='weights_conv1')
            _b = _bias([32], name='bias_conv1')
            conv1 = conv2d(bn1, _W, _b, padding='SAME')
        
        bn2 = batch_normalization(conv1, self.phase)
        
        # pool1
        mp1 = maxpool2d(bn2)
        
        with tf.name_scope('conv2'):
            _W = _weights([3, 3, 32, 64], name='weights_conv2')
            _b = _bias([64], name='bias_conv2')
            conv2 = conv2d(mp1, _W, _b, padding='SAME')
        
        mp2 = maxpool2d(conv2)
        
        # Fully Connected
        
        fc1 = tf.reshape(mp2, [-1, 2 * 2 * 64])
        
        with tf.name_scope('dense1'):
            _W = _weights([2 * 2 * 64, 1024], name='weights_dense1')
            _b = _bias([1024], name='bias_dense1')
            fc1 = fully_connected(fc1, _W, _b)
        
        fc1 = tf.nn.dropout(fc1, rate=1 - self.keep_prob)
        
        with tf.name_scope('output'):
            _W = _weights([1024, self.n_classes], name='weights_output')
            _b = _bias([self.n_classes], name='bias_output')
            out = fully_connected(fc1, _W, _b)
        
        return out
    
    def build_network(self):
        
        with tf.name_scope('placeholder'):
            self.x = tf.placeholder(tf.float32, shape=[None, *self.input_shape], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y')
            self.phase = tf.placeholder(tf.bool, name='phase')
        
        self.y_hat = self.forward_pass()
        
        
        
        
    def _loss(self):
        with tf.name_scope('cost_function'):

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_hat, labels=self.y))
        return loss
    
    def _accuracy(self):
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.y_hat, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy
    
    def _optimizer(self):
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss())
    
    def build_graph(self):
        
        with tf.Graph().as_default() as g:
            
            self.build_network()
            self._optimizer()
            self.g = g          
                
            
    
    def fit(self, X, y):
       
        num_batches = X.shape[0] // self.batch_size
        self.trace_loss = []
        self.trace_accuracy = []
        
        with tf.Session(graph=self.g) as sess:
            sess.run(tf.global_variables_initializer())
            tf.summary.FileWriter("./graphs/cnn", self.g)
            print('Start Training.')
            for i in range(self.epochs):
                
                epoch_loss, epoch_acc = 0, 0
                print('Epoch {}'.format(i + 1))
                for j in range(num_batches):
                    print('-', end='')
                    
                    batch_x, batch_y = (X[j * self.batch_size: (j + 1)* self.batch_size],
                                        y[j * self.batch_size: (j + 1)* self.batch_size])
                    actual_batch_size = X[j*self.batch_size:(j+1)*self.batch_size].shape[0]
                    
                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.phase: True})
                    loss = sess.run(self._loss(), feed_dict={self.x: batch_x, self.y: batch_y, self.phase: False})
                    
                    epoch_loss += actual_batch_size * loss
                    
                    
                epoch_loss = epoch_loss / X.shape[0]
                epoch_acc = sess.run(self._accuracy(),  feed_dict={self.x: X, self.y: y, self.phase: False})
                
                self.trace_loss.append(epoch_loss)
                self.trace_accuracy.append(epoch_acc)
                print()
                
                print('\nEpoch: {0:d}, Average Loss: {1:.3f}, accuracy: {2:.3f}'.format((i + 1), epoch_loss, epoch_acc))
                
            print('-' * 50)
            print('Final Epoch Training Results: ', end=' ') 
            print('Average Loss: {0:.3f}'.format(epoch_loss), end=' ')
            print('Accuracy: {0:.3f}'.format(epoch_acc), end=' ')
                
    
    def predict(self, X): 
        y_fake = np.ones((X.shape[0],self.n_classes))
        with tf.Session(graph=self.g) as sess:
            y_pred = sess.run(self.y_hat, feed_dict={self.x:X, self.y: y_fake, self.phase: False})
        return np.argmax(y_pred, axis=1)
    
if __name__ == '__main__':
    p = 5
    b = 220
    nn = LULC_CNN([p, p, b], 17)

        