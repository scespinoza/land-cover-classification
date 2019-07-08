import tensorflow as tf
from tensorflow.keras import layers, optimizers



def LULC_CNN(num_classes=17, p=5, b=220, name='IndianPines'):
    
    model = tf.keras.Sequential([

        layers.BatchNormalization(axis=-1, input_shape=(p, p, 220), trainable=True),

        layers.Convolution2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last'),

        layers.BatchNormalization(axis=-1),

        layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last'),

        layers.Convolution2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last'),

        layers.BatchNormalization(axis=-1, trainable=True),

        layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last'),

        layers.Flatten(),

        layers.Dense(1024),

        layers.Dropout(rate=0.2),

        layers.Dense(num_classes, activation='softmax')
        
    ], name=name)
    model.compile(optimizer=optimizers.SGD(lr=0.01, decay=1e-6),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    return model

def test_lulc_cnn(num_classes=16, p=5, b=220, name='IndianPines'):
    
    model = tf.keras.Sequential([
            layers.BatchNormalization(axis=-1, input_shape=(p, p, 220)),
            layers.Convolution2D(32, (3, 3), padding='same', activation='relu', data_format='channels_last'),
            layers.BatchNormalization(axis=-1),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last'),
            layers.Flatten(),
            layers.Dense(512),
            layers.Dropout(rate=0.2),
            layers.Dense(num_classes, activation='softmax')
        ], name=name)
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                    loss=tf.keras.losses.categorical_crossentropy,
                    metrics=[tf.keras.metrics.categorical_accuracy])
    return model
        
        
        
if __name__ == '__main__':
    print(LULC_CNN().summary())
