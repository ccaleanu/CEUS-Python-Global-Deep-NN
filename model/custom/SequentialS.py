# -*- coding: utf-8 -*-
# SequentialS.py
'''
Implementation of a custom Seqential CNN 12 layers architecture.
INPUT --> AUG --> CONV --> POOL --> CONV --> POOL --> CONV --> POOL--> DROP --> FLAT --> FC --> FC
'''
import tensorflow as tf
import config
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


class SequentialS:
    '''
    Custom Seqential CNN 12 layers architecture
    '''

    @staticmethod
    def build(num_classes:int):
        '''
        Build the Seq12 architecture given width, height and depth
        as dimensions of the input tensor and the corresponding
        number of classes of the data.

        parameters
        ----------
            num_classes:  output size
        
        returns
        -------
            model: the Seq12 model compatible with given inputs
                    as a keras sequential model.
        '''
        # initialize model
        print("[INFO] preparing model...")

        # augment images
        data_augmentation = tf.keras.Sequential(
        [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(config.img_height, config.img_width, config.depth)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.Rescaling(1./255)
        ]
        )
        
        model = Sequential(name='SeqS')
        model.add(data_augmentation)
        model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.PReLU())
        model.add(layers.MaxPooling2D())

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.PReLU())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax'))

        # model.summary()
        
        # return the constructed network architecture
        return model