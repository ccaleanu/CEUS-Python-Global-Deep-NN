# -*- coding: utf-8 -*-
# ResNet50.py
'''
Implementation of the ResNet50 architecture.
Structure: INPUT --> ...
'''
import tensorflow as tf
import config
from tensorflow.keras import layers

class ResNet50:
    '''
    ResNet50 Architecture implemented using tf.keras.applications
    '''

    @staticmethod
    def build(num_classes:int):
        
        '''
        Build the ResNet50 architecture given the corresponding
        number of classes of the data.
        
        parameters
        ----------
            num_classes: number of classes of the corresponding data.

        returns
        -------
            model: the ResNet50.py model compatible with given inputs
        '''
        # initialize model
        print("[INFO] preparing model...")

        # Create a model that includes the augmentation stage
        img_height = 180
        img_width = 180
        depth = 3
        input_shape = (img_height, img_width, depth)
        inputs = tf.keras.Input(shape=input_shape)

        # augment images
        data_augmentation = tf.keras.Sequential(
        [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(config.img_height, config.img_width, config.depth)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.Rescaling(1./255)
        ]
        )

        x = data_augmentation(inputs)
        # !? preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        
        # load the ResNet-50 network, ensuring the head FC layer sets are left off
        baseModel = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        baseModel.trainable = False
        #baseModel.summary()
            
        # construct the head of the model that will be placed on top of the the base model
        headModel = baseModel.output

        headModel = tf.keras.layers.GlobalAveragePooling2D()(headModel)
        '''
        headModel = tf.keras.layers.AveragePooling2D(pool_size=(6, 6))(headModel)
        headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
        headModel = tf.keras.layers.Dense(256, activation="relu")(headModel)
        '''
        
        headModel = tf.keras.layers.Dropout(0.5)(headModel)
        headModel = tf.keras.layers.Dense(num_classes, activation="softmax")(headModel)
        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)

        return model