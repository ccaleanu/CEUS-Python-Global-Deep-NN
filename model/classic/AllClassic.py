# -*- coding: utf-8 -*-
# AllClassic.py
'''
Implementation of the AllClassic architecture.
'''
import tensorflow as tf
import config
from tensorflow.keras import layers

class AllClassic:
    '''
    AllClassic Architecture implemented using tf.keras.applications
    '''

    @staticmethod
    def build(num_classes:int):
        
        '''
        Build the AllClassic architecture given the corresponding
        number of classes of the data.
        
        parameters
        ----------
            num_classes: number of classes of the corresponding data.

        returns
        -------
            model: the AllClassic model compatible with given inputs
        '''
        # initialize model
        print("[INFO] preparing model...")

        # Create a model that includes the augmentation stage
        input_shape=(config.img_height, config.img_width, config.depth)
        inputs = tf.keras.Input(shape=input_shape)

        # augment images
        data_augmentation = tf.keras.Sequential(
        [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(config.img_height, config.img_width, config.depth)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1)
        ]
        )
        
        x = inputs

        if config.AUG:
             x = data_augmentation(x)        
        if config.myModelName == 'InceptionResNetV2':
            preproc = tf.keras.applications.inception_resnet_v2.preprocess_input
        if config.myModelName == 'Xception':
            preproc = tf.keras.applications.xception.preprocess_input
        if config.myModelName == 'MobileNet':
            preproc = tf.keras.applications.mobilenet.preprocess_input
        if config.myModelName == 'MobileNetV2':
            preproc = tf.keras.applications.mobilenet_v2.preprocess_input
        if config.myModelName == 'MobileNetV3Small':
            preproc = tf.keras.applications.mobilenet_v3.preprocess_input
        if (config.myModelName == 'NASNetMobile' or
            config.myModelName == 'NASNetLarge'):
            preproc = tf.keras.applications.nasnet.preprocess_input
        if (config.myModelName == 'EfficientNetB0' or
            config.myModelName == 'EfficientNetB1' or
            config.myModelName == 'EfficientNetB2' or
            config.myModelName == 'EfficientNetB3'):
            preproc = tf.keras.applications.efficientnet.preprocess_input
        if (config.myModelName == 'DenseNet121' or
            config.myModelName == 'DenseNet169' or
            config.myModelName == 'DenseNet201'):
            preproc = tf.keras.applications.densenet.preprocess_input
        if config.myModelName == 'ResNet50':
            preproc = tf.keras.applications.resnet.preprocess_input
        if config.myModelName == 'ResNet50V2':
            preproc = tf.keras.applications.resnet_v2.preprocess_input
        if config.myModelName == 'MobileNetV2':
            preproc = tf.keras.applications.mobilenet_v2.preprocess_input

#myModelName = 'EfficientNetB3'
        if config.PREPROC:
             x = preproc(x)

        # load the network, ensuring the head FC layer sets are left off
        class_bM = getattr(tf.keras.applications, config.myModelName)
        baseModel = class_bM(include_top=False, weights=config.weights, input_shape=input_shape)
        baseModel.trainable = config.trainable

        x = baseModel(x)
        headModel = tf.keras.layers.GlobalAveragePooling2D()(x)
        '''
        headModel = tf.keras.layers.AveragePooling2D(pool_size=(6, 6))(headModel)
        headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
        headModel = tf.keras.layers.Dense(256, activation="relu")(headModel)
        '''
        
        headModel = tf.keras.layers.Dropout(0.2)(headModel)
        headModel = tf.keras.layers.Dense(num_classes, activation="softmax")(headModel)
        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = tf.keras.Model(inputs, headModel)

        return model