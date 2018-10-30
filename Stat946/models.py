import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.models import Model
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3

class cifar100vgg:
    def __init__(self, train=False):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]
        self.model = self.build_model()
        if train is not True:
            self.model.load_weights('cifar100vgg.h5')

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model







        '''model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        return model'''

class VGG19Keras:
    def __init__(self, train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [48, 48, 3]
        self.model = self.build_model()

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        base_model = VGG19(#weights='imagenet',
        weights = 'imagenet', include_top=True, input_shape=self.x_shape)
        base_model.summary()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # This is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # First: train only the top layers (which were initialized from imagenet)
        for layer in base_model.layers:
            layer.trainable = False
        model.summary()
        return model

class VGG16Keras:
    def __init__(self, train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [48, 48, 3]
        self.model = self.build_model()

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        base_model = VGG16(#weights='imagenet',
        weights = 'imagenet', include_top=False, input_shape=self.x_shape)
        base_model.summary()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # This is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # First: train only the top layers (which were initialized from imagenet)
        for layer in base_model.layers:
            layer.trainable = False
        model.summary()
        return model

class VGG16Keras_fast:
    def __init__(self, train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [48, 48, 3]
        self.model = self.build_model()
        self.base_model = self.build_base_model()

    def build_base_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        base_model = VGG16(#weights='imagenet',
        weights = 'imagenet', include_top=False, input_shape=self.x_shape)
        print ('base model summary')
        base_model.summary()
        return base_model

    def build_model(self):                                                                                                                                                                                                                                                                                                                                              
        model = Sequential()
        #model.add(GlobalAveragePooling2D(input_shape=[1,1,512]))
        model.add(GlobalMaxPooling2D(input_shape=[1,1,512]))                                                                                                        
        #model.add(Flatten(input_shape=[1,1,512]))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        '''model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))'''
        model.add(Dense(self.num_classes, activation='softmax'))
        print('model simmary')
        model.summary()
        return model


class VGG16Keras_fast_unfrozen:
    def __init__(self, train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [48, 48, 3]
        self.model = self.build_model()

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        base_model = VGG16(#weights='imagenet',
        weights = 'imagenet', include_top=False, input_shape=self.x_shape)
        print ('base model summary')
        base_model.summary()
                                                                                                                                                                                                                                                                                                                                              
        top_model = Sequential()
        #model.add(GlobalAveragePooling2D(input_shape=[1,1,512]))
        top_model.add(GlobalMaxPooling2D(input_shape=[1,1,512]))                                                                                                        
        #model.add(Flatten(input_shape=[1,1,512]))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(BatchNormalization())
        top_model.add(Dropout(0.5))
        '''model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))'''
        top_model.add(Dense(self.num_classes, activation='softmax'))
        top_model.load_weights("vgg16keras_fast.h5")
        print(' top model summary')
        top_model.summary()
        '''model = Sequential() #new model
        for layer in base_model.layers: 
            model.add(layer)
        model.add(top_model)'''
        model = Model(inputs= base_model.input, outputs= top_model(base_model.output))
        model.summary()
        mid_start = model.get_layer('block4_pool')
        all_layers = model.layers
        for i in range(model.layers.index(mid_start)):
            all_layers[i].trainable = False
        #for layer in model.layers[:25]:
        #    layer.trainable = False
        print('model summary')
        model.summary()
        return model

class InceptionV3Keras:
    def __init__(self, train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [139, 139, 3]
        self.model = self.build_model()

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        base_model = InceptionV3(#weights='imagenet',
        weights = 'imagenet', include_top=False, input_shape=self.x_shape)
        base_model.summary()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # First: train only the top layers (which were initialized from imagenet)
        for layer in base_model.layers:
            layer.trainable = False
        model.summary()
        return model


class InceptionV3Keras_fast:
    def __init__(self, train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [139, 139, 3]
        self.model = self.build_model()
        self.base_model = self.build_base_model()

    def build_base_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        base_model = InceptionV3(#weights='imagenet',
        weights = 'imagenet', include_top=False, input_shape=self.x_shape)
        print ('base model summary')
        base_model.summary()
        return base_model

    def build_model(self):                                                                                                                                                                                                                                                                                                                                              
        model = Sequential()
        #model.add(GlobalAveragePooling2D(input_shape=[1,1,512]))
        model.add(GlobalMaxPooling2D(input_shape=[1,1,512]))                                                                                                        
        #model.add(Flatten(input_shape=[1,1,512]))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        '''model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))'''
        model.add(Dense(self.num_classes, activation='softmax'))
        print('model simmary')
        model.summary()
        return model

