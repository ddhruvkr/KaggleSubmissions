import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.models import Model
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

class VGG_truncated:
    def __init__(self):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]
        self.model = self.build_model()

    def build_model(self):
        #smaller version of VGG
        #was not able to train completely on a CPU
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
        model.add(Dropout(0.4))

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

        model.add(GlobalMaxPooling2D())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model

class VGG16Keras_fast:
    def __init__(self):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [48, 48, 3]
        self.model = self.build_model()
        self.base_model = self.build_base_model()

    def build_base_model(self):
        base_model = VGG16(weights = 'imagenet', include_top=False, input_shape=self.x_shape)
        print ('base model summary')
        base_model.summary()
        return base_model

    def build_model(self):                                                                                                                                                                                                                                                                                                                                              
        model = Sequential()
        #model.add(GlobalAveragePooling2D(input_shape=[1,1,512]))
        model.add(GlobalMaxPooling2D(input_shape=[None,None,512]))                                                                                                        
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
    def __init__(self):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [48, 48, 3]
        self.model = self.build_model()

    def build_model(self):
        base_model = VGG16(weights = 'imagenet', include_top=False, input_shape=self.x_shape)
        print ('base model summary')
        base_model.summary()
                                                                                                                                                                                                                                                                                                                                              
        top_model = Sequential()
        #model.add(GlobalAveragePooling2D(input_shape=[1,1,512]))
        top_model.add(GlobalMaxPooling2D(input_shape=[None,None,512]))                                                                                                        
        #model.add(Flatten(input_shape=[1,1,512]))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(BatchNormalization())
        top_model.add(Dropout(0.5))
        '''model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))'''
        top_model.add(Dense(self.num_classes, activation='softmax'))
        top_model.load_weights("VGG16.h5")
        print(' top model summary')
        top_model.summary()
        model = Model(inputs= base_model.input, outputs= top_model(base_model.output))
        print('model summary')
        model.summary()
        return model

class ResNet50Keras_fast:
    def __init__(self):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [224,224, 3]
        self.model = self.build_model()
        self.base_model = self.build_base_model()

    def build_base_model(self):
        base_model = ResNet50(weights = 'imagenet', include_top=False, input_shape=self.x_shape)
        #print ('base model summary')
        #base_model.summary()
        return base_model

    def build_model(self):                                                                                                                                                                                                                                                                                                                                              
        model = Sequential()
        #model.add(GlobalAveragePooling2D(input_shape=[1,1,2048]))
        model.add(GlobalMaxPooling2D(input_shape=[None,None,2048]))                                                                                                        
        #model.add(Flatten(input_shape=[1,1,2048]))
        model.add(Dropout(0.4))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        '''model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))'''
        model.add(Dense(self.num_classes, activation='softmax'))
        #model.load_weights("Resnet50.h5")
        #print('model simmary')
        #model.summary()
        return model

class ResNet50Keras_fast_unfrozen:
    def __init__(self):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [224, 224, 3]
        self.model, self.base_model, self.top_model = self.build_model()

    def build_model(self):
        base_model = ResNet50(weights = 'imagenet', include_top=False, input_shape=self.x_shape)
        #print ('base model summary')
        #base_model.summary()
        for layer in base_model.layers:
            layer.trainable = False
        '''mid_start = base_model.get_layer('activation_40')
        all_layers = base_model.layers
        for i in range(base_model.layers.index(mid_start)):
            #print(i)
            all_layers[i].trainable = False'''

       
        top_model = Sequential()
        #model.add(GlobalAveragePooling2D(input_shape=[1,1,512]))
        top_model.add(GlobalMaxPooling2D(input_shape=[None,None,2048]))                                                                                                        
        #model.add(Flatten(input_shape=[1,1,512]))
        top_model.add(Dropout(0.4))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(BatchNormalization())
        top_model.add(Dropout(0.5))

        top_model.add(Dense(self.num_classes, activation='softmax'))
        top_model.load_weights("Resnet50.h5")
        #print(' top model summary')
        #top_model.summary()

        model = Model(inputs = base_model.input, outputs=top_model(base_model.output))

        '''top_model = base_model

        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.4)(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # This is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)'''
        '''mid_start = model.get_layer('activation_40')
        all_layers = model.layers
        for i in range(model.layers.index(mid_start)):
            all_layers[i].trainable = False'''
        #print('final model summary')
        #model.summary()
        return model, base_model, top_model