import models
from inputs import get_processed_data, upscale_images
from test import predict, predict_fast, predict_fast_unfrozen, save_results, load_predict_save, predict_fast_for_Model, predict_for_Model
import numpy as np
import os.path
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import math

def different_train(model, base_model, x_train, y_train, x_test, data_augment, learning_rate,
    maxepoches, lr_drop, file_path, train_features_file, test_features_file):

    # training parameters
    batch_size = 128
    #maxepoches = 250

    #learning_rate = 0.0001
    lr_decay = 1e-6
    #lr_drop = 20

    #x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    if data_augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            vertical_flip=False)

        generator = datagen.flow(  
            x=x_train,
            y=y_train,   
            batch_size=batch_size,
            shuffle=False)

        '''nb_train_samples = len(generator.filenames)  
        num_classes = len(generator.class_indices)''' 

        predict_size_train = int(math.ceil(x_train.shape[0] / batch_size))

        if os.path.isfile(train_features_file):
            train_features = np.load(train_features_file)
            print("loaded train features")

        else:
            train_features = base_model.predict_generator(generator, steps=predict_size_train, verbose=1)
            #Saving the bottleneck features
            np.savez(train_features_file, train_features)
            train_features = np.load(train_features_file)
            #datagen.fit(x_train)

    else:
        #x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
        print ("creating bottleneck features for training data")
        #Creating bottleneck features for the testing data

        if os.path.isfile(train_features_file):
            train_features = np.load(train_features_file)
            print("loaded train features")

        else:
            train_features = base_model.predict(x_train, verbose=1)
            #Saving the bottleneck features
            np.savez(train_features_file, features=train_features)
            train_features = np.load(train_features_file)
        
    if os.path.isfile(test_features_file):
        test_features = np.load(test_features_file)
        print("loaded test features")
 
    else:
        test_features = base_model.predict(x_test, verbose=1)
        np.savez(test_features_file, features=test_features)
        print("saved features")
        test_features = np.load(test_features_file)
    print("from bottleneck features")
    predict_fast(model, x_test, test_features['features'])

    '''check that is npz files created are different for different day'''
    #the two predicted values from pre calculated bottleneck features and the new bottleneck features
    #should be equal, cannot do this calculation here since gpu has issues with npz

    #try for val_loss
    checkpointer = ModelCheckpoint(filepath=file_path, 
        monitor='val_acc', verbose=1, save_best_only=True)

    # maybe move optimizer and loss part to the models section?
    # optimization details
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #when loading features from .npz files this works fine else not, don't know the reason for this
    #happenning on macbook not on desktop, most probably some version issue
    historytemp = model.fit(train_features['features'], y_train, validation_split=0.05, shuffle=True,
        batch_size=batch_size, epochs=maxepoches, verbose=1, callbacks=[reduce_lr, checkpointer])
    '''historytemp = model.fit_generator(datagen.flow(x_train, y_train, 
        batch_size=batch_size), steps_per_epoch = x_train.shape[0], 
        validation_data=(x_validation, y_validation), callbacks=[reduce_lr], epochs=maxepoches)'''

    model.load_weights(file_path)
    predict_fast(model, x_test, test_features['features'])
    print('prediction done')

    #return model



def different_train_unfrozen(model, x_train, y_train, x_test, learning_rate, maxepoches, lr_drop, file_path):

    # training parameters
    batch_size = 128
    #maxepoches = 30
    #learning_rate = 0.0001
    lr_decay = 1e-6
    #lr_drop = 25

    #x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    checkpointer = ModelCheckpoint(filepath=file_path, 
        monitor='val_acc', verbose=1, save_best_only=True)
    # TODO: move optimizer and loss part as functions in the models section
    # optimization details
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    historytemp = model.fit(x_train, y_train, validation_split=0.05, shuffle=True, batch_size=batch_size, 
        epochs=maxepoches, verbose=1, callbacks=[reduce_lr, checkpointer])
    
    '''historytemp = model.fit_generator(datagen.flow(x_train, y_train, 
        batch_size=batch_size), steps_per_epoch = x_train.shape[0], 
        validation_data=(x_validation, y_validation), callbacks=[reduce_lr], epochs=maxepoches)'''
    model.load_weights(file_path)
    predict_for_Model(model, x_test)
    print('prediction done')
    #return model


def get_data():
    return get_processed_data()

if __name__ == '__main__':

    train_data, train_label, test_data = get_data()
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')


    #VGG16

    '''size = 48
    train_data, test_data = upscale_images(train_data, test_data, size)
    VGG_obj = models.VGG16Keras_fast()
    model = VGG_obj.model
    base_model = VGG_obj.base_model
    different_train(model, base_model, train_data, train_label, test_data, False, 0.1,
        10, 5, "VGG16.h5", "VGG16_48_features_train.npz", "VGG16_48_features_test.npz")

    VGG_obj1 = models.VGG16Keras_fast_unfrozen()
    model1 = VGG_obj1.model
    mid_start = model1.get_layer('block4_pool')
    all_layers = model1.layers
    for i in range(model1.layers.index(mid_start)):
        all_layers[i].trainable = False
    predict_for_Model(model1, test_data)
    different_train_unfrozen(model1, train_data, train_label, test_data, 0.01,
        5, 1, "VGG16_48_keras_unfrozen_black_pool4.h5")

    VGG_obj2 = models.VGG16Keras_fast_unfrozen()
    model2 = VGG_obj2.model
    model2.load_weights("VGG16_48_keras_unfrozen_black_pool4.h5")
    mid_start = model2.get_layer('block3_pool')
    all_layers = model2.layers
    for i in range(model2.layers.index(mid_start)):
        all_layers[i].trainable = False
    predict_for_Model(model2, test_data)
    different_train_unfrozen(model2, train_data, train_label, test_data, 0.001,
        5, 1, "VGG16_48_keras_unfrozen_black_pool3.h5")'''



    #RESNET50
    '''size = 224
    train_data, test_data = upscale_images(train_data, test_data, size)
    ResNet_obj = models.ResNet50Keras_fast()
    ResNet_model = ResNet_obj.model
    ResNet_base_model = ResNet_obj.base_model
    different_train(ResNet_model,ResNet_base_model, train_data, train_label, test_data, False, 0.1,
        5, 5, "Resnet50.h5", "Resnet50_224_features_train.npz", "Resnet50_224_features_test.npz")

    #model.load_weights("Resnet50.h5")
    ResNet_obj1 = models.ResNet50Keras_fast_unfrozen()
    ResNet_model1 = ResNet_obj1.model
    ResNet_model1.summary()
    #model1.load_weights("Resnet50.h5")
    predict_for_Model(ResNet_model1, test_data)
    Resnet_mid_start = ResNet_model1.get_layer('activation_89')
    all_layers = ResNet_model1.layers
    for i in range(ResNet_model1.layers.index(Resnet_mid_start)):
        all_layers[i].trainable = False
    different_train_unfrozen(ResNet_model1, train_data, train_label, test_data, 0.01,
        3, 2, "Resnet50_224_keras_activation_40.h5")'''
    

    '''ResNet_obj2 = models.ResNet50Keras_fast_unfrozen()
    ResNet_model2 = ResNet_obj2.model
    ResNet_model2.summary()
    ResNet_model2.load_weights('Resnet50_224_keras_activation_40.h5')
    predict_for_Model(ResNet_model2, test_data)
    mid_start = ResNet_model2.get_layer('activation_31')
    all_layers = ResNet_model2.layers
    for i in range(ResNet_model2.layers.index(mid_start)):
        all_layers[i].trainable = False
    different_train_unfrozen(ResNet_model2, train_data, train_label, test_data, 0.001,
        3, 2, "Resnet50_224_keras_activation_31.h5")

    ResNet_obj3 = models.ResNet50Keras_fast_unfrozen()
    ResNet_model3 = ResNet_obj3.model
    ResNet_model3.summary()
    ResNet_model3.load_weights('Resnet50_224_keras_activation_31.h5')
    predict_for_Model(ResNet_model3, test_data)
    mid_start = ResNet_model3.get_layer('activation_22')
    all_layers = ResNet_model3.layers
    for i in range(ResNet_model3.layers.index(mid_start)):
        all_layers[i].trainable = False
    different_train_unfrozen(ResNet_model3, train_data, train_label, test_data, 0.0001,
        3, 2, "Resnet50_224_keras_activation_22.h5")'''

    ResNet_obj = models.ResNet50Keras_fast_unfrozen()
    ResNet_model = ResNet_obj.model
    ResNet_model.summary()
    ResNet_model.load_weights('Resnet50_224_keras_fast_checkpoint_acc_unfrozen_32_noval.h5')
    predict_for_Model(ResNet_model, test_data)
    
    '''
    # at compilation the weights get reset, not sure about this
    # https://github.com/keras-team/keras/issues/2379
    #integrated_model.load_weights("resnet50keras_fast_checkpoint_acc.h5")
    predict_fast_for_Model(integrated_model, test_data, test_features['features'])'''
