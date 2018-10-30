import models
from inputs import get_processed_data, get_validation_data, upscale_images
from test import predict, predict_fast, predict_fast_unfrozen
import numpy as np
import os.path
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

def train(model, x_train, y_train, x_test):

    # training parameters
    batch_size = 128
    maxepoches = 5

    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 15

    x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    #datagen.fit(x_train)

    # optimization details

    # maybe move optimizer and loss part to the models section?
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training process in a for loop with learning rate drop every 25 epoches.

    historytemp = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_validation, y_validation), epochs=maxepoches, verbose=1)
    '''historytemp = model.fit_generator(datagen.flow(x_train, y_train, 
        batch_size=batch_size), steps_per_epoch = x_train.shape[0], 
        validation_data=(x_validation, y_validation), callbacks=[reduce_lr], epochs=maxepoches)'''
    model.save_weights('vgg16keras.h5')
    return model




def different_train(model, base_model, x_train, y_train, x_test):

    # training parameters
    batch_size = 128
    maxepoches = 100

    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 20

    x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    print ("creating bottleneck features for training data")
    #Creating bottleneck features for the testing data

    if os.path.isfile("Inceptionv3_features_train.npz"):
        train_features = data = np.load('Inceptionv3_features_train.npz')
        print("loaded train features")
        #print(train_features['features'])
        #print(train_features.shape)

    else:
        train_features = base_model.predict(x_train, verbose=1)
        #Saving the bottleneck features
        np.savez('Inceptionv3_features_train', features=train_features)
    
    if os.path.isfile("Inceptionv3_features_test.npz"):
        test_features = data = np.load('Inceptionv3_features_test.npz')
        print("loaded test features")
        #print(test_features.shape)
    else:
        test_features = base_model.predict(x_test, verbose=1)
        #Saving the bottleneck features
        np.savez('Inceptionv3_features_test', features=test_features)
    
    if os.path.isfile("Inceptionv3_features_validation.npz"):
        validation_features = data = np.load('Inceptionv3_features_validation.npz')
        print("loaded validation features")
        #print(validation_features['features'])
    else:
        validation_features = base_model.predict(x_validation, verbose=1)
        #Saving the bottleneck features
        np.savez('Inceptionv3_features_validation', features=validation_features)

    # maybe move optimizer and loss part to the models section?
    # optimization details
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training process in a for loop with learning rate drop every 25 epoches.


    #put validation features too
    historytemp = model.fit(train_features['features'], y_train, validation_data=(validation_features['features'], y_validation), batch_size=batch_size, epochs=maxepoches, verbose=1, callbacks=[reduce_lr])
    '''historytemp = model.fit_generator(datagen.flow(x_train, y_train, 
        batch_size=batch_size), steps_per_epoch = x_train.shape[0], 
        validation_data=(x_validation, y_validation), callbacks=[reduce_lr], epochs=maxepoches)'''
    model.save_weights('vgg16keras_fast.h5')
    predict_fast(model, x_test, test_features['features'])
    print('prediction done')
    #Creating bottleneck features for the testing data
    #test_features = base_model.predict(x_test)

    #Saving the bottleneck features
    #np.savez('vgg16_features_test', features=test_features)
    # optimization details
    return model



def different_train_unfrozen(model, x_train, y_train, x_test):

    # training parameters
    batch_size = 128
    maxepoches = 100

    learning_rate = 0.001
    lr_decay = 1e-6
    lr_drop = 30

    x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    # maybe move optimizer and loss part to the models section?
    # optimization details
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training process in a for loop with learning rate drop every 25 epoches.


    #put validation features too
    historytemp = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=batch_size, epochs=maxepoches, verbose=1, callbacks=[reduce_lr])
    '''historytemp = model.fit_generator(datagen.flow(x_train, y_train, 
        batch_size=batch_size), steps_per_epoch = x_train.shape[0], 
        validation_data=(x_validation, y_validation), callbacks=[reduce_lr], epochs=maxepoches)'''
    model.save_weights('vgg16keras_fast_unnfrozen.h5')
    predict_fast_unfrozen(model, x_test)
    print('prediction done')
    #Creating bottleneck features for the testing data
    #test_features = base_model.predict(x_test)

    #Saving the bottleneck features
    #np.savez('vgg16_features_test', features=test_features)
    # optimization details
    return model



def get_data():
    return get_processed_data()

if __name__ == '__main__':

    size = 48
    train_data, train_label, test_data = get_data()
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data, test_data = upscale_images(train_data, test_data, size)
    #model = models.cifar100vgg().model
    #model = models.InceptionV3Keras().model
    #model = models.VGG16Keras().model
    obj = models.VGG16Keras_fast_unfrozen()
    model = obj.model
    #base_model = obj.base_model
    different_train_unfrozen(model, train_data, train_label, test_data)
    #train(model, train_data, train_label, test_data)

    #predict(model, test_data)
