import models
from inputs import get_processed_data, upscale_images
from test import predict, predict_fast, predict_fast_unfrozen, save_results, load_predict_save, predict_fast_for_Model
import numpy as np
import os.path
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.datasets import cifar100
from keras.callbacks import ModelCheckpoint
import math

def train(model, x_train, y_train, x_test):

    # training parameters
    batch_size = 128
    maxepoches = 5

    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 15

    #x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
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

    historytemp = model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2, epochs=maxepoches, verbose=1)
    '''historytemp = model.fit_generator(datagen.flow(x_train, y_train, 
        batch_size=batch_size), steps_per_epoch = x_train.shape[0], 
        validation_data=(x_validation, y_validation), callbacks=[reduce_lr], epochs=maxepoches)'''
    model.save_weights('vgg16keras.h5')
    return model




def different_train(model, base_model, x_train, y_train, x_test, data_augment, learning_rate, maxepoches):

    # training parameters
    batch_size = 128
    #maxepoches = 250

    #learning_rate = 0.0001
    lr_decay = 1e-6
    lr_drop = 20
    (x_train1, y_train1), (x_test1, y_test) = cifar100.load_data(label_mode='fine')
    #x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
    def lr_scheduler(epoch):
        '''if epoch > 90:
            print('saving file')
            save_results(model, test_features, x_test, epoch)'''
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    if data_augment:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True, # randomly flip images
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest',
            vertical_flip=False)  # randomly flip images

        generator = datagen.flow(  
            x=x_train,
            y=y_train,   
            batch_size=batch_size,
            shuffle=False)

        '''nb_train_samples = len(generator.filenames)  
        num_classes = len(generator.class_indices)''' 

        predict_size_train = int(math.ceil(x_train.shape[0] / batch_size))

        if os.path.isfile("Resnet50_100_features_train.npz"):
            train_features = np.load('Resnet50_100_features_train.npz')
            print("loaded train features")
            #print(train_features['features'])
            #print(train_features.shape)

        else:
            train_features = base_model.predict_generator(generator, steps=predict_size_train, verbose=1)
            #Saving the bottleneck features
            np.savez('Resnet50_100_features_train', features=train_features)
            train_features = np.load('Resnet50_100_features_train.npz')
            # (std, mean, and principal components if ZCA whitening is applied).
            #datagen.fit(x_train)

        '''generator = datagen.flow(  
            x=x_test,
            y=y_test,
            batch_size=1,
            shuffle=False)  

        predict_size_test = int(math.ceil(x_test.shape[0] / 1))

        if os.path.isfile("Resnet50_224_features_test_augmented.npz"):
            test_features = np.load('Resnet50_224_features_test_augmented.npz')
            print("loaded train features")
            #print(train_features['features'])
            #print(train_features.shape)

        else:
            test_features = base_model.predict_generator(generator, x_test.shape[0])
            #Saving the bottleneck features
            np.savez('Resnet50_224_features_test_augmented', features=test_features)
            test_features = np.load('Resnet50_224_features_test_augmented.npz')
            # (std, mean, and principal components if ZCA whitening is applied).
            #datagen.fit(x_train)'''

    else:
        #x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
        print ("creating bottleneck features for training data")
        #Creating bottleneck features for the testing data

        if os.path.isfile("Resnet50_224_features_train.npz"):
            train_features = np.load('Resnet50_224_features_train.npz')
            print("loaded train features")
            #print(train_features['features'])
            #print(train_features.shape)

        else:
            train_features = base_model.predict(x_train, verbose=1)
            #Saving the bottleneck features
            #np.savez('Resnet50_224_features_train', features=train_features)
            #train_features = np.load('Resnet50_224_features_train.npz')
        
    if os.path.isfile("Resnet50_224_features_test.npz"):
        test_features = np.load('Resnet50_224_features_test.npz')
        print("loaded test features")
        #print(test_features.shape)
    else:
        test_features = base_model.predict(x_test, verbose=1)
        #Saving the bottleneck features
        #np.savez('Resnet50_224_features_test', features=test_features)
        #test_features = np.load('Resnet50_224_features_test.npz')

    #try for val_loss
    checkpointer = ModelCheckpoint(filepath='Resnet50_224_keras_fast_checkpoint_acc_final.h5', 
        monitor='val_acc', verbose=1, save_best_only=True)
    '''if os.path.isfile("vgg16_features_validation_vals.npz"):
        validation_features = data = np.load('vgg16_features_validation_vals.npz')
        print("loaded validation features")
        #print(validation_features['features'])
    else:
        validation_features = base_model.predict(x_validation, verbose=1)
        #Saving the bottleneck features
        np.savez('vgg16_features_validation_vals', features=validation_features)'''

    # maybe move optimizer and loss part to the models section?
    # optimization details
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training process in a for loop with learning rate drop every 25 epoches.

    #print(train_features['features'])
    #put validation features too
    '''historytemp = model.fit(train_features['features'], y_train, 
        validation_data=(validation_features['features'], y_validation), 
        batch_size=batch_size, epochs=maxepoches, verbose=1, callbacks=[reduce_lr])'''

    #  when loading features from .npz files this works fine else not, don't know the reason for this
    # happenning on macbook not on desktop, most probably some version issue
    '''historytemp = model.fit(train_features['features'], y_train, 
        validation_split=0.1, shuffle=True, 
        batch_size=batch_size, epochs=maxepoches, verbose=1, callbacks=[reduce_lr])'''
    y_test = keras.utils.to_categorical(y_test, 100)
    historytemp = model.fit(train_features['features'], y_train, validation_data=(test_features['features'], y_test),
        batch_size=batch_size, epochs=maxepoches, verbose=1, callbacks=[reduce_lr, checkpointer])
    '''historytemp = model.fit_generator(datagen.flow(x_train, y_train, 
        batch_size=batch_size), steps_per_epoch = x_train.shape[0], 
        validation_data=(x_validation, y_validation), callbacks=[reduce_lr], epochs=maxepoches)'''
    #model.save_weights('resnet50keras_fast.h5')
    model.load_weights("Resnet50_224_keras_fast_checkpoint_acc_final.h5")
    predict_fast(model, x_test, test_features['features'])
    print('prediction done')
    #Creating bottleneck features for the testing data
    #test_features = base_model.predict(x_test)

    #Saving the bottleneck features
    #np.savez('vgg16_features_test', features=test_features)
    # optimization details
    return model



def different_train_unfrozen(model, x_train, y_train, x_test, learning_rate, maxepoches):

    # training parameters
    batch_size = 128
    #maxepoches = 30

    #learning_rate = 0.0001
    lr_decay = 1e-6
    lr_drop = 25

    (x_train1, y_train1), (x_test1, y_test) = cifar100.load_data(label_mode='fine')
    y_test = keras.utils.to_categorical(y_test, 100)
    #x_train, x_validation, y_train, y_validation = get_validation_data(x_train, y_train)
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    checkpointer = ModelCheckpoint(filepath='Resnet50_224_keras_fast_checkpoint_acc_unfrozen.h5', 
        monitor='val_acc', verbose=1, save_best_only=True)
    # maybe move optimizer and loss part to the models section?
    # optimization details
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training process in a for loop with learning rate drop every 25 epoches.


    #put validation features too
    '''historytemp = model.fit(x_train, y_train, validation_split=0.1, shuffle=True, batch_size=batch_size, 
        epochs=maxepoches, verbose=1, callbacks=[reduce_lr, checkpointer])'''
    
    historytemp = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=maxepoches, verbose=1, callbacks=[reduce_lr, checkpointer])
    '''historytemp = model.fit_generator(datagen.flow(x_train, y_train, 
        batch_size=batch_size), steps_per_epoch = x_train.shape[0], 
        validation_data=(x_validation, y_validation), callbacks=[reduce_lr], epochs=maxepoches)'''
    model.load_weights("Resnet50_224_keras_fast_checkpoint_acc_unfrozen.h5")
    #predict_fast_unfrozen(model, x_test)
    #print('prediction done')
    #Creating bottleneck features for the testing data
    #test_features = base_model.predict(x_test)

    #Saving the bottleneck features
    #np.savez('vgg16_features_test', features=test_features)
    # optimization details
    return model


def get_data():
    return get_processed_data()

if __name__ == '__main__':

    size = 224
    train_data, train_label, test_data = get_data()
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data, test_data = upscale_images(train_data, test_data, size)


    '''obj = models.InceptionV3Keras_fast()
    model = obj.model
    base_model = obj.base_model
    different_train(model, base_model, train_data, train_label, test_data)'''
    #model = models.cifar100vgg().model
    #model = models.InceptionV3Keras().model
    

    '''model = models.VGG16Keras().model
    train(model, train_data, train_label, test_data)'''


    obj = models.ResNet50Keras_fast_unfrozen()
    model = obj.model
    predict(model, test_data)
    base_model = obj.base_model
    top_model = obj.top_model
    model = different_train_unfrozen(model, train_data, train_label, test_data, 0.1, 150)

    '''for layer in model.layers:
        layer.trainable = True
    mid_start = model.get_layer('activation_40')
    all_layers = model.layers
    for i in range(model.layers.index(mid_start)):
        all_layers[i].trainable = False

    model.summary()

    different_train_unfrozen(model, train_data, train_label, test_data, 0.001, 5)'''





    #different_train(top_model, base_model, train_data, train_label, test_data, False)
    '''obj = models.ResNet50Keras_fast()
    model = obj.model
    base_model = obj.base_model
    different_train(model, base_model, train_data, train_label, test_data, False, 0.1, 200)'''
 
    '''obj = models.DenseNet121Keras_fast()
    model = obj.model
    base_model = obj.base_model
    different_train(model, base_model, train_data, train_label, test_data, False)'''

    #different_train_unfrozen(model, train_data, train_label, test_data)
    #train(model, train_data, train_label, test_data)

    #predict(model, test_data)
    '''
    integrated_model = obj.integrated_model
    test_features = data = np.load('ResNet50_features_test.npz')
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    integrated_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # at compilation the weights get reset, not sure about this
    # https://github.com/keras-team/keras/issues/2379
    #integrated_model.load_weights("resnet50keras_fast_checkpoint_acc.h5")
    predict_fast_for_Model(integrated_model, test_data, test_features['features'])'''
