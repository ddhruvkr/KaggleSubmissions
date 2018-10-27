import models
from inputs import get_processed_data
from test import predict

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

def train(model, x_train, y_train, x_test):

    # training parameters
    batch_size = 128
    maxepoches = 2

    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 20

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
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)

    # optimization details

    # maybe move optimizer and loss part to the models section?
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training process in a for loop with learning rate drop every 25 epoches.

    '''historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=maxepoches,
                        validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)'''
    historytemp = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

    # model.save_weights('cifar100vgg.h5')
    return model

def get_data():
    return get_processed_data()

if __name__ == '__main__':

    train_data, train_label, test_data = get_data()
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    model = models.cifar100vgg().model
    train(model, train_data, train_label, test_data)

    predict(model, test_data)
