from keras.datasets import cifar100


def predict(model,test_data,normalize=True):
    '''if normalize:
        x = self.normalize_production(x)'''
    predictions = model.predict_classes(test_data, verbose=1)

    f = open('Submission.csv', 'w')
    f.write('ids,labels\n')

    for i in range(0, test_data.shape[0]):
    	f.write(str(i)+","+str(predictions[i])+'\n')

    f.close()


def predict_fast(model,test_data, test_data_features,normalize=True):
    '''if normalize:
        x = self.normalize_production(x)'''
    predictions = model.predict_classes(test_data_features, verbose=1)
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    correct = 0
    for i in range(0, test_data.shape[0]):
        if predictions[i] == y_test[i]:
            correct += 1

    print(correct/test_data.shape[0])
    '''
    f = open('Submission_aisehi.csv', 'w')
    f.write('ids,labels\n')

    for i in range(0, test_data.shape[0]):
    	f.write(str(i)+","+str(predictions[i])+'\n')

    f.close()'''
    

def predict_fast_unfrozen(model,test_data,normalize=True):
    '''if normalize:
        x = self.normalize_production(x)'''
    predictions = model.predict_classes(test_data, verbose=1)
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    correct = 0
    for i in range(0, test_data.shape[0]):
        if predictions[i] == y_test[i]:
            correct += 1

    print(correct/test_data.shape[0])
    '''
    f = open('Submission_aisehi.csv', 'w')
    f.write('ids,labels\n')

    for i in range(0, test_data.shape[0]):
        f.write(str(i)+","+str(predictions[i])+'\n')

    f.close()'''
    