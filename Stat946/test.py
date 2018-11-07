from keras.datasets import cifar100


def predict(model,test_data,normalize=True):
    '''if normalize:
        x = self.normalize_production(x)'''
    get_accuracy(model, test_data)
    predictions = model.predict_classes(test_data, verbose=1)

    f = open('Submission.csv', 'w')
    f.write('ids,labels\n')

    for i in range(0, test_data.shape[0]):
    	f.write(str(i)+","+str(predictions[i])+'\n')

    f.close()

def get_accuracy(model, test_data):
	(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
	predictions = model.predict_classes(test_data, verbose=1)
	correct = 0
	for i in range(0, x_test.shape[0]):
		if predictions[i] == y_test[i]:
			correct += 1
	print (correct/x_test.shape[0])

def predict_fast(model,test_data, test_data_features,normalize=True):
    '''if normalize:
        x = self.normalize_production(x)'''
    predictions = model.predict_classes(test_data_features, verbose=1)
    get_accuracy(model, test_data_features)
    f = open('Submission_Resnet50_100_acc.csv', 'w')
    f.write('ids,labels\n')

    for i in range(0, test_data.shape[0]):
    	f.write(str(i)+","+str(predictions[i])+'\n')

    f.close()

def predict_fast_for_Model(model,test_data, test_data_features,normalize=True):
    '''if normalize:
        x = self.normalize_production(x)'''
    predictions = model.predict(test_data_features, verbose=1)
    print(predictions)
    predictions = predictions.argmax(axis=-1)
    print(predictions)
    f = open('Submission_resnet50_acc_nodropout.csv', 'w')
    f.write('ids,labels\n')

    for i in range(0, test_data.shape[0]):
    	f.write(str(i)+","+str(predictions[i])+'\n')

    f.close()
    
def save_results(model, test_data_features, test_data, iteration_number):
	predictions = model.predict_classes(test_data_features, verbose=1)
	f = open('Submission_resnet50_' + iteration_number + '.csv', 'w')
	f.write('ids,labels\n')
	for i in range(0, test_data.shape[0]):
		f.write(str(i)+","+str(predictions[i])+'\n')
		f.close()

def load_predict_save(model, test_data, test_data_features):
	model.load_weights('resnet50keras_fast_checkpoint_acc.hf')
	predict_fast(model, test_data, test_data_features)

def predict_fast_unfrozen(model,test_data,normalize=True):
    '''if normalize:
        x = self.normalize_production(x)'''
    predictions = model.predict_classes(test_data, verbose=1)
    get_accuracy(model)
    '''
    f = open('Submission_aisehi.csv', 'w')
    f.write('ids,labels\n')

    for i in range(0, test_data.shape[0]):
        f.write(str(i)+","+str(predictions[i])+'\n')

    f.close()'''
    