import pickle
import keras
from sklearn.cross_validation import train_test_split
import scipy
from scipy import misc
import numpy as np

def get_inputs():
	with open('./data/train_data', 'rb') as f:
	    train_data = pickle.load(f)
	    train_label= pickle.load(f)
	with open('./data/test_data', 'rb') as f:
	    test_data = pickle.load(f)

	return train_data, train_label, test_data

def get_processed_data():
	num_classes = 100
	train_data, train_label, test_data = get_inputs()
	train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)
	test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose(0, 2, 3, 1)
	train_label = keras.utils.to_categorical(train_label, num_classes)
	return train_data, train_label, test_data

def upscale_images(train_data, test_data, size):
	train_data = np.array([scipy.misc.imresize(train_data[i], (size, size, 3)) for i in range(0, len(train_data))]).astype('float32')
	test_data = np.array([scipy.misc.imresize(test_data[i], (size, size, 3)) for i in range(0, len(test_data))]).astype('float32')
	print(test_data.shape)
	return train_data, test_data

def get_validation_data(train_data, train_label, split_ratio=0.1):
	return train_test_split(train_data, train_label, test_size = split_ratio, random_state = 0) 
