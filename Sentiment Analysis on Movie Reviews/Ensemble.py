import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical



from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential,Model,load_model
from keras.layers import Embedding,Conv1D,MaxPooling1D
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.layers.recurrent import LSTM
from keras.constraints import maxnorm
import keras
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau,EarlyStopping


data = pd.read_csv('./train.tsv', sep='\t', header=0)
trainX = np.array(list(data['Phrase']))
trainY = np.array(list(data['Sentiment']));

length=-1
for x in trainX:
    if length < len(x):
        length=len(x)
#print(length)

validation_count=30000

validationY = trainY[:validation_count]
validationX = trainX[:validation_count]

trainX = trainX[validation_count:]
trainY = trainY[validation_count:]

data = pd.read_csv('./test.tsv', sep='\t', header=0)
testX=np.array(list(data['Phrase']))
X_test_PhraseID=np.array(list(data['PhraseId']))

# once we have the data, we would need to convert it into tokens i.e. represent each unique work with a unique integer.
# this could be done by keras tokenizer
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(np.concatenate((trainX, testX), axis=0))
# Tokenizer.fit_on_texts(X_train)
tokenizer_vocab_size = len(tokenizer.word_index) + 1
print ("Vocab size",tokenizer_vocab_size)

'''One thing in my experiments I could not explain is when I encode the words to integers 
if I randomly assign unique integers to words the best accuracy I get is 50–55% 
(basically the model is not doing much better than random guessing). However if the words 
are encoded such that highest frequency words get the lowest number then the model accuracy 
is 80% in 3–5 epochs. My guess is this is necessary to train the embedding layer but cannot 
find an explanation on why anywhere.'''


'''Also one thing to try is the LSTM-CNN model, that seems to work better than the CNN-LSTM
and just the LSTM model'''

# next step is to convert this to word embeddings
# but first we need to convert the whole array of text to array of numbers as we indexed them previously
# currently i am not putting validation data in this conversion
# therefore just making two sets of conversion text to number encodings(training and test data, no validation data)
# 
encoded_words_train = tokenizer.texts_to_sequences(trainX)
encoded_words_test = tokenizer.texts_to_sequences(testX)
encoded_words_validation = tokenizer.texts_to_sequences(validationX)
#print(encoded_words_test)



maxWordCount=60

#padding all text to same size
trainX_encodedPadded_words = sequence.pad_sequences(encoded_words_train, maxlen=maxWordCount)
testX_encodedPadded_words = sequence.pad_sequences(encoded_words_test, maxlen=maxWordCount)
validationX_encodedPadded_words = sequence.pad_sequences(encoded_words_validation, maxlen=maxWordCount)

# One Hot Encoding
trainY = to_categorical(trainY, 5)
validationY = to_categorical(validationY, 5)




model = Sequential()
output_dim = 32
model.add(Embedding(tokenizer_vocab_size, output_dim, input_length = maxWordCount))
# embedding paramenter (input_size: vocab size, output_dim: size of vector space in which word is embedded, input_length: sentence size)
model.add(LSTM(64,return_sequences=True))
# lstm output is 32
model.add(Dropout(0.6))
#model.add(LSTM(32,return_sequences=True))
# lstm output is 32
#model.add(Dropout(0.5))
#model.add(LSTM(64,return_sequences=True))
# lstm output is 32
#model.add(Dropout(0.5))
model.add(Conv1D(32, 6,activation='relu'))
model.add(MaxPooling1D(pool_size=5))
#model.add(Dense(1200, activation='relu',W_constraint=maxnorm(1)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32, activation='relu',W_constraint=maxnorm(1)))


# model.add(Dropout(0.5))
 #output layer
model.add(Dense(5, activation='softmax'))


earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])


model.summary()


epochs=20
batch_size=64
#model.fit(trainX_encodedPadded_words, trainY, epochs = epochs, batch_size=batch_size, verbose=2)
model.fit(trainX_encodedPadded_words, trainY, epochs = epochs, batch_size=batch_size, verbose=2, 
shuffle = True, validation_data=(validationX_encodedPadded_words, validationY), callbacks=[earlyStopping])



f = open('Submission_LSTM_CNN_Validation_GPU.csv', 'w')
f.write('PhraseId,Sentiment\n')


# predictions = model.predict(X_test_encodedPadded_words)
predicted_classes = model.predict_classes(testX_encodedPadded_words, batch_size=batch_size, verbose=1)
# print np.sum(predicted_classes==Y_Val2)/(1.0*Y_Val2.shape[0])
# print predicted_classes
# preds = new_model.predict(x)
# print predicted_classes
for i in range(0,X_test_PhraseID.shape[0]):
    # pred =np.argmax(predictions[i])
    f.write(str(X_test_PhraseID[i])+","+str(predicted_classes[i])+'\n')
    # print predictions[i],"=>",pred

f.close()