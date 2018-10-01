import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical



from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential,Model,load_model
from keras.layers import Embedding,Conv1D,MaxPooling1D,TimeDistributed
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.layers.recurrent import LSTM
from keras.constraints import maxnorm
import keras
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau,EarlyStopping

print ("correct version")

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



# next step is to convert this to word embeddings
# but first we need to convert the whole array of text to array of numbers as we indexed them previously
# currently i am not putting validation data in this conversion
# therefore just making two sets of conversion text to number encodings(training and test data, no validation data)
# 
encoded_words_train = tokenizer.texts_to_sequences(trainX)
encoded_words_test = tokenizer.texts_to_sequences(testX)
encoded_words_validation = tokenizer.texts_to_sequences(validationX)

#print(encoded_words_test)



maxWordCount=64

#padding all text to same size
trainX_encodedPadded_words = sequence.pad_sequences(encoded_words_train, maxlen=maxWordCount)
testX_encodedPadded_words = sequence.pad_sequences(encoded_words_test, maxlen=maxWordCount)
validationX_encodedPadded_words = sequence.pad_sequences(encoded_words_validation, maxlen=maxWordCount)

# One Hot Encoding
trainY = to_categorical(trainY, 5)
validationY = to_categorical(validationY, 5)

earlyStopping = EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=1,
                              verbose=0, mode='auto')


model = Sequential()
output_dim = 32
model.add(Embedding(tokenizer_vocab_size, output_dim, input_length = maxWordCount))
# embedding paramenter (input_size: vocab size, output_dim: size of vector space in which word is embedded, input_length: sentence size)
model.add(LSTM(32, return_sequences=True))
# lstm output is 32
model.add(Dropout(0.6))

model.add(LSTM(32, return_sequences=True))
# lstm output is 32
model.add(Dropout(0.6))
#model.add(Dense(1200, activation='relu',W_constraint=maxnorm(1)))
#model.add(Dropout(0.6))
#model.add(Dense(32, activation='relu',W_constraint=maxnorm(1)))

#model.add(Dropout(0.5))
 #output layer

model.add(Flatten())
model.add(Dense(5, activation='softmax'))

Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])


model.summary()


epochs=20
batch_size=32
model.fit(trainX_encodedPadded_words, trainY, epochs = epochs, batch_size=batch_size, verbose=2, 
shuffle = True, validation_data=(validationX_encodedPadded_words, validationY), callbacks=[earlyStopping])


f = open('Submission_LSTM_Validation_1.csv', 'w')
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





# Things to be noted/learned


# RETURN SEQUENCES
''' return sequences give the unwrapped version of the rnn
i.e. the output at each timestep, so essentially another dimenstion is added to the output
'''
# to visually understand return sequences
# https://stackoverflow.com/questions/42755820/how-to-use-return-sequences-option-and-timedistributed-layer-in-keras


# DIFFERENCE BETWEEN DENSE AND TIMEDISTRIBUTEDDENSE
'''Dense only receives 2D tensor, which means that there is NO time dimension. i.e. an 2D -> 2D conversion.
TimeDistributedDense only receives 3D tensor, which includes time dimension. i.e. an 3D -> 3D conversion.

Q: So Dense actually only apply activation function to the last time step?
A: No, there is no time dimension in Dense layer
Q: for Dense , it is used in Many-to-One or One-to-One cases
A: it is one-to-one
Q: And TimeDistributedDense is used in Many-to-Many and One-to-Many cases?
A: it is many-to-many'''


'''One thing in my experiments I could not explain is when I encode the words to integers 
if I randomly assign unique integers to words the best accuracy I get is 50–55% 
(basically the model is not doing much better than random guessing). However if the words 
are encoded such that highest frequency words get the lowest number then the model accuracy 
is 80% in 3–5 epochs. My guess is this is necessary to train the embedding layer but cannot 
find an explanation on why anywhere.'''


''' THE PROBLEM I FACED WAS THAT I HAD TO USE RETURN SEQUENCE IN LSTM FOR THE CNN PART
THIS CAUSED DIMENSION MISMATCH IN THE LAST DENSE LAYER AS IT WAS EXPECTING A 2 DIM INPUT 
WHEREAS IT WAS GETTING A 3 DIM FROM THE PREVIOUS DENSE LAYER OR THE CNN OR LSTM
I TRIED USING TIMEDISTRIBUTEDDENSE BUT THAT DID NOT WORK (NEED TO INVESTIGATE WHY)
SO FINALLY I USED THE FLATTENED LAYER TO TURN IT TO 2 DIM

https://github.com/keras-team/keras/issues/6351
'''

'''The problem is that you start with a three dimensional layer but never reduce the dimensionality in any of the following layers.
Try adding mode.add(Flatten()) before the last Dense layer'''


'''Also one thing to try is the LSTM-CNN model, that seems to work better than the CNN-LSTM
and just the LSTM model'''



'''
http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/
'''