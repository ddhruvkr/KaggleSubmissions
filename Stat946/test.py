def predict(model,test_data,normalize=True):
    '''if normalize:
        x = self.normalize_production(x)'''
    predictions = model.predict_classes(test_data, verbose=1)

    f = open('Submission.csv', 'w')
    f.write('ids,labels\n')

    for i in range(0, test_data.shape[0]):
    	f.write(str(i)+","+str(predictions[i])+'\n')

    f.close()