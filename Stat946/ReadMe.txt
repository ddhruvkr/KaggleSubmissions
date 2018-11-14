I have used Transfer Learning for this data challenge.

Although I tried various models, my final submission is an ensemble model(voting between different submission files).

I trained a Resnet with first training the randomnly initialized Fully connected layers and then subsequently
training the rest of the layers in small batches. The submission files generated after training of various layers were used in the emsemble method.
The best individual model (M1) gave me an accuracy of 0.7633. After ensembling the rest of the Resnet outputs I got an accuracy of 0.7803

I also used the final output of a VGG16 model which I trained at the start of the data challenge(Though, I couldn't optimize it fully, it gave me an accuracy of 0.63) in the ensemble which made the final accuracy to go to 0.7853.

You can run the code by running the train.py file. By default it would load the weights of the trained Resnet model M1(currently placed in the h5 folder) and create a submission file after running the forward pass. As mentioned earlier this file would give the accuracy of 0.7633.

The code to run and train the VGG16 and Resnet50 is commented for now. Once uncommented both models could be trained.

The npz folder is currently empty but at the time of training the code would save the bottleneck features(output features) vectors in it. I created them for both Resnet and VGG to save time(as I was working with a CPU). With these features, forward pass is only required once when training the Fully-Connected layers.

Finally the csv folder contains all the csv files that I used in the ensemble method and the submitted csv files too.