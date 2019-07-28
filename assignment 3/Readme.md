http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

Consider the Indiana Pines data. The data is in .mat format (indian_pines_corrected.mat)...This contain the image: no of rows x no of columns x no of bands

In the ground truth file, the label per pixel is given: there are 16 labels in total.

In order to work with the softmax classifier, you have to convert the labels in the one hot format.

Consider 50% pixels per class as training and remaining 50% pixels as testing.

Task will be to design a softmax classifier. Training is to be done using the maximum likelihood criteria (in this case, this is also known as cross-entropy loss).

First try to work out the derivation on your own (the qs was given in one of the home works) and then it will be easy to implement.