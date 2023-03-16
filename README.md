# 2nd_Iteration_YOLO v8
This repo includes the code for training and testing YOLO v8 on the dataset we have (Deer, Badger, Hare, Fox)

train.py --> The code here splits the labelled dataset randomly into training and validation datasets, then it trains the model using YOLO v8n with parameters (epochs=50, batch=9)

test_random30.py --> It test the model that we just trained to 30 random images from the test dataset we have and predicts its classifications. Test set is unlabelled images that the model never saw before, and it has full sequence of animal images that haven't been trained on before.
