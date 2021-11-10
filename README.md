# 1HiddenLayerNN


# Make Moons (a toy dataset) and 1 hidden layer Neural Net. 
Make Moons is a data set containing non-linearly separable data; using the NN with the Logistic Regression Function does not work here since it is intended for linearly separable data. 
The figure illustrates this pictorially showing the decision plane is straight, and it fails to correctly classify edge points in the vicinity of the opposite classes. 

When one runs a ML algorithm, it is tempting to simply look at accuracies and conclude progress is made when observing increasing accuracies as the learning progresses (like in this case) and to conclude the algorithm works. 
However, simply looking at accuracies can be misleading. Accuracies here improve to 86% from 69%; however, this problem cannot be solved with this NN effectively.

Other details:
* Loss = - (y * ln(yhat) + (1-y)*ln(1-yhat) )
* Activation Function for the hidden layer is sigmoid
* Yhat is coverted to the prediction class based on if Yhat is greater or equal to, or less than 0.5.

````
Cost after iteration 0: 0.693147
Train Accuracy (fraction out of 1):  0.7925
Cost after iteration 10000: 0.308631
Train Accuracy (fraction out of 1):  0.8555
Cost after iteration 20000: 0.294575
Train Accuracy (fraction out of 1):  0.8654999999999999
Cost after iteration 30000: 0.291133
Train Accuracy (fraction out of 1):  0.8685
Cost after iteration 40000: 0.290060
Train Accuracy (fraction out of 1):  0.8685
Cost after iteration 50000: 0.289685
Train Accuracy (fraction out of 1):  0.8685
              precision    recall  f1-score   support

       moon1       0.87      0.87      0.87      1000
       moon2       0.87      0.87      0.87      1000

    accuracy                           0.87      2000
   macro avg       0.87      0.87      0.87      2000
weighted avg       0.87      0.87      0.87      2000


````


![Figure_1](https://user-images.githubusercontent.com/20401990/141185914-ffdab0f0-4568-42c8-8a37-d66425cb59c8.png)



