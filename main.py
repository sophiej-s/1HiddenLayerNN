
#------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import sklearn

def SIGMOID(input):
    #input is an np array
    out=1/(1+ np.exp(-1*input))
    return out



from sklearn import * #use to make the moons data set


np.random.seed(0)
X, Y = sklearn.datasets.make_moons(2000, noise=0.20)  # X shape is (2000, 2)



N_features=X.shape[1] #two features x1 and x2
N_tuples=X.shape[0]
W=np.zeros((N_features, 1))#initialize the W coeff to zero. Size is 2x1
B=0.0
Y=Y.reshape(Y.shape[0],1) #Y shape is (2000, 1)
Yhat=np.zeros((N_tuples,1)) #initialize the Yhat. Size is 2000 x 1

num_iterationsGradient=60000
learning_rateGradient=0.005

Cost=0
dJdW=[0.0, 0.0]
dCost_dB=0.0



Y_predict = np.zeros(( N_tuples, 1))


for i in range(0, num_iterationsGradient):

    
    Yhat=SIGMOID(np.dot(np.transpose(W),np.transpose(X))+B) 
    Yhat=np.transpose(Yhat) #make it  the size of (num of tuples x 1)
    Loss=-1.0*(Y*np.log(Yhat)  + (1-Y)* np.log(1-Yhat)) #1-Y will compute using the  broadcasing in python
        
    Cost=1/N_tuples*np.sum(Loss, axis=0)
    dCost_dW=1/N_tuples * np.dot(np.transpose(X), (Yhat-Y))  #dot of (features x #tuples) and (1, #tuples). Size of DJ/DW = 2x1
    dCost_dB=float(1/N_tuples * np.sum(Yhat-Y, axis=0) )

    W=W-learning_rateGradient*dCost_dW
    B=B-learning_rateGradient*dCost_dB

    #predict using the learned coefficients in W and B and update the Y-hat:
    Yhat_pred=SIGMOID(np.dot(np.transpose(W),np.transpose(X))+B) 
    Yhat_pred=np.transpose(Yhat_pred) #make it  the size of (num of tuples x 1)

    # create predictions: Yhat>0.5-> class 1 otherwise class 0
    
    for j in range(Yhat_pred.shape[0]):        
        if Yhat_pred[j,0] >= 0.5 :
            Y_predict[j,0] = 1
        else:
            Y_predict[j,0] = 0
    
    if i % 10000 == 0:       
        print ("Cost after iteration %i: %f" %(i, Cost))
        fraction_wrong_predictions=float(np.count_nonzero(Y_predict - Y, axis=0))/N_tuples
        fraction_accurate_predictions=1-fraction_wrong_predictions
        print("Train Accuracy (fraction out of 1): ", fraction_accurate_predictions)

    
    # print(i)

    
    
#  perform a check on the test set 

from sklearn.metrics import classification_report
target_names = ['moon1', 'moon2']
print(classification_report(Y, Y_predict, target_names=target_names))





x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = 0.1
    # Generate a grid of points with distance h between them
x1, x2 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
xx1=x1.ravel()
xx2=x2.ravel()
RESULT=np.array([xx1, xx2])
RESULT=np.transpose(RESULT)
    
Y_predict2 = np.zeros(( xx2.shape[0], 1))

Yhat_pred2=SIGMOID(np.dot(np.transpose(W),np.transpose(RESULT))+B) 
Yhat_pred2=np.transpose(Yhat_pred2) #make it  the size of (num of tuples x 1)

for j in range(Yhat_pred2.shape[0]):        
        if Yhat_pred2[j,0] >= 0.5 :
            Y_predict2[j,0] = 1
        else:
            Y_predict2[j,0] = 0



Y_predict2= Y_predict2.reshape(x1.shape)
    
    
    
plt.close()
# Plot the contour and training examples
plt.subplot(1,3,1)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
plt.title("Moons (true  classes)")
plt.xlabel('Feature x1') 
plt.ylabel('Feature x2') 


plt.subplot(1,3,2)
plt.scatter(X[:, 0], X[:, 1], c=Y_predict, cmap=plt.cm.Spectral)
plt.title("Moons (predicted classes)")
plt.xlabel('Feature x1') 
plt.ylabel('Feature x2') 


plt.subplot(1,3,3)
plt.contourf(x1, x2, Y_predict2, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=Y_predict, cmap=plt.cm.Spectral)
plt.title("Moons (predicted classes)and the decision plane")
plt.xlabel('Feature x1') 
plt.ylabel('Feature x2') 
