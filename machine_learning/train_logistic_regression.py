import generic_tools
import numpy as np
import pandas as dp
import random
from scipy import optimize

def create_X_y_arrays(data):
    # split the data into the training data and their labels
    i=data.shape[1]-1
    X=np.matrix(data[:,:i])
    X = np.c_[np.ones(len(X)), X]
    y=np.matrix(data[:,i])
    # return the training data and the classification labels
    return X, y

def sigmoid(z):
    # the sigmoid function used to predict the classification of a source
    g = 1/(1+np.exp(-z))
    # return the predicted classification
    return g

def reg_cost_func(theta, X, y, lda):
    # the regularised cost function, J. This is minimalised to find the best classification solution.
    m=y.shape[1]
    J=0
    sig=sigmoid(X * np.c_[theta])
    J=(1/float(m))*(-y.dot(np.log(sig)) - (1-y).dot(np.log(1-sig)))
    temp=np.copy(theta)
    temp[0]=0.0
    J=J+(lda/(2*m))*np.sum(np.multiply(temp,temp))
    Jnum=np.array(J)[0][0]
    # return the value of the regularised cost function
    return Jnum

def quadratic_features(X):
    # the option to use a quadratic model instead of a simple linear model
    X_quad=np.matrix(np.zeros((X.shape[0],X.shape[1]*2)))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_quad[i,j]=X[i,j]
            X_quad[i,j+X.shape[1]]=X[i,j]**2
    # return the model parameters with additional quadratic parameters
    return X_quad

def learning_curve(X, y, Xval, yval, lda, options):
    # Obtain the learning curve data
    # train using 1 datapoint, 2 datapoints, 3 datapoints... n datapoints and monitor the classification error
    
    m=X.shape[0]
    n=Xval.shape[0]
    error_train=np.zeros((m))
    error_val=np.zeros((m))
    for i in range(1,m):
        theta=np.zeros((X.shape[1]))
        initial_theta=np.zeros((X.shape[1]))
        theta, cost, _, _, _ = optimize.fmin(lambda t: reg_cost_func(t,X[:i,:],y[:,:i],lda), initial_theta, **options)
        error_train[i]=check_error(X,y,theta)
        error_val[i]=check_error(Xval,yval,theta)
    # return the training and validation errors for a given model
    return error_train, error_val, theta

def check_error(X,y,theta):
    # Calculate the classification error between the labelled data and the predictions
    error = (1./(2.*float(X.shape[0])))*np.sum(np.power((sigmoid(X * np.c_[theta])-y.T),2))
    return error

def validation_curve(X, y, Xval, yval,options):
    # Obtain the validation curve data
    # train using different values of lamdba and monitor the classification error
    lambda_vec = np.array([10.**a for a in np.arange(-5,5,0.1)])
    error_train=np.zeros((lambda_vec.shape[0]))
    error_val=np.zeros((lambda_vec.shape[0]))
    theta_rec=np.zeros((lambda_vec.shape[0],X.shape[1]))
    m=X.shape[0]
    n=Xval.shape[0]
    for i in range(0,lambda_vec.shape[0]):
        theta=np.zeros((X.shape[1]))
        initial_theta=np.zeros((X.shape[1]))
        lda=lambda_vec[i]
        theta, cost, _, _, _ = optimize.fmin(lambda t: reg_cost_func(t,X,y,lda), initial_theta, **options)
        theta_rec[i]=theta
        error_train[i]=check_error(X,y,theta)
        error_val[i]=check_error(Xval,yval,theta)
        print theta
    min_err_val = min(error_val)
    for i in range(0,lambda_vec.shape[0]):
        if error_val[i] == min_err_val:
            lda=lambda_vec[i]
    #return the training and validation errors for given lambda values and the optimal lambda.
    return error_train, error_val, lambda_vec, lda

def classify_data(X,y,theta):
    # classify a given dataset X, using the input model theta and compare to the input labels y
    tp=0
    fp=0
    fn=0
    tn=0
    classified_data=[]
    predictions=predict(X,theta)
    y=y.T
    for i in range(predictions.shape[0]):
        if predictions[i] > 0.5 and y[i] == 1:
            tp=tp+1
            classified_data.append([X[i,1],X[i,2],X[i,3],X[i,4],'TP'])
        elif predictions[i] > 0.5 and y[i] == 0:
            fp=fp+1
            classified_data.append([X[i,1],X[i,2],X[i,3],X[i,4],'FP'])
        elif predictions[i] < 0.5 and y[i] == 1:
            fn=fn+1
            classified_data.append([X[i,1],X[i,2],X[i,3],X[i,4],'FN'])
        elif predictions[i] < 0.5 and y[i] == 0:
            tn=tn+1
            classified_data.append([X[i,1],X[i,2],X[i,3],X[i,4],'TN'])
    # return the true positives, false positives, false negatives, true negatives and the classified dataset
    return tp, fp, fn, tn, classified_data

def predict(X,theta):
    # predict the classification of a given dataset X using a given model theta
    predictions=sigmoid(X * np.c_[theta])
    return predictions
