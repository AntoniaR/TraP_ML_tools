import pandas as pd
import plotting_tools
import generic_tools
import numpy as np
import train_anomaly_detect
import train_logistic_regression
import train_sigma_margin
import os
import random
from scipy import optimize


def learning_curve(anomaly,logistic,transSrc,all_data,lda,options,precis_thresh,recall_thresh,path):
    # Conduct tests to ensure that the machine learning algorithm is working effectively
    print "Creating learning curves"

    data=all_data.loc[(all_data['ttype'] == 2) & (all_data['V']>0.) & (all_data['eta']>0.)]
    datatmp=data.apply(lambda row:[row['#Runcat'],row['eta'],row['V'],row['flux'],row['fluxrat'],row['variable']],axis=1)
    data = datatmp.as_matrix()
    stable=all_data.loc[(all_data['ttype'] == 2) & (all_data['V']>0.) & (all_data['eta']>0.) & (all_data['variable'] == 0)]
    stable=stable.apply(lambda row:[row['#Runcat'],row['eta'],row['V'],row['flux'],row['fluxrat'],row['variable']],axis=1)
    stable=stable.as_matrix()
    variable=all_data.loc[(all_data['ttype'] == 2) & (all_data['V']>0.) & (all_data['eta']>0.) & (all_data['variable'] == 1)]
    variable=variable.apply(lambda row:[row['#Runcat'],row['eta'],row['V'],row['flux'],row['fluxrat'],row['variable']],axis=1)
    variable=variable.as_matrix()
    
    # shuffle up the transient and stable data
    shuffled = np.matrix(generic_tools.shuffle_datasets(data))

    # sort the data into a training, validation and testing dataset. This is hardcoded to be 60%, 30% and 10% (respectively) of the total dataset
    train, valid, test = generic_tools.create_datasets(shuffled, int(len(shuffled)*0.6), int(len(shuffled)*0.9))
    
    if anomaly:
        if not os.path.exists(path+"AD_learn_data.csv"):
            rangeNums=np.unique([int(10.**a) for a in np.arange(0.08,4.01,4./50.)])
            train=np.array(train)
            valid=np.array(valid)
            error_train, error_valid = train_anomaly_detect.learning_curve(stable,variable,train, valid, precis_thresh, recall_thresh, rangeNums,path)
            tmp = [[rangeNums[x], error_train[x], error_valid[x]] for x in range(len(rangeNums))]
            generic_tools.write_test_data(path+"AD_learn_data.csv",tmp)
        else:
            tmp = generic_tools.extract_data(path+"AD_learn_data.csv")
            rangeNums = [int(x[0]) for x in tmp]
            error_train = [float(x[1]) for x in tmp]
            error_valid = [float(x[2]) for x in tmp]      
        plotting_tools.plotLC(rangeNums,error_train, error_valid, path+"AD_learning", True, True, "Number")        

    if logistic:
        if not os.path.exists(path+"LR_learn_data.csv"):
            Xtrain, ytrain = train_logistic_regression.create_X_y_arrays(np.matrix([[np.log10(a[1]), np.log10(a[2]), np.log10(a[3]), a[4], a[5]] for a in np.array(train)]))
            Xvalid, yvalid = train_logistic_regression.create_X_y_arrays(np.matrix([[np.log10(a[1]), np.log10(a[2]), np.log10(a[3]), a[4], a[5]] for a in np.array(valid)]))

            # plot the learning curve to check that it is converging to a solution as you increase the size of the training dataset (Optional but recommended). Basically, it repeatedly trains using 1 datapoint, 2, 3, 4, ... upto the full training dataset size. If the training and validation errors converge, then all is well.
            error_train, error_valid, theta = train_logistic_regression.learning_curve(Xtrain, ytrain.T, Xvalid, yvalid.T, lda, options)
            tmp = [[x, error_train[x], error_valid[x]] for x in range(len(error_train))]
            generic_tools.write_test_data(path+"LR_learn_data.csv",tmp)
        else:
            tmp = generic_tools.extract_data(path+"LR_learn_data.csv")
            rangeNums = [int(x[0]) for x in tmp]
            error_train = [float(x[1]) for x in tmp]
            error_valid = [float(x[2]) for x in tmp]      
        plotting_tools.plotLC(range(len(error_train)),error_train, error_valid, path+"LR_learning", True, True, "Number")

    if transSrc:
        if not os.path.exists(path+"TransBest_learn_data.csv"):
            # Extract out the possible and candidate transient sources
            possTransData = all_data.loc[((all_data['ttype'] == 0) | (all_data['ttype'] == 1)) & (all_data['variable'] == 0)]
            possTransData = possTransData.as_matrix()
            possTransSims = all_data.loc[((all_data['ttype'] == 0) | (all_data['ttype'] == 1)) & (all_data['variable'] == 1)]
            possTransSims = possTransSims.as_matrix()
    
            # Sort out the sigma data for plotting and training
            best_data=all_data[['minRmsSigma','variable']].as_matrix()
            worst_data=all_data[['maxRmsSigma','variable']].as_matrix()

            # shuffle up the transient and stable data
            best_shuffled = np.matrix(generic_tools.shuffle_datasets(best_data))
            worst_shuffled = np.matrix(generic_tools.shuffle_datasets(worst_data))
    
            # sort the data into a training, validation and testing dataset. This is hardcoded to be 60%, 30% and 10% (respectively) of the total dataset
            best_train, best_valid, best_test = generic_tools.create_datasets(best_shuffled, len(best_shuffled)*0.6, len(best_shuffled)*0.9)
            worst_train, worst_valid, worst_test = generic_tools.create_datasets(worst_shuffled, len(worst_shuffled)*0.6, len(worst_shuffled)*0.9)

            best_error_train, worst_error_train, best_error_valid, worst_error_valid = train_sigma_margin.learning_curve(best_train,worst_train,best_valid,worst_valid,detection_threshold)

            tmp = [[x, best_error_train[x], best_error_valid[x]] for x in range(len(best_error_train))]
            generic_tools.write_test_data(path+"TransBest_learn_data.csv",tmp)
            tmp = [[x, worst_error_train[x], worst_error_valid[x]] for x in range(len(worst_error_train))]
            generic_tools.write_test_data(path+"TransWorst_learn_data.csv",tmp)
        else:
            tmp = generic_tools.extract_data(path+"TransBest_learn_data.csv")
            rangeNums = [int(x[0]) for x in tmp]
            best_error_train = [float(x[1]) for x in tmp]
            best_error_valid = [float(x[2]) for x in tmp]      
            tmp = generic_tools.extract_data(path+"TransWorst_learn_data.csv")
            rangeNums = [int(x[0]) for x in tmp]
            worst_error_train = [float(x[1]) for x in tmp]
            worst_error_valid = [float(x[2]) for x in tmp]      

        plotting_tools.plotLC(range(len(best_error_train)),best_error_train, best_error_valid, path+"transBest_learning", True, True, "Number")
        plotting_tools.plotLC(range(len(worst_error_train)),worst_error_train, worst_error_valid, path+"transWorst_learning", True, True, "Number")
    return


def lambda_curve(data,lda,options,path):
    data=all_data.loc[(all_data['ttype'] == 2) & (all_data['V']>0.) & (all_data['eta']>0.)]
    datatmp=data.apply(lambda row:[row['#Runcat'],row['eta'],row['V'],row['flux'],row['fluxrat'],row['variable']],axis=1)
    data = datatmp.as_matrix()

    # shuffle up the transient and stable data
    shuffled = np.matrix(generic_tools.shuffle_datasets(data))
    
    # sort the data into a training, validation and testing dataset. This is hardcoded to be 60%, 30% and 10% (respectively) of the total dataset
    train, valid, test = generic_tools.create_datasets(shuffled, int(len(shuffled)*0.6), int(len(shuffled)*0.9))

    print "creating Lambda curve"
    if not os.path.exists("LR_lambda_data.csv"):
        Xtrain, ytrain = train_logistic_regression.create_X_y_arrays(np.matrix([[np.log10(a[1]), np.log10(a[2]), np.log10(a[3]), a[4], a[5]] for a in np.array(train)]))
        Xvalid, yvalid = train_logistic_regression.create_X_y_arrays(np.matrix([[np.log10(a[1]), np.log10(a[2]), np.log10(a[3]), a[4], a[5]] for a in np.array(valid)]))
        
        # check that the lambda (lda) parameter chosen is appropriate for your dataset (Optional but recommended). This parameter controls the 'weighting' given to the different parameters in the model. If the learning curve converges quickly and the validation curve is relatively flat, you are ok having a small lambda value such as 1e-4.
        error_train, error_valid, lambda_vec, lda = train_logistic_regression.validation_curve(Xtrain, ytrain.T, Xvalid, yvalid.T, options)
        tmp = [[lambda_vec[x], error_train[x], error_valid[x]] for x in range(len(lambda_vec))]
        generic_tools.write_test_data(path+"LR_lambda_data.csv",tmp)
    else:
        tmp = generic_tools.extract_data(path+"LR_lambda_data.csv")
        lambda_vec = [float(x[0]) for x in tmp]
        error_train = [float(x[1]) for x in tmp]
        error_valid = [float(x[2]) for x in tmp]      
    plotting_tools.plotLC(lambda_vec, error_train, error_valid, "LR_validation", True, True, r"$\lambda$")
    return

def repeat_curve(anomaly,logistic,transSrc,all_data,lda,options,precis_thresh,recall_thresh,path):

    data=all_data.loc[(all_data['ttype'] == 2) & (all_data['V']>0.) & (all_data['eta']>0.)]
    datatmp=data.apply(lambda row:[row['#Runcat'],row['eta'],row['V'],row['flux'],row['fluxrat'],row['variable']],axis=1)
    data = datatmp.as_matrix()
    stable=all_data.loc[(all_data['ttype'] == 2) & (all_data['V']>0.) & (all_data['eta']>0.) & (all_data['variable'] == 0)]
    stable=stable.apply(lambda row:[row['#Runcat'],row['eta'],row['V'],row['flux'],row['fluxrat'],row['variable']],axis=1)
    stable=stable.as_matrix()
    variable=all_data.loc[(all_data['ttype'] == 2) & (all_data['V']>0.) & (all_data['eta']>0.) & (all_data['variable'] == 1)]
    variable=variable.apply(lambda row:[row['#Runcat'],row['eta'],row['V'],row['flux'],row['fluxrat'],row['variable']],axis=1)
    variable=variable.as_matrix()
    
    
    # check that the results are not dependent upon the subsample of the dataset chosen to train the algorithm by repeating the training a large number of times and checking that the training and validation errors are roughly constant (Optional but recommended).
    print "Creating repeat curve"
    error_train_LR=[]
    error_valid_LR=[]
    error_train_AD=[]
    error_valid_AD=[]
    error_train_TBest=[]
    error_train_TWorst=[]
    error_valid_TBest=[]
    error_valid_TWorst=[]
    randomChoice=sorted(random.sample(range(1000),50))

    # Extract out the possible and candidate transient sources
    if transSrc:
        # Extract out the possible and candidate transient sources
        possTransData = all_data.loc[((all_data['ttype'] == 0) | (all_data['ttype'] == 1)) & (all_data['variable'] == 0)]
        possTransData = possTransData.as_matrix()
        possTransSims = all_data.loc[((all_data['ttype'] == 0) | (all_data['ttype'] == 1)) & (all_data['variable'] == 1)]
        possTransSims = possTransSims.as_matrix()
    
        # Sort out the sigma data for plotting and training
        best_data=all_data[['minRmsSigma','variable']].as_matrix()
        worst_data=all_data[['maxRmsSigma','variable']].as_matrix()

        # shuffle up the transient and stable data
        best = np.matrix(generic_tools.shuffle_datasets(best_data))
        worst = np.matrix(generic_tools.shuffle_datasets(worst_data))

    for counter in range(1000):
        # shuffle up the transient and stable data
        shuffled = np.matrix(generic_tools.shuffle_datasets(data))
    
        # sort the data into a training, validation and testing dataset. This is hardcoded to be 60%, 30% and 10% (respectively) of the total dataset
        train, valid, test = generic_tools.create_datasets(shuffled, int(len(shuffled)*0.6), int(len(shuffled)*0.9))
        train=np.array(train)
        valid=np.array(valid)

        Xtrain, ytrain = train_logistic_regression.create_X_y_arrays(np.matrix([[np.log10(a[1]), np.log10(a[2]), np.log10(a[3]), a[4], a[5]] for a in train]))
        Xvalid, yvalid = train_logistic_regression.create_X_y_arrays(np.matrix([[np.log10(a[1]), np.log10(a[2]), np.log10(a[3]), a[4], a[5]] for a in valid]))
        
        if anomaly:
            if not os.path.exists(path+"AD_repeat_data.csv"):
                if counter in randomChoice:
                    tmp1,tmp2 = train_anomaly_detect.random_test(stable,variable,train, valid, precis_thresh, recall_thresh)
                    error_train_AD.append(tmp1)
                    error_valid_AD.append(tmp2)
                    print counter,error_train_AD[-1],error_valid_AD[-1]

        if logistic:
            if not os.path.exists("LR_repeat_data.csv"):
                initial_theta=np.zeros((Xtrain.shape[1]))
                theta, cost, _, _, _ = optimize.fmin(lambda t: train_logistic_regression.reg_cost_func(t,Xtrain,ytrain.T,lda), initial_theta, **options)
                error_train_LR.append(train_logistic_regression.check_error(Xtrain,ytrain.T,theta))
                error_valid_LR.append(train_logistic_regression.check_error(Xvalid,yvalid.T,theta))

        if transSrc:
            if not os.path.exists("TransBest_repeat_data.csv"):
                # shuffle up the transient and stable data
                best_shuffled = np.matrix(generic_tools.shuffle_datasets(best))
                worst_shuffled = np.matrix(generic_tools.shuffle_datasets(worst))
            
                # sort the data into a training, validation and testing dataset. This is hardcoded to be 60%, 30% and 10% (respectively) of the total dataset
                best_train, best_valid, best_test = generic_tools.create_datasets(best_shuffled, int(len(best_shuffled)*0.6), int(len(best_shuffled)*0.9))
                worst_train, worst_valid, worst_test = generic_tools.create_datasets(worst_shuffled, int(len(worst_shuffled)*0.6), int(len(worst_shuffled)*0.9))
    
                best_plot_data, worst_plot_data, sigBest, sigWorst = train_sigma_margin.find_sigma_margin(best_train, worst_train, detection_threshold)
                error_train_TBest.append(train_sigma_margin.training_error(best_train,sigBest,detection_threshold))
                error_train_TWorst.append(train_sigma_margin.training_error(worst_train,sigWorst,detection_threshold))
                error_valid_TBest.append(train_sigma_margin.training_error(best_valid,sigBest,detection_threshold))
                error_valid_TWorst.append(train_sigma_margin.training_error(worst_valid,sigWorst,detection_threshold))
        
    if anomaly:
        if not os.path.exists(path+"AD_repeat_data.csv"):
            tmp = [[randomChoice[x], error_train_AD[x], error_valid_AD[x]] for x in range(len(randomChoice))]
            generic_tools.write_test_data(path+"AD_repeat_data.csv",tmp)
        else:
            tmp = generic_tools.extract_data(path+"AD_repeat_data.csv")
            randomChoice = [int(x[0]) for x in tmp]
            error_train_AD = [float(x[1]) for x in tmp]
            error_valid_AD = [float(x[2]) for x in tmp] 
        plotting_tools.plotLC(randomChoice, error_train_AD, error_valid_AD, path+"AD_repeat", False, True, "Trial number")
    if logistic:
        if not os.path.exists(path+"LR_repeat_data.csv"):
            tmp = [[x, error_train_LR[x], error_valid_LR[x]] for x in range(len(error_train_LR))]
            generic_tools.write_test_data(path+"LR_repeat_data.csv",tmp)
        else:
            tmp = generic_tools.extract_data(path+"LR_repeat_data.csv")
            randomChoice = [int(x[0]) for x in tmp]
            error_train_LR = [float(x[1]) for x in tmp]
            error_valid_LR = [float(x[2]) for x in tmp] 
        plotting_tools.plotLC(range(len(error_train_LR)), error_train_LR, error_valid_LR, path+"LR_repeat", False, True, "Trial number")
    if transSrc:
        if not os.path.exists(path+"TransBest_repeat_data.csv"):
            tmp = [[x, error_train_TBest[x], error_valid_TBest[x]] for x in range(len(error_train_TBest))]
            generic_tools.write_test_data(path+"TransBest_repeat_data.csv",tmp)
            tmp = [[x, error_train_TWorst[x], error_valid_TWorst[x]] for x in range(len(error_train_TWorst))]
            generic_tools.write_test_data(path+"TransWorst_repeat_data.csv",tmp)
        else:
            tmp = generic_tools.extract_data(path+"TransBest_repeat_data.csv")
            randomChoice = [int(x[0]) for x in tmp]
            error_train_TBest = [float(x[1]) for x in tmp]
            error_valid_TBest = [float(x[2]) for x in tmp] 
            tmp = generic_tools.extract_data(path+"TransWorst_repeat_data.csv")
            randomChoice = [int(x[0]) for x in tmp]
            error_train_TWorst = [float(x[1]) for x in tmp]
            error_valid_TWorst = [float(x[2]) for x in tmp] 
        plotting_tools.plotLC(range(len(error_train_TBest)), error_train_TBest, error_valid_TBest, path+"transBest_repeat", False, True, "Trial number")
        plotting_tools.plotLC(range(len(error_train_TWorst)), error_train_TWorst, error_valid_TWorst, path+"transWorst_repeat", False, True, "Trial number")
    return
