import generic_tools
import numpy as np
from multiprocessing import Pool
from scipy.interpolate import griddata
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['legend.loc'] = 'best'
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
import os


def trial_data(args):
    # Find the precision and recall for a given pair of thresholds
    data,sigma1,sigma2 = args

    # Sort data into transient and non-transient
    xvals = [float(x[0]) for x in data if float(x[-1]) != 0.]
    yvals = [float(x[1]) for x in data if float(x[-1]) != 0.]
    xstable = [float(x[0]) for x in data if float(x[-1]) == 0.]
    ystable = [float(x[1]) for x in data if float(x[-1]) == 0.]

    # Find the thresholds for a given sigma, by fitting data with a Gaussian model
    sigcutx,paramx,range_x = generic_tools.get_sigcut([float(x[0]) for x in data if float(x[-1]) == 0.],sigma1)
    sigcuty,paramy,range_y = generic_tools.get_sigcut([float(x[1]) for x in data if float(x[-1]) == 0.],sigma2)

    # Count up the different numbers of tn, tp, fp, fn
    fp=len([z for z in range(len(xstable)) if (xstable[z]>sigcutx and ystable[z]>sigcuty)]) # False Positive
    tn=len([z for z in range(len(xstable)) if (xstable[z]<sigcutx or ystable[z]<sigcuty)]) # True Negative
    tp=len([z for z in range(len(xvals)) if (xvals[z]>sigcutx and yvals[z]>sigcuty)]) # True Positive
    fn=len([z for z in range(len(xvals)) if (xvals[z]<sigcutx or yvals[z]<sigcuty)]) # False Negative

    # Use these values to calculate the precision and recall values
    precision, recall = generic_tools.precision_and_recall(tp,fp,fn)
    return [sigma1, sigma2, precision, recall]

def multiple_trials(data,filename):
    # Find the precision and recall for all combinations of the sigma thresholds
    sigmas=np.arange(0.,3.5,(3.5/500.))
    pool = Pool(processes=20)              # start 20 worker processes
    inputs = range(2)
    
    # Loop through all the trial sigmas and on multiple workers, then append to a file
    for sigma1 in sigmas:
        sigma_data = pool.map(trial_data, [(data,sigma1,sigma2) for sigma2 in sigmas])
        with open(filename,'a') as f_handle:
            np.savetxt(f_handle,sigma_data)
        f_handle.close()
    pool.close() # Close the worker processes once training is complete
    return

def tests(args):
    # Test multiple input precision and recall values to check out if we are meeting and exceeding the input parameters
    xi,yi,zi1,zi2, data, xvals, yvals, xstable, ystable, precis, recall = args

    # Find the combination of x and y which is closest to the two thresholds
    combinations=[[xi[a][b],yi[a][b],zi1[a][b],zi2[a][b]] for a in range(len(zi1)) for b in range(len(zi1[0])) if zi1[a][b]>=precis]
    ID=np.array([((a[2]-precis)**2. + (a[3]-recall)**2.) for a in combinations]).argmin()
    above_thresh_sigma=combinations[ID]
    
    # Find the thresholds for these sigmas, by fitting the observed data with a Gaussian model
    sigcutx,paramx,range_x = generic_tools.get_sigcut([float(x[0]) for x in data],above_thresh_sigma[0])
    sigcuty,paramy,range_y = generic_tools.get_sigcut([float(x[1]) for x in data],above_thresh_sigma[1])

    # Count up the different numbers of tp, fp, fn
    fp=len([z for z in range(len(xstable)) if (xstable[z]>sigcutx and ystable[z]>sigcuty)]) # False Positive
    tp=len([z for z in range(len(xvals)) if (xvals[z]>sigcutx and yvals[z]>sigcuty)]) # True Positive
    fn=len([z for z in range(len(xvals)) if (xvals[z]<sigcutx or yvals[z]<sigcuty)]) # False Negative
    
    # Use these values to calculate the precision and recall values obtained with the trained threshold.
    # If the test is successful, the outputs should meet or exceed the input parameters.
    results1, results2 = generic_tools.precision_and_recall(tp,fp,fn)

    return [precis, recall, results1, results2]

def check_method_works(xi,yi,zi1,zi2, data,above_thresh_sigma,path):
    # Multiple trials using input precisions and recalls between 0 and 1 are conducted to confirm that this method provides the input precision and recall
    trials = np.arange(0.0,1.,1./500.)
    trials2 = np.arange(0.0,1.,1./500.)
    Z1=[]
    Z2=[]
    X=[]
    Y=[]

    print "Producing anomaly detection test data"
    if not os.path.exists(path+'anomaly_test_data.txt'): # Only run the test if the output data are not already available
        open(path+'anomaly_test_data.txt','w').close()
        pool = Pool(processes=20)              # start 20 worker processes
        inputs = range(2)

        # Sort data into variable and non-variable
        xvals = [float(x[0]) for x in data if float(x[-1]) != 0.]
        yvals = [float(x[1]) for x in data if float(x[-1]) != 0.]
        xstable = [float(x[0]) for x in data if float(x[-1]) == 0.]
        ystable = [float(x[1]) for x in data if float(x[-1]) == 0.]

        # Run through each of the precision and recall trials and calculate the output precision and recalls
        for precis in trials:
            test_data = pool.map(tests, [(xi,yi,zi1,zi2, data, xvals, yvals, xstable, ystable, precis, recall) for recall in trials2])
            with open(path+'anomaly_test_data.txt','a') as f_handle: # append data to a file
                np.savetxt(f_handle,test_data)
            f_handle.close()
        pool.close() # close all the workers

    # sort the data for plotting
    test_data = np.genfromtxt(path+'anomaly_test_data.txt')
    X = [x[0] for x in test_data] # input precision
    Y = [x[1] for x in test_data] # input recall
    Z1 = [(x[2]-x[0]) for x in test_data] # output precision
    Z2 = [(x[3]-x[1]) for x in test_data] # output precision
    
    print("Anomaly Detection test data collated, now gridding")
    xi,yi = np.mgrid[0:1:1000j, 0:1:1000j]
    zi1 = griddata((X, Y), Z1, (xi, yi), method='cubic')
    zi2 = griddata((X, Y), Z2, (xi, yi), method='cubic')
    # Plot results
    # Settings
    nullfmt   = NullFormatter()         # no labels
    fig = plt.figure(1,figsize=(5,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cax = fig.add_axes([0.1, 0.95, 0.8, 0.03])
    fig.subplots_adjust(hspace = .001, wspace = 0.001)
    levels=np.arange(-0.2,0.25,0.05)
    ax2.set_xlabel(r'Input Precision', fontsize=24)
    ax1.set_ylabel(r'Input Recall', fontsize=24)
    ax2.set_ylabel(r'Input Recall', fontsize=24)
    fontP = FontProperties()
    fontP.set_size('x-small')
    ax1.set_xlim([0.0,1.0])
    ax2.set_xlim([0.0,1.0])
    ax1.set_ylim([0.0,0.999])
    ax2.set_ylim([0.0,0.999])
    ax1.xaxis.set_major_formatter(nullfmt)
    ax2.set_xlim( ax1.get_xlim() )
    ax1.text(2.5, 3.5, 'Precision', bbox=dict(facecolor='white'), fontsize=24)
    ax2.text(2.5, 3.5, 'Recall', bbox=dict(facecolor='white'), fontsize=24)
    CS1 = ax1.contourf(xi,yi,zi1,levels, cmap=plt.get_cmap('RdBu'),alpha=1,extend='both')
    CS2 = ax2.contourf(xi,yi,zi2,levels, cmap=plt.get_cmap('RdBu'),alpha=1,extend='both')
    fig.colorbar(CS2, cax, orientation='horizontal')
    ax1.axvline(x=above_thresh_sigma[0], linewidth=2, color='k', linestyle='--')
    ax2.axvline(x=above_thresh_sigma[0], linewidth=2, color='k', linestyle='--')
    ax1.axhline(y=above_thresh_sigma[1], linewidth=2, color='k', linestyle='--')
    ax2.axhline(y=above_thresh_sigma[1], linewidth=2, color='k', linestyle='--')
    plt.savefig(path+'sim_check_precisions_and_recalls.png')
    plt.close()
    return

def find_best_sigmas(precis_thresh,recall_thresh,data,tests, data2,plots,path):
    # Gridding the data for precision and recall
    X=[float(x[0]) for x in data]
    Y=[float(x[1]) for x in data]
    Z1=[float(x[2]) for x in data]
    Z2=[float(x[3]) for x in data]
    xi,yi = np.mgrid[0:3.5:1000j, 0:3.5:1000j]
    zi1 = griddata((X, Y), Z1, (xi, yi), method='cubic', fill_value=1.)
    zi2 = griddata((X, Y), Z2, (xi, yi), method='cubic', fill_value=0.)

    # Find the combination of x and y which is closest to the two thresholds
    combinations=[[xi[a][b],yi[a][b],zi1[a][b],zi2[a][b]] for a in range(len(zi1)) for b in range(len(zi1[0])) if zi1[a][b]>=precis_thresh]
    ID=np.array([((a[2]-precis_thresh)**2. + (a[3]-recall_thresh)**2.) for a in combinations]).argmin()
    above_thresh_sigma=combinations[ID]

    if plots:
        # Plot results
        # Settings
        nullfmt   = NullFormatter()         # no labels
        fig = plt.figure(1,figsize=(5,10))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        cax = fig.add_axes([0.1, 0.95, 0.8, 0.03])
        fig.subplots_adjust(hspace = .001, wspace = 0.001)
        levels=np.arange(0.0,1.1,0.1)
        levels2=[0.1, 0.3, 0.5, 0.7, 0.9]
        ax2.set_xlabel(r'$\sigma$ threshold ($\eta_{\nu}$)', fontsize=18)
        ax1.set_ylabel(r'$\sigma$ threshold ($V_{\nu}$)', fontsize=18)
        ax2.set_ylabel(r'$\sigma$ threshold ($V_{\nu}$)', fontsize=18)
        fontP = FontProperties()
        fontP.set_size('x-small')
        ax1.set_xlim([0.0,3.5])
        ax2.set_xlim([0.0,3.5])
        ax1.set_ylim([0.0,3.49])
        ax2.set_ylim([0.0,3.49])
        ax1.xaxis.set_major_formatter(nullfmt)
        ax2.set_xlim( ax1.get_xlim() )
        ax1.text(2.5, 3.0, 'Precision', bbox=dict(facecolor='white'), fontsize=18)
        ax2.text(2.5, 3.0, 'Recall', bbox=dict(facecolor='white'), fontsize=18)
        CS1 = ax1.contourf(xi,yi,zi1,levels, cmap=plt.get_cmap('Blues'),alpha=1,extend='both')
        CS2 = ax2.contourf(xi,yi,zi2,levels, cmap=plt.get_cmap('Blues'),alpha=1,extend='both')
        fig.colorbar(CS2, cax, orientation='horizontal')
        ax1.axvline(x=above_thresh_sigma[0], linewidth=2, color='k', linestyle='--')
        ax2.axvline(x=above_thresh_sigma[0], linewidth=2, color='k', linestyle='--')
        ax1.axhline(y=above_thresh_sigma[1], linewidth=2, color='k', linestyle='--')
        ax2.axhline(y=above_thresh_sigma[1], linewidth=2, color='k', linestyle='--')
        plt.savefig(path+'sim_precisions_and_recalls.png')
        plt.close()

    print("Best sigmas found:"+str(above_thresh_sigma[0])+', '+str(above_thresh_sigma[1]))
    return above_thresh_sigma[0],above_thresh_sigma[1]

def learning_curve(stable,variable,train, valid, precis_thresh, recall_thresh, rangeNums,path):
    validErr=[]
    valid_list=np.unique([x[0] for x in valid])
    output=open(path+'sigma_train.txt','w')
    trainErr=[]
    for num in rangeNums:
        if num>0:
            filename = open(path+"temp_sigma_data.txt", "w")
            filename.write('')
            filename.close()
            trainTMP=train[:num,:]
            train_list=np.unique([int(x[0]) for x in trainTMP])
            multiple_trials([[np.log10(float(x[1])), np.log10(float(x[2])), float(x[-1])] for x in trainTMP if float(x[1]) > 0 if float(x[2]) > 0],path+"temp_sigma_data.txt")
            data2=np.genfromtxt(path+'temp_sigma_data.txt', delimiter=' ')
            data=[[np.log10(float(trainTMP[n][1])),np.log10(float(trainTMP[n][2])),trainTMP[n][5],float(trainTMP[n][-1])] for n in range(num) if float(trainTMP[n][1]) > 0 if float(trainTMP[n][2]) > 0]
            best_sigma1, best_sigma2 = find_best_sigmas(precis_thresh,recall_thresh,data2,False,data,False,path)
            
            # Find the thresholds for a given sigma (in log space)
            sigcutx,paramx,range_x = generic_tools.get_sigcut([a[0] for a in data if a[3]==0.],best_sigma1)
            sigcuty,paramy,range_y = generic_tools.get_sigcut([a[1] for a in data if a[3]==0.],best_sigma2)

            # Calculate the training error
            fp=len([[z[0],float(z[1]),float(z[2]),float(z[3]),float(z[4]),'FP'] for z in stable if (float(z[1])>=10.**sigcutx and float(z[2])>=10.**sigcuty) if int(z[0]) in train_list]) # False Positive
            fn=len([[z[0],float(z[1]),float(z[2]),float(z[3]),float(z[4]),'FN'] for z in variable if (float(z[1])<10.**sigcutx or float(z[2])<10.**sigcuty) if int(z[0]) in train_list]) # False Negative
            trainErr.append(check_error(fp,fn,len(trainTMP)))

            # Caluculate the validation error
            fp=len([[z[0],float(z[1]),float(z[2]),float(z[3]),float(z[4]),'FP'] for z in stable if (float(z[1])>=10.**sigcutx and float(z[2])>=10.**sigcuty) if int(z[0]) in valid_list]) # False Positive
            fn=len([[z[0],float(z[1]),float(z[2]),float(z[3]),float(z[4]),'FN'] for z in variable if (float(z[1])<10.**sigcutx or float(z[2])<10.**sigcuty) if int(z[0]) in valid_list]) # False Negative
            validErr.append(check_error(fp,fn,len(valid)))
            output.write(str(num)+','+str(trainErr)+','+str(validErr))
    output.close()
     
    return  trainErr, validErr
        
def random_test(stable,variable,train, valid, precis_thresh, recall_thresh,path):

    multiple_trials([[np.log10(float(x[1])), np.log10(float(x[2])), float(x[-1])] for x in train if float(x[1]) > 0 if float(x[2]) > 0],"temp_sigma_data.txt")
    data2=np.genfromtxt('temp_sigma_data.txt', delimiter=' ')
    data=[[np.log10(float(train[n][1])),np.log10(float(train[n][2])),train[n][5],float(train[n][-1])] for n in range(len(train)) if float(train[n][1]) > 0 if float(train[n][2]) > 0]
    best_sigma1, best_sigma2 = find_best_sigmas(precis_thresh,recall_thresh,data2,False,data,False)

    # Find the thresholds for a given sigma (in log space)
    sigcutx,paramx,range_x = generic_tools.get_sigcut([a[0] for a in data if a[3]==0.],best_sigma1)
    sigcuty,paramy,range_y = generic_tools.get_sigcut([a[1] for a in data if a[3]==0.],best_sigma2)

    trainIDs=[str(int(x[0])) for x in train]
    validIDs=[str(int(x[0])) for x in valid]
    
    # Calculate the training error
    fp=len([[z[0],float(z[1]),float(z[2]),float(z[3]),float(z[4]),'FP'] for z in stable if (float(z[1])>=10.**sigcutx and float(z[2])>=10.**sigcuty) if z[0] in trainIDs]) # False Positive
    fn=len([[z[0],float(z[1]),float(z[2]),float(z[3]),float(z[4]),'FN'] for z in variable if (float(z[1])<10.**sigcutx or float(z[2])<10.**sigcuty) if z[0] in trainIDs]) # False Negative
    trainErr = check_error(fp,fn,len(train))

    # Caluculate the validation error
    fp=len([[z[0],float(z[1]),float(z[2]),float(z[3]),float(z[4]),'FP'] for z in stable if (float(z[1])>=10.**sigcutx and float(z[2])>=10.**sigcuty) if z[0] in validIDs]) # False Positive
    fn=len([[z[0],float(z[1]),float(z[2]),float(z[3]),float(z[4]),'FN'] for z in variable if (float(z[1])<10.**sigcutx or float(z[2])<10.**sigcuty) if z[0] in validIDs]) # False Negative
    validErr = check_error(fp,fn,len(valid))
    
    plotLC(len(validErr), trainErr, validErr, path+'random', False, True, 'Trial number')
    return trainErr, validErr
        
def check_error(fp,fn,total):
    classif_err = float(fp+fn)/float(total)
    return classif_err
