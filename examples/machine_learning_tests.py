# machine_learning_tests.py
#
# This script performs the standard machine learning tests used in
# Rowlinson et al. (2018). As the tests are very slow, there is not a
# Jupyter Notebook example and it is recommended that this script is
# run in a screen.
#

import sys
sys.path.append('../')
from machine_learning import generic_tools
from machine_learning import MLtests


# Inputs and settings
precis_thresh = 0.99
recall_thresh = 0.95
lda=0.1
detection_threshold=8
path='ml_csv_files/'
stableData = path+'stable_sources.csv'
simulatedData = path+'sim_*_trans_data.csv'
anomaly = False
logistic = True
transSrc = False
# setting the options for the scipy optimise function
options = {'full_output': True, 'maxiter': 5000, 'ftol': 1e-4, 'maxfun': 5000, 'disp': True}

#load the data required
all_data = generic_tools.load_data(stableData,simulatedData)


#create the lambda curve for the logistic regreassion algorithm
MLtests.lambda_curve(all_data,lda,options,path)

# Create the learning curve for all three machine learning algorithms
MLtests.learning_curve(anomaly,logistic,transSrc,all_data,lda,options,precis_thresh,recall_thresh,path,detection_threshold)

# Create the repitition curve for all three machine learning algorithms
MLtests.repeat_curve(anomaly,logistic,transSrc,all_data,lda,options,precis_thresh,recall_thresh,path,detection_threshold)
