# transient_margins.py
# See also FilterVariables.ipynb
#
# A code to find the best sigma margins to identify transient sources
#
# Import all the dependencies and generic setup
import matplotlib
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['legend.loc'] = 'best'
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
import os
import glob
import numpy as np
import pandas as pd
from scipy import optimize
import tkp.db
import logging
query_loglevel = logging.WARNING  # Set to INFO to see queries, otherwise WARNING
import sqlalchemy
from sqlalchemy import *
from sqlalchemy.orm import relationship
import sys
sys.path.append('../')
#from dblogin import * # This file contains all the variables required to connect to the database
#from database_tools import dbtools
from tools import tools
from plotting import plot_varib_params as pltvp
from machine_learning import train_sigma_margin
from machine_learning import generic_tools

matplotlib.rcParams.update({'font.size': 26})
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['xtick.major.size'] = 12
matplotlib.rcParams['ytick.major.size'] = 12
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['xtick.minor.size'] = 8
matplotlib.rcParams['ytick.minor.size'] = 8
matplotlib.rcParams['xtick.minor.width'] = 1
matplotlib.rcParams['ytick.minor.width'] = 1

# The input data and thresholds
tests = False
plots = False
path='ml_csv_files/'
stableData = path+'stable_sources.csv'
simulatedData = path+'sim_*_trans_data.csv'
detection_threshold = 8.

# Load the data and give appropriate labels
all_data = generic_tools.load_data(stableData,simulatedData)
# Identify all the new sources detected during the pipeline run
all_data2=all_data.loc[all_data['ttype'] != 2]
# Create a histogram of the detection thresholds assuming the source
# was found in the lowest RMS region of image
x=all_data2.loc[all_data2['variable'] == 0]
x2=all_data2.loc[all_data2['variable'] == 1]
plt.figure(figsize=(12,10))
plt.xscale('log')
bins=np.logspace(np.log10(min(x['minRmsSigma'])), np.log10(max(x2['minRmsSigma'])), num=50, endpoint=True, base=10.0)
plt.hist(x['minRmsSigma'], bins=bins, histtype='stepfilled', color='b')
plt.hist(x2['minRmsSigma'], bins=bins, histtype='step', linewidth=2, color='r')
plt.xlim(0.1,1e4)
plt.xlabel(r'Expected detection significance of source ($\sigma$)', fontsize=24)
plt.axvline(x=detection_threshold, linestyle='--', linewidth=2, color='k')
plt.savefig(path+'lowRMS_sigma_histogram.png')

# Create a histogram of the detection thresholds assuming the source
# was found in the lowest RMS region of image
plt.figure(figsize=(12,10))
plt.xscale('log')
bins=np.logspace(np.log10(min(x['maxRmsSigma'])), np.log10(max(x2['minRmsSigma'])), num=50, endpoint=True, base=10.0)
plt.hist(x['maxRmsSigma'], bins=bins, histtype='stepfilled', color='b')
plt.hist(x2['maxRmsSigma'], bins=bins, histtype='step', linewidth=2, color='r')
plt.xlim(0,1e4)
plt.xlabel(r'Expected detection significance of source ($\sigma$)', fontsize=24)
plt.axvline(x=detection_threshold, linestyle='--', linewidth=2, color='k')
plt.savefig(path+'highRMS_sigma_histogram.png')

# Find the best margins to maximise detection and reliability
best_data=all_data[['minRmsSigma','variable']].as_matrix()
worst_data=all_data[['maxRmsSigma','variable']].as_matrix()
best_plot_data, worst_plot_data, sigBest, sigWorst = train_sigma_margin.find_sigma_margin(best_data,worst_data, detection_threshold)

# Search and identify the optimal sigma margin for the best and
# worst parts of the image
sigWorst, sigBest = train_sigma_margin.plot_diagnostic(best_plot_data,worst_plot_data,path)

# Identify the ids of interesting transient candidates assuming they're in the
# worst part of the image
tmpData = all_data.loc[(all_data['maxRmsSigma'] >= sigWorst+detection_threshold) & (all_data['variable']==0)]
tmpData.to_csv(path+'candidate_transients_worst_region.csv',index=False)

# Identify the ids of interesting transient candidates assuming they're in the
# best part of the image
tmpData = all_data.loc[(all_data['minRmsSigma'] >= sigBest+detection_threshold) & (all_data['variable']==0)]
tmpData.to_csv(path+'candidate_transients_best_region.csv',index=False)

# And the best thresholds are:
print('Lowest RMS region threshold: '+str(sigBest+detection_threshold)+' sigma')
print('Highest RMS region threshold: '+str(sigWorst+detection_threshold)+' sigma')

