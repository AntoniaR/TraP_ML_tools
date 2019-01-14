# anomaly_detection.py
# See also anomaly_detection.ipynb
#
# A code to use the anomaly detection strategy to train the thresholds for the
# variability parameters eta and V
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
from dblogin import * # This file contains all the variables required to connect to the database
from database_tools import dbtools
from tools import tools
from plotting import plot_varib_params as pltvp
from machine_learning import train_anomaly_detect
from machine_learning import generic_tools
import random
from matplotlib.colors import LogNorm
import seaborn as sns

# The input data and thresholds
tests = False
plots = False
precis_thresh = 0.95
recall_thresh = 0.95
path='ml_csv_files/'
stableData = path+'stable_sources.csv'
simulatedData = path+'sim_*_trans_data.csv'

def newCmap(colour):
    n,r,g,b = colour
    cdict = {'red':((0,1,1),(1,r,r)),
            'green':((0,1,1),(1,g,g)),
            'blue':((0,1,1),(1,b,b))}
    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

# Load the data and give appropriate labels
all_data = generic_tools.load_data(stableData,simulatedData)

# Load the simulations and only keep those with a full lightcurve and with variability parameters > 0
all_data=all_data.loc[(all_data['ttype'] == 2) & (all_data['V']>0.) & (all_data['eta']>0.)]

# put the training data into the format required for the training
train_data = all_data.apply(lambda row:[np.log10(row['eta']),np.log10(row['V']),row['variable'],row['label']],axis=1)
train_data = train_data.as_matrix()

# Obtain the training data if it doesn't already exist on disk (SLOW!)
if not os.path.exists(path+'sigma_data.txt'):
    filename = open(path+"sigma_data.txt", "w")
    filename.write('')
    filename.close()
    train_anomaly_detect.multiple_trials(train_data,path+"sigma_data.txt")
data2=np.genfromtxt(path+'sigma_data.txt', delimiter=' ')

# Using the training data, find the sigma threshold combination that best fits input thresholds
best_sigma1, best_sigma2 = train_anomaly_detect.find_best_sigmas(precis_thresh,recall_thresh,data2,tests,train_data,plots,path)
print 'sigma_(eta_nu)='+str(best_sigma1)+', sigma_(V_nu)='+str(best_sigma2)

# Find the eta and V thresholds for the data
sigcutx,paramx,range_x = generic_tools.get_sigcut([a[0] for a in train_data if a[2]==0.],best_sigma1)
sigcuty,paramy,range_y = generic_tools.get_sigcut([a[1] for a in train_data if a[2]==0.],best_sigma2)
print(r'Gaussian Fit $\eta$: '+str(round(10.**paramx[0],2))+'(+'+str(round((10.**(paramx[0]+paramx[1])-10.**paramx[0]),2))+' '+str(round((10.**(paramx[0]-paramx[1])-10.**paramx[0]),2))+')')
print(r'Gaussian Fit $V$: '+str(round(10.**paramy[0],2))+'(+'+str(round((10.**(paramy[0]+paramy[1])-10.**paramy[0]),2))+' '+str(round((10.**(paramy[0]-paramy[1])-10.**paramy[0]),2))+')')
print 'Eta_nu threshold='+str(10.**sigcutx)+', V_nu threshold='+str(10.**sigcuty)
threshx=10.**sigcutx
threshy=10.**sigcuty

# Calculate the false positives (FP), true negatives (TN), true positives (TP) and false negatives (FN)
all_data.loc[(((all_data['eta']<threshx) | (all_data['V']<threshy)) & (all_data['variable'] == 1)),'classified'] = 'FN'
all_data.loc[((all_data['eta']>=threshx) & (all_data['V']>=threshy) & (all_data['variable'] == 1)),'classified'] = 'TP'
all_data.loc[(((all_data['eta']<threshx) | (all_data['V']<threshy)) & (all_data['variable'] == 0)),'classified'] = 'TN'
all_data.loc[((all_data['eta']>=threshx) & (all_data['V']>=threshy) & (all_data['variable'] == 0)),'classified'] = 'FP'

# Find candidates
all_data.loc[(all_data['classified'] == 'FP')].to_csv(path+'AD_candidate_variables.csv',index=False)

# Calculate the precision and recall
precision, recall =  generic_tools.precision_and_recall(len(all_data.loc[(all_data['classified'] == 'TP')]),len(all_data.loc[(all_data['classified'] == 'FP')]),len(all_data.loc[(all_data['classified'] == 'FN')]))
print "Precision: "+str(precision)+", Recall: "+str(recall)

# Create eta V plot showing training results
plotdata=all_data

frequencies=['TN','TP','FN','FP']
col=['k','b','g','r']


nullfmt   = NullFormatter()         # no labels
fontP = FontProperties()
fontP.set_size('small')
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left+width+0.02
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]
fig = plt.figure(1,figsize=(12,12))
axScatter = fig.add_subplot(223, position=rect_scatter)
plt.xlabel(r'$\eta_{\nu}$', fontsize=28)
plt.ylabel(r'$V_{\nu}$', fontsize=28)
axHistx=fig.add_subplot(221, position=rect_histx)
axHisty=fig.add_subplot(224, position=rect_histy)
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)
axHistx.axes.yaxis.set_ticklabels([])
axHisty.axes.xaxis.set_ticklabels([])
                    


for i in range(len(frequencies)):
    plotdataTMP=plotdata.loc[(plotdata['classified']==frequencies[i])]
    xdata_var=np.log10(plotdataTMP['eta'])
    ydata_var=np.log10(plotdataTMP['V'])
    if frequencies[i] != 'TN':
        axScatter.scatter(xdata_var, ydata_var,color=col[i], s=20., zorder=5)
freq_labels=[f for f in frequencies if f!='TN']
axScatter.legend(freq_labels,loc=4, prop=fontP)

        
for i in range(len(frequencies)):
    plotdataTMP=plotdata.loc[(plotdata['classified']==frequencies[i])]
    xdata_var=np.log10(plotdataTMP['eta'])
    ydata_var=np.log10(plotdataTMP['V'])
    if frequencies[i] == 'TN':
        sns.kdeplot(np.log10(plotdataTMP.eta), np.log10(plotdataTMP.V), 
                    n_levels=1000, zorder=i, shade_lowest=False, shade=True, 
                    color='k', ax=axScatter, alpha=1)


x = np.log10(plotdata['eta'])
y = np.log10(plotdata['V'])

axHistx.hist(x, bins=pltvp.make_bins(x), normed=1, histtype='stepfilled', color='k',alpha=0.2)
axHisty.hist(y, bins=pltvp.make_bins(y), normed=1, histtype='stepfilled', orientation='horizontal', color='k', alpha=0.2)

xmin=-5#int(min(x)-1.1)
xmax=int(max(x)+1.1)
ymin=-3#int(min(y)-1.1)
ymax=int(max(y)+1.1)
xvals=range(xmin,xmax)
xtxts=[r'$10^{'+str(a)+'}$' for a in xvals]
yvals=range(ymin,ymax)
ytxts=[r'$10^{'+str(a)+'}$' for a in yvals]
axScatter.set_xlim([xmin,xmax])
axScatter.set_ylim([ymin,ymax])
axScatter.set_xticks(xvals)
axScatter.set_xticklabels(xtxts, fontsize=20)
axScatter.set_yticks(yvals)
axScatter.set_yticklabels(ytxts, fontsize=20)
axHistx.set_xlim( axScatter.get_xlim())
axHisty.set_ylim( axScatter.get_ylim())

if sigcutx != 0 or sigcuty != 0:
    axHistx.axvline(x=sigcutx, linewidth=2, color='k', linestyle='--')
    axHisty.axhline(y=sigcuty, linewidth=2, color='k', linestyle='--')
    axScatter.axhline(y=sigcuty, linewidth=2, color='k', linestyle='--')
    axScatter.axvline(x=sigcutx, linewidth=2, color='k', linestyle='--')

range_x,fitx = pltvp.gaussian_fit(x,paramx)
axHistx.plot(range_x,fitx, 'k:', linewidth=6)
range_y,fity = pltvp.gaussian_fit(y,paramy)
axHisty.plot(fity,range_y, 'k:', linewidth=6)
axScatter.set_xlabel(r'$\eta_{\nu}$', fontsize=28)
axScatter.set_ylabel(r'$V_{\nu}$', fontsize=28)
plt.savefig(path+'AD_scatter_hist.png')

plt.close()

# Create the diagnostic plot
fig = plt.figure(1,figsize=(12,12))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
fontP = FontProperties()
fontP.set_size('small')
fig.subplots_adjust(hspace = .001, wspace = 0.001)


for i in range(len(frequencies)):
    plotdataTMP=plotdata.loc[(plotdata['classified']==frequencies[i])]
    xdata_ax3=np.log10(plotdataTMP['flux'])
    xdata_ax4=plotdataTMP['fluxrat']
    ydata_ax1=np.log10(plotdataTMP['eta'])
    ydata_ax3=np.log10(plotdataTMP['V'])
    if frequencies[i] != 'TN':
        ax1.scatter(xdata_ax3, ydata_ax1,color=col[i], s=20., zorder=i)
        ax2.scatter(xdata_ax4, ydata_ax1,color=col[i], s=20., zorder=i)
        ax3.scatter(xdata_ax3, ydata_ax3,color=col[i], s=20., zorder=i)
        ax4.scatter(xdata_ax4, ydata_ax3,color=col[i], s=20., zorder=i)


ax4.legend(freq_labels, loc=4, prop=fontP)


for i in range(len(frequencies)):
    plotdataTMP=plotdata.loc[(plotdata['classified']==frequencies[i])]
    xdata_ax3=np.log10(plotdataTMP['flux'])
    xdata_ax4=plotdataTMP['fluxrat']
    ydata_ax1=np.log10(plotdataTMP['eta'])
    ydata_ax3=np.log10(plotdataTMP['V'])
    if frequencies[i] == 'TN':
        sns.kdeplot(np.log10(plotdataTMP.flux), np.log10(plotdataTMP.eta), 
                    n_levels=1000, zorder=i, shade_lowest=False, shade=True, 
                    color='k', ax=ax1, alpha=1)
        sns.kdeplot(plotdataTMP.fluxrat, np.log10(plotdataTMP.eta), 
                    n_levels=1000, zorder=i, shade_lowest=False, shade=True, 
                    color='k', ax=ax2, alpha=1)
        sns.kdeplot(np.log10(plotdataTMP.flux), np.log10(plotdataTMP.V), 
                    n_levels=1000, zorder=i, shade_lowest=False, shade=True, 
                    color='k', ax=ax3, alpha=1)
        sns.kdeplot(plotdataTMP.fluxrat, np.log10(plotdataTMP.V), 
                    n_levels=1000, zorder=i, shade_lowest=False, shade=True, 
                    color='k', ax=ax4, alpha=1)


Xax3=np.log10(plotdata['flux'])
Xax4=plotdata['fluxrat']
Yax1=np.log10(plotdata['eta'])
Yax3=np.log10(plotdata['V'])
    
if sigcutx != 0 or sigcuty != 0:
    ax1.axhline(y=sigcutx, linewidth=2, color='k', linestyle='--')
    ax2.axhline(y=sigcutx, linewidth=2, color='k', linestyle='--')
    ax3.axhline(y=sigcuty, linewidth=2, color='k', linestyle='--')
    ax4.axhline(y=sigcuty, linewidth=2, color='k', linestyle='--')

xmin_ax3=int(min(Xax3)-1.1)
xmax_ax3=int(max(Xax3)+1.1)
xmin_ax4=0.9
xmax_ax4=int(max(xdata_ax4))+1.5
ymin_ax1=-5#int(min(Yax1)-1.1)
ymax_ax1=int(max(Yax1)+1.1)
ymin_ax3=-3#int(min(Yax3)-1.1)
ymax_ax3=int(max(Yax3)+1.1)

xvals_ax3=range(xmin_ax3,xmax_ax3)
xtxts_ax3=[r'$10^{'+str(a)+'}$' for a in xvals_ax3]
yvals_ax1=range(ymin_ax1,ymax_ax1)
ytxts_ax1=[r'$10^{'+str(a)+'}$' for a in yvals_ax1]
yvals_ax3=range(ymin_ax3,ymax_ax3)
ytxts_ax3=[r'$10^{'+str(a)+'}$' for a in yvals_ax3]

ax1.set_ylim(ymin_ax1,ymax_ax1)
ax3.set_ylim(ymin_ax3,ymax_ax3)
ax3.set_xlim(xmin_ax3,xmax_ax3)
ax4.set_xlim(xmin_ax4,xmax_ax4)


ax3.set_xticks(xvals_ax3)
ax3.set_xticklabels(xtxts_ax3, fontsize=12)
ax1.set_yticks(yvals_ax1)
ax2.set_yticks(yvals_ax1)
ax1.set_yticklabels(ytxts_ax1, fontsize=12)
ax3.set_yticks(yvals_ax3)
ax4.set_yticks(yvals_ax3)
ax3.set_yticklabels(ytxts_ax3, fontsize=12)

ax1.set_xlim( ax3.get_xlim() )
ax4.set_ylim( ax3.get_ylim() )
ax2.set_xlim( ax4.get_xlim() )
ax2.set_ylim( ax1.get_ylim() )

ax1.xaxis.set_major_formatter(nullfmt)
ax4.yaxis.set_major_formatter(nullfmt)
ax2.xaxis.set_major_formatter(nullfmt)
ax2.yaxis.set_major_formatter(nullfmt)

ax1.set_ylabel(r'$\eta_\nu$', fontsize=28)
ax2.set_ylabel('')
ax3.set_ylabel(r'$V_\nu$', fontsize=28)
ax3.set_xlabel('Max Flux (Jy)', fontsize=24)
ax4.set_xlabel('Max Flux / Median Flux', fontsize=24)
ax4.set_ylabel('')



plt.savefig(path+'AD_diagnostic_plots.png')
