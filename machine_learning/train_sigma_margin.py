import numpy as np
import pandas as dp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['legend.loc'] = 'best'
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
import generic_tools

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

def training_error(data,sigma,detection_threshold):
    fn=len([a[0,0] for a in data if a[0,0]<(detection_threshold+sigma) if a[0,1]==1])
    fp=len([a[0,0] for a in data if a[0,0]>=(detection_threshold+sigma) if a[0,1]==0])
    return float(fp+fn) / float(len(data))
    

def find_sigma_margin(best_data, worst_data, detection_threshold):
    # find the precision, recall and F-score for different margins using the best and worst expected significances
    
    sigma_thresh=np.arange(0.,100.,1)
    best_plot_data=[]
    worst_plot_data=[]

    for sigma in sigma_thresh:
        best_tp=len([best_data[a,0] for a in range(len(best_data)) if best_data[a,0]>=(detection_threshold+sigma) if best_data[a,1]==1])
        best_fn=len([best_data[a,0] for a in range(len(best_data)) if best_data[a,0]<(detection_threshold+sigma) if best_data[a,1]==1])
        best_fp=len([best_data[a,0] for a in range(len(best_data)) if best_data[a,0]>=(detection_threshold+sigma) if best_data[a,1]==0])
        best_tn=len([best_data[a,0] for a in range(len(best_data)) if best_data[a,0]<(detection_threshold+sigma) if best_data[a,1]==0])
        worst_tp=len([worst_data[a,0] for a in range(len(worst_data)) if worst_data[a,0]>=(detection_threshold+sigma) if worst_data[a,1]==1])
        worst_fn=len([worst_data[a,0] for a in range(len(worst_data)) if worst_data[a,0]<(detection_threshold+sigma) if worst_data[a,1]==1])
        worst_fp=len([worst_data[a,0] for a in range(len(worst_data)) if worst_data[a,0]>=(detection_threshold+sigma) if worst_data[a,1]==0])
        worst_tn=len([worst_data[a,0] for a in range(len(worst_data)) if worst_data[a,0]<(detection_threshold+sigma) if worst_data[a,1]==0])
        best_precision,best_recall = generic_tools.precision_and_recall(best_tp,best_fp,best_fn)
        worst_precision,worst_recall = generic_tools.precision_and_recall(worst_tp,worst_fp,worst_fn)
        if best_precision==0 or best_recall==0:
            best_plot_data.append([sigma,best_precision,best_recall,0])
        else:
            best_plot_data.append([sigma,best_precision,best_recall,(2*best_precision*best_recall)/(best_precision+best_recall)])
        if worst_precision==0 or best_recall==0:
            worst_plot_data.append([sigma,worst_precision,worst_recall,0])
        else:
            worst_plot_data.append([sigma,worst_precision,worst_recall,(2*worst_precision*worst_recall)/(worst_precision+worst_recall)])
    Fbest = max([x[3] for x in best_plot_data])
    Fworst = max([x[3] for x in worst_plot_data])
    sigBest = [x[0] for x in best_plot_data if x[3] == Fbest][0]
    sigWorst = [x[0] for x in worst_plot_data if x[3] == Fworst][0]
    return best_plot_data, worst_plot_data, sigBest, sigWorst

def learning_curve(best_train,worst_train,best_valid,worst_valid,detection_threshold):
    BestTrainErr=[]
    WorstTrainErr=[]
    BestValidErr=[]
    WorstValidErr=[]
    for m in range(len(best_train)):
        n=m+1
        best_plot_data, worst_plot_data, sigBest, sigWorst = find_sigma_margin(best_train[:n,:],worst_train[:n,:], detection_threshold)
        BestTrainErr.append(training_error(best_train[:n,:],sigBest,detection_threshold))
        WorstTrainErr.append(training_error(worst_train[:n,:],sigWorst,detection_threshold))
        BestValidErr.append(training_error(best_valid,sigBest,detection_threshold))
        WorstValidErr.append(training_error(worst_valid,sigWorst,detection_threshold))
        print BestTrainErr[-1], WorstTrainErr[-1], BestValidErr[-1], WorstValidErr[-1]
    return BestTrainErr, WorstTrainErr, BestValidErr, WorstValidErr


def plot_diagnostic(best_data,worst_data, path):
    # plot a diagnostic plot illustrating the precision, recall and F-score as a function of increasing sigma margin
    # identify the sigma margin which optimises the precision and recall
    
    plt.figure(figsize=(8,8))
    plt.plot([a[0] for a in worst_data],[a[1] for a in worst_data], 'r-', linewidth=5.0)
    plt.plot([a[0] for a in worst_data],[a[2] for a in worst_data], 'b-', linewidth=5.0)
    plt.plot([a[0] for a in worst_data],[a[3] for a in worst_data], 'k-', linewidth=5.0)
    plt.plot([a[0] for a in best_data],[a[1] for a in best_data], 'r--', linewidth=5.0)
    plt.plot([a[0] for a in best_data],[a[2] for a in best_data], 'b--', linewidth=5.0)
    plt.plot([a[0] for a in best_data],[a[3] for a in best_data], 'k--', linewidth=5.0)
    worst_maxF=max([a[3] for a in worst_data])
    best_maxF=max([a[3] for a in best_data])
    sigWorst=[a for a in worst_data if a[3]==worst_maxF][0]
    sigBest=[a for a in best_data if a[3]==best_maxF][0]
    plt.xlabel(r'$\sigma$ margin')
    plt.ylabel(r'%')

    plt.xscale('log')
    print 'Best sigma parameters:'
    print 'Worst RMS = '+str(sigWorst[0])+' Precision = '+str(sigWorst[1])+' Recall = '+str(sigWorst[2])
    print 'Best RMS = '+str(sigBest[0])+' Precision = '+str(sigBest[1])+' Recall = '+str(sigBest[2])
    plt.tight_layout()
    plt.savefig(path+'sigma_margin_diagnostic.png')
    plt.close()
    return sigWorst[0], sigBest[0]
