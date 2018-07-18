import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import pylab
pylab.rcParams['legend.loc'] = 'best'
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
import generic_tools
from astroML import density_estimation

#
matplotlib.rcParams.update({'font.size': 20})

def make_colours(frequencies):
    # using a matplotlib colourmap, assign a different colour to each of the unique fields in the input list
    cm = matplotlib.cm.get_cmap('jet')
    col = [cm(1.*i/len(frequencies)) for i in range(len(frequencies))]
    return col

def create_scatter_hist(data,sigcutx,sigcuty,paramx,paramy,range_x,range_y,dataset_id,frequencies):
    # create the figure with eta and V histograms and scatter plot
    
    print('plotting figure: scatter histogram plot')

    frequencies.sort()
    if "TP" in frequencies:
        # if the data is classified, we ensure that the "frequencies" are correct
        frequencies = ["TN","TP","FN","FP"]
    if "stable" in frequencies:
        freq_labels= [name.replace("_", " ") for name in frequencies]
#    elif "~" in frequencies[0]:
#        freq_labels= [name.replace("~", ",") for name in frequencies]
    else:
        freq_labels=frequencies

    # Setting up the plot
    nullfmt   = NullFormatter()         # no labels
    fontP = FontProperties()
#    fontP.set_size('large')
    col = make_colours(frequencies)
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

    # Plotting data - scatter plot
    for i in range(len(frequencies)):
        xdata_var=[data[n][1] for n in range(len(data)) if data[n][3]==frequencies[i]]
        ydata_var=[data[n][2] for n in range(len(data)) if data[n][3]==frequencies[i]]
        if frequencies[i]=='stable':
            axScatter.scatter(xdata_var, ydata_var,color='0.75', s=10., zorder=1)
        else:
            axScatter.scatter(xdata_var, ydata_var,color=col[i], s=10., zorder=5)
    if 'stable' in frequencies or 'TN' in frequencies:
        x=[data[n][1] for n in range(len(data)) if (data[n][3]=='stable' or data[n][3]=='FP' or data[n][3]=='TN')]
        y=[data[n][2] for n in range(len(data)) if (data[n][3]=='stable' or data[n][3]=='FP' or data[n][3]=='TN')]
    else:
        x=[data[n][1] for n in range(len(data))]
        y=[data[n][2] for n in range(len(data))]

    # Plotting histograms with bayesian blocks binning
    new_bins = density_estimation.bayesian_blocks(x)
    binsx = [new_bins[a] for a in range(len(new_bins)-1) if abs((new_bins[a+1]-new_bins[a])/new_bins[a])>0.05]
    binsx = binsx + [new_bins[-1]]
    new_bins = density_estimation.bayesian_blocks(y)
    binsy = [new_bins[a] for a in range(len(new_bins)-1) if abs((new_bins[a+1]-new_bins[a])/new_bins[a])>0.05]
    binsy = binsy + [new_bins[-1]]
    axHistx.hist(x, bins=binsx, normed=1, histtype='stepfilled', color='b')
    axHisty.hist(y, bins=binsy, normed=1, histtype='stepfilled', orientation='horizontal', color='b')
    axScatter.legend(freq_labels,loc=4, prop=fontP)

    # Plotting lines representing thresholds (unless no thresholds)
    if sigcutx != 0 or sigcuty != 0:
        axHistx.axvline(x=sigcutx, linewidth=2, color='k', linestyle='--')
        axHisty.axhline(y=sigcuty, linewidth=2, color='k', linestyle='--')
        axScatter.axhline(y=sigcuty, linewidth=2, color='k', linestyle='--')
        axScatter.axvline(x=sigcutx, linewidth=2, color='k', linestyle='--')

    # Plotting the Gaussian fits
    fit=norm.pdf(range_x,loc=paramx[0],scale=paramx[1])
    axHistx.plot(range_x,fit, 'k:', linewidth=2)
    fit2=norm.pdf(range_y,loc=paramy[0],scale=paramy[1])
    axHisty.plot(fit2, range_y, 'k:', linewidth=2)

    # Final plot settings
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    axHistx.axes.yaxis.set_ticklabels([])
    axHisty.axes.xaxis.set_ticklabels([])
    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )
    xmin=int(min([data[n][1] for n in range(len(data))])-1)
    xmax=int(max([data[n][1] for n in range(len(data))])+1)
    ymin=int(min([data[n][2] for n in range(len(data))])-1)
    ymax=int(max([data[n][2] for n in range(len(data))])+1)
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
    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )
    plt.savefig('ds'+str(dataset_id)+'_scatter_hist.png')

    # find all the variable candidates
    tmp=[x for x in data if x[1]>sigcutx if x[2]>sigcuty]
    tmp2=[]
    for line in tmp:
        if line[0] not in tmp2:
            tmp2.append(line[0])
    IdTrans=np.sort(tmp2, axis=0)

    plt.close()

    return IdTrans


def create_diagnostic(trans_data,sigcut_etanu,sigcut_Vnu,frequencies,dataset_id):
    print('plotting figure: diagnostic plots')
    if "TP" in frequencies:
        frequencies = ["TN","TP","FN","FP"]
    if "stable" in frequencies:
        freq_labels= [name.replace("_", " ") for name in frequencies]
    elif "~" in frequencies[0]:
        freq_labels= [name.replace("~", ",") for name in frequencies]
    else:
        freq_labels=frequencies

    # Setting up the plot
    nullfmt   = NullFormatter()         # no labels
    fig = plt.figure(1,figsize=(12,12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    fontP = FontProperties()
#    fontP.set_size('large')
    fig.subplots_adjust(hspace = .001, wspace = 0.001)
    ax1.set_ylabel(r'$\eta_\nu$', fontsize=28)
    ax3.set_ylabel(r'$V_\nu$', fontsize=28)
    ax3.set_xlabel('Max Flux (Jy)', fontsize=24)
    ax4.set_xlabel('Max Flux / Average Flux', fontsize=24)
    col = make_colours(frequencies)

    # Plotting data
    for i in range(len(frequencies)):
        xdata_ax3=[trans_data[x][3] for x in range(len(trans_data)) if trans_data[x][5]==frequencies[i]]
        xdata_ax4=[trans_data[x][4] for x in range(len(trans_data)) if trans_data[x][5]==frequencies[i]]
        ydata_ax1=[trans_data[x][1] for x in range(len(trans_data)) if trans_data[x][5]==frequencies[i]]
        ydata_ax3=[trans_data[x][2] for x in range(len(trans_data)) if trans_data[x][5]==frequencies[i]]
        if frequencies[i]=='stable':
            ax1.scatter(xdata_ax3, ydata_ax1,color='0.75', s=10., zorder=1)
            ax2.scatter(xdata_ax4, ydata_ax1,color='0.75', s=10., zorder=1)
            ax3.scatter(xdata_ax3, ydata_ax3,color='0.75', s=10., zorder=1)
            ax4.scatter(xdata_ax4, ydata_ax3,color='0.75', s=10., zorder=1)
        else:
            ax1.scatter(xdata_ax3, ydata_ax1,color=col[i], s=10., zorder=5)
            ax2.scatter(xdata_ax4, ydata_ax1,color=col[i], s=10., zorder=6)
            ax3.scatter(xdata_ax3, ydata_ax3,color=col[i], s=10., zorder=7)
            ax4.scatter(xdata_ax4, ydata_ax3,color=col[i], s=10., zorder=8)
    ax4.legend(freq_labels, loc=4, prop=fontP)

    # Plotting lines representing thresholds (unless no thresholds)
    if sigcut_etanu != 0 or sigcut_Vnu != 0:
        ax1.axhline(y=10.**sigcut_etanu, linewidth=2, color='k', linestyle='--')
        ax2.axhline(y=10.**sigcut_etanu, linewidth=2, color='k', linestyle='--')
        ax3.axhline(y=10.**sigcut_Vnu, linewidth=2, color='k', linestyle='--')
        ax4.axhline(y=10.**sigcut_Vnu, linewidth=2, color='k', linestyle='--')

    # Plotting settings
    xmin_ax3=int(np.log10(min([trans_data[x][3] for x in range(len(trans_data))])))
    xmax_ax3=int(np.log10(max([trans_data[x][3] for x in range(len(trans_data))])))
    xmin_ax4=0.8
    xmax_ax4=max([trans_data[x][4] for x in range(len(trans_data))])
    ymin_ax1=int(np.log10(min([trans_data[x][1] for x in range(len(trans_data)) if trans_data[x][1]>0.])))
    ymax_ax1=int(np.log10(max([trans_data[x][1] for x in range(len(trans_data))])))
    ymin_ax3=int(np.log10(min([trans_data[x][2] for x in range(len(trans_data)) if trans_data[x][2]>0.])))
    ymax_ax3=int(np.log10(max([trans_data[x][2] for x in range(len(trans_data))])))
    xmin_ax4=0
    xmax_ax4=int(np.log10(max([trans_data[x][4] for x in range(len(trans_data))])))    
    xvals_ax3=range(int(xmin_ax3),int(xmax_ax3+1))
    xtxts_ax3=[r'$10^{'+str(a)+'}$' for a in xvals_ax3]
    yvals_ax1=range(int(ymin_ax1),int(ymax_ax1+1))
    ytxts_ax1=[r'$10^{'+str(a)+'}$' for a in yvals_ax1]
    yvals_ax3=range(int(ymin_ax3),int(ymax_ax3+1))
    ytxts_ax3=[r'$10^{'+str(a)+'}$' for a in yvals_ax3]
    xvals_ax4=range(int(xmin_ax4),int(xmax_ax4+1))
    xtxts_ax4=[r'$10^{'+str(a)+'}$' for a in xvals_ax4]
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax1.set_ylim(10.**int(ymin_ax1-1.),10.**(int(ymax_ax1)+1.))
    ax3.set_ylim(10.**int(ymin_ax3-1.),10.**(int(ymax_ax3)+1.))
    ax3.set_xlim(10.**(int(xmin_ax3)-1.),10.**(int(xmax_ax3)+1.))
    ax4.set_xlim(0.9,10.**(int(xmax_ax4)+0.5))
    ax3.set_xticks([10.**x for x in xvals_ax3])
    ax1.set_yticks([10.**y for y in yvals_ax1])
    ax3.set_yticks([10.**y for y in yvals_ax3])
    ax4.set_xticks([10.**x for x in xvals_ax4])
    ax1.set_xlim( ax3.get_xlim() )
    ax4.set_ylim( ax3.get_ylim() )
    ax2.set_xlim( ax4.get_xlim() )
    ax2.set_ylim( ax1.get_ylim() )
    ax3.set_xticklabels(xtxts_ax3, fontsize=20)
    ax1.set_yticklabels(ytxts_ax1, fontsize=20)
    ax3.set_yticklabels(ytxts_ax3, fontsize=20)
    ax4.set_xticklabels(xtxts_ax4, fontsize=20)
    ax1.xaxis.set_major_formatter(nullfmt)
    ax4.yaxis.set_major_formatter(nullfmt)
    ax2.xaxis.set_major_formatter(nullfmt)
    ax2.yaxis.set_major_formatter(nullfmt)
    plt.savefig('ds'+str(dataset_id)+'_diagnostic_plots.png')
    plt.close()

    return

def plotLC(num, error_train, error_val, fname, xlog, ylog, xlabel):
    # Plot the learning curves
    plt.figure(1,figsize=(8,8))
    error_train=[a if a!=0 else 1e-6 for a in error_train]
    error_val=[a if a!=0 else 1e-6 for a in error_val]
    plt.plot(num, error_train, 'b-', linewidth=2.0)
    plt.plot(num, error_val, 'g-', linewidth=2.0)
    if ylog:
        plt.yscale('log')
    if xlog:
        plt.xscale('log')
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel('Error', fontsize=22)
    plt.axis([min(num), max(num)*1.2, 1e-4,1e-0])
    #plt.legend(['training', 'validation'], loc=4, fontsize=28)
    plt.savefig(fname+'_curve.png')
    plt.close()
    return
