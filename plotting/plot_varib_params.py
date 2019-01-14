import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from astroML import density_estimation
import numpy as np


def make_cmap(frequencies):
    cm = matplotlib.cm.get_cmap('hsv')
    col = [cm(1.*i/len(frequencies)) for i in range(len(frequencies))]
    return col

def gaussian_fit(data,param):
    range_data=np.linspace(min(data),max(data),1000)
    fit=norm.pdf(range_data,loc=param[0],scale=param[1])
    return range_data,fit

def make_bins(x):
    new_bins = density_estimation.bayesian_blocks(x)
    binsx = [new_bins[a] for a in range(len(new_bins)-1) if abs((new_bins[a+1]-new_bins[a])/new_bins[a])>0.05]
    binsx = binsx + [new_bins[-1]]
    return binsx
