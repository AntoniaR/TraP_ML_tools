import scipy as sp
import numpy as np
import os

from scipy.stats import norm

    
def SigmaFit(data):
    median = np.median(data)
    std_median = np.sqrt(np.mean([(i-median)**2. for i in data]))
    tmp_data = [a for a in data if a < 3.*std_median+median and a > median - 3.*std_median]
    param1 = norm.fit(tmp_data)
    param2 = norm.fit(data)
    return param1, param2

def extract_data(filename):
    # extract data in a csv file into a list
    info=[]
    data=open(filename,'r')
    for lines in data:
        if not lines.startswith("#"):
            lines=lines.rstrip().replace(" ", "")
            info.append(lines.split(','))
    data.close()
    return info

def write_data(filename,tmp):
    output = open(filename,'w')
    for line in tmp:
        output.write(str(line[0])+','+str(line[1])+','+str(line[2])+'\n')
    output.close()
    return

def read_src_lc(sources):
#
# Reads all the extracted source data and sorts them into unique sources with their lightcurves
#
    runcat=[] # A list comprising all the unique runcat source ids.
    new_source={} # A dictionary containing all the lightcurves, etc, for each unique source
    frequencies=[] # The frequencies in the dataset
    for a in range(len(sources)):
        new_runcat=sources[a][8]
        new_freq=int((float(sources[a][7])/1e6)+0.5) # observing frequency in MHz
        # check if it's a new runcat source and then either create a new entry or append
        if new_runcat not in runcat:
            runcat.append(new_runcat)
            new_source[new_runcat]=[sources[a]]
        else:
            new_source[new_runcat]=new_source[new_runcat]+[sources[a]]
        # If the observing frequency is new, append to the list
        if new_freq not in frequencies: 
            frequencies.append(new_freq)

    # return the list of observing frequencies and the full dictionary of source information
    return frequencies, new_source

def collate_trans_data(new_source,frequencies,transients):
#
# Using the data stored in the new_source dictionary and the transients list, store the transient and variability
# parameters for each unique source in the dataset.
#    
    trans_data=[]
    bands={}
    # sort the information in the transients list into a dictionary
    # transient id:[transient type, flux/max_rms, flux/min_rms, detection_thresh]
    transRuncat={x[5]:[x[2],float(x[1])/float(x[3]), float(x[1])/float(x[4]), x[0]] for x in transients}
    for freq in frequencies:
        for keys in new_source.keys():
            flux=[]
            flux_err=[]
            date=[]
            band=[]
            tmp=0.
            # Extract the different parameters for the source
            for b in range(len(new_source[keys])):
                if int((float(new_source[keys][b][7])/1e6)+0.5)==freq:
                    band.append(new_source[keys][b][0])
                    flux.append(float(new_source[keys][b][5]))
                    flux_err.append(float(new_source[keys][b][6]))
                    if tmp<int(new_source[keys][b][14]):
                        eta=float(new_source[keys][b][2])
                        V=float(new_source[keys][b][11])
                        N=float(new_source[keys][b][4])
                        tmp=int(new_source[keys][b][14])
                    ra=new_source[keys][b][-2]
                    dec=new_source[keys][b][-3]
            # if the source has been observed in the given observing frequency, extract the variability parameters
            # from the final observation at that frequency.
            if len(flux)!=0:
                bands[freq]=band
                ### Calculate the ratios...
                avg_flux_ratio = [x/(sum(flux)/len(flux)) for x in flux]
                ### Collate and store the transient parameters (these are across all the pipeline runs for the final figures)
                if keys in transRuncat.keys():
                # identify if this source is in the new source transient list and extract parameters
                    transType=transRuncat[keys][0]
                    min_sig=transRuncat[keys][1]
                    max_sig=transRuncat[keys][2]
                    detect_thresh=transRuncat[keys][3]
                else:
                # if not in the transient list, then insert standard parameters for non-transients
                    transType=2
                    min_sig=0
                    max_sig=0
                    detect_thresh=0
                # write out the key parameters for each source at each observing frequency
                trans_data.append([keys, eta, V, max(flux), max(avg_flux_ratio), freq, len(flux), ra, dec, transType, min_sig, max_sig, detect_thresh])
    print 'Number of transients in sample: '+str(len(trans_data))
    # Return the array of key parameters for each source
    return trans_data
