import scipy as sp
import numpy as np

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

