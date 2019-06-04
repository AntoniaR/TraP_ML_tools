##### Code to output data directly from a TraP database ready for use with the machine learning tools

import scipy as sp
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import *
from sqlalchemy.orm import relationship
import tkp.db
import logging
logging.basicConfig(level=logging.INFO)
query_loglevel = logging.WARNING  # Set to INFO to see queries, otherwise WARNING
import sys
sys.path.append('../')
from dblogin import * # This file contains all the variables required to connect to the database
from database_tools import dbtools
from tools import tools
from plotting import plot_varib_params as pltvp
import matplotlib
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['legend.loc'] = 'best'
from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties
import seaborn as sns
import random



# The input database, dataset and thresholds
dataset_id = 9
database = 'AR_R4'
websiteURL = 'http://banana.transientskp.org/r4/vlo_'+database+'/runningcatalog/'

path='ml_csv_files/'
stableData = path+'stable_sources.csv'
 
# Connect to the database and run the queries
session = dbtools.access(engine,host,port,user,password,database)

transients = dbtools.GetTransDataForML(session,dataset_id)
transrcs = [[transients[i].Newsource.trigger_xtrsrc.image.detection_thresh,
                 transients[i].Newsource.trigger_xtrsrc.f_int,
                 transients[i].Newsource.newsource_type,
                 transients[i].Newsource.trigger_xtrsrc.image.rms_max,
                 transients[i].Newsource.trigger_xtrsrc.image.rms_min,
                 transients[i].Runningcatalog.id,
                 ] for i in range(len(transients))]


Runcats = dbtools.GetRuncatDataForML(session,dataset_id)
runcats = [[Runcats[i].xtrsrc.image.band.freq_central,
                Runcats[i].dataset.id,
                Runcats[i].varmetric.eta_int,
                Runcats[i].xtrsrc.extract_type,
                Runcats[i].datapoints,
                Runcats[i].xtrsrc.f_int,
                Runcats[i].xtrsrc.f_int_err,
                Runcats[i].xtrsrc.image.freq_eff,
                Runcats[i].id,
                Runcats[i].xtrsrc.image.tau_time,
                Runcats[i].xtrsrc.image.taustart_ts,
                Runcats[i].varmetric.v_int,
                Runcats[i].wm_decl,
                Runcats[i].wm_ra,
                Runcats[i].xtrsrc.id
             ] for i in range(len(Runcats))]


frequencies, srcs = tools.read_src_lc(runcats)

print runcats[-1]
print frequencies

trans_data = tools.collate_trans_data(srcs,frequencies,transrcs)


print trans_data[-1]

output3 = open('ds'+str(dataset_id)+'_trans_data.txt','w')
output3.write('#Runcat, eta, V, flux, fluxrat, freq, dpts, RA, Dec, ttype, maxRmsSigma, minRmsSigma, detectionThreshold  \n')
for x in range(len(trans_data)):
    string='%s' % ','.join(str(val) for val in trans_data[x])
    output3.write(string+'\n')
output3.close()
print 'Data extracted and saved'

