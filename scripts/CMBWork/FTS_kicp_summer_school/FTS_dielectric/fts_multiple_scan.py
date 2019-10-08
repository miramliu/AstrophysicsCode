'''
 Zhaodi Pan  Aug8 2017
 FTS analysis code for multiple scans
'''
import pylab as pl
import cPickle as pkl
from glob import glob
from pyfts_package import *
import numpy as np
import pickle

raw_data='../Data/fluoro_gold.pkl'
output_band='../Data/fluorogold_band.pkl'
best_guess_bands=[50,300] # best guess bands

# load up the output data
def load_file(filename):
   data=pickle.load(open(filename,'rb'))
   speed= data['speed']
   freq= data['sample freq']
   scan_data=[]
   scan_data_keys=[]
   for key in data.keys():
     if len(key)>3 and key[:3]=='sig':
        scan_data_keys.append(key)
   for scan_data_key in sorted(scan_data_keys):
      scan_data=np.concatenate((scan_data, data[scan_data_key]))
   return {'scan_data':scan_data, 'speed':speed/10.0, 'freq':freq}

# analyze the data
# You'll need to enter your best-guess bands, and the name you want to save the data as in the beginning of the code

def fts(filename, guessed_band=[50,300], filename_to_save='file_name_to_save', save=True):
  data= load_file(filename)
  timestream, speed, freq= data['scan_data'], data['speed'], data['freq']
  pl.plot(np.arange(0,len(timestream),1), timestream)
  pl.title('Timestream ')
  pl.xlabel('Sample number')
  pl.ylabel('Raw ADC unit') 
  out = analyze_fts_scan(timestream, speed, band=guessed_band, rate=freq, chop=False,hann=True, pband = [0, 650],absv=False,plot=True,length=6.0)  #process the FTS data# if it's high enough signal-to-noise
  avg = add_fts_scan(out)  #average the fts scans
  if avg!=0: 
   figure=pl.figure()
   pl.clf()
   pl.plot(avg['freq'], avg['real']/max(avg['real'])) # plot the average
   pl.xlim(10, 650)
   pl.xlabel('Frequency (GHz)')
   pl.ylabel('Normalized Response')
   pl.plot(avg['freq'], avg['im']/max(avg['real']), 'b--') #overplot the imagninary component to get an estimate of the noise
   pl.grid(b=True)
   pl.title('Averaged Spectra')
   if save:
      pkl.dump(avg,open(filename_to_save,'wb'))
   return avg  
fts(filename=raw_data, guessed_band=best_guess_bands, filename_to_save=output_band)
pl.show()
