import pylab as pl
import pickle
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
data1=pickle.load(open('../../data/raw_data/20180411_1325__continuous_scan_wo_filter_large_aperture_chopped.pkl', 'rb'))
data2=pickle.load(open('../../data/raw_data/20180319_0332_300GHz_new_polarizer_34.pkl', 'rb'))

pl.figure(figsize=(12,12))
pl.subplot(2, 1, 1)
signal1= data1['sig0F']
signal1=signal1-np.mean( signal1)
signal1= signal1/max(abs(signal1)) 



maxima= argrelextrema(signal1, np.greater)
minima= argrelextrema(signal1, np.less)
index1= np.where(abs(data1['delay0F'][maxima]+3.10)>2.0)
index2=np.where(abs(data1['delay0F'][minima]+3.10)>2.0)
spline1= UnivariateSpline(data1['delay0F'][maxima][index1]+3.10, signal1[maxima][index1], s=.05)
spline2= UnivariateSpline(data1['delay0F'][minima][index2]+3.10, signal1[minima][index2], s=.05)
average= (spline1(data1['delay0F']+3.10)+spline2(data1['delay0F']+3.10))/2
pl.plot(data1['delay0F']+3.10, signal1-average, color='b', linewidth=.06)
pl.plot([], linewidth=.7, label='Broadband source')
pl.plot(data1['delay0F']+3.10, spline1(data1['delay0F']+3.10)-average, 'r', linewidth=2, label='Profile')
pl.plot(data1['delay0F']+3.10, spline2(data1['delay0F']+3.10)-average, 'r', linewidth=2)
pl.ylabel('Signal', fontsize=23)
pl.title('Chopped interferogram and oscillator interferogram', fontsize=23)
pl.legend( loc=1, fontsize=20)
pl.xlim([-65,65])
pl.ylim([-.9, 1.5])
#pl.ylabel('Signal')
pl.tick_params(axis='both', which='major', labelsize=23)


pl.subplot(2, 1, 2)
signal2= data2['sig0F']
signal2=signal2-np.mean( signal2)
signal2= signal2/max(abs(signal2)) 
#pl.plot(data2['delay0F']+3.10, signal2, color='b',label= 'Oscillator')

#pl.title('Inteferograms')
maxima= argrelextrema(signal2, np.greater)
minima= argrelextrema(signal2, np.less)
#index1= np.where(abs(data1['delay0F'][maxima]+3.10)>2.0)
#index2=np.where(abs(data1['delay0F'][minima]+3.10)>2.0)
spline1= UnivariateSpline(data2['delay0F'][maxima]+3.10, signal2[maxima], s=.05)
spline2= UnivariateSpline(data2['delay0F'][minima]+3.10, signal2[minima], s=.05)
average= (spline1(data2['delay0F']+3.10)+spline2(data2['delay0F']+3.10))/2

pl.plot(data2['delay0F']+6.10, signal2-average, color='b',label= 'Gunn oscillator', linewidth=.7)
pl.xlabel('Optical delay(mm)', fontsize=23)
pl.plot(data2['delay0F']+6.10, spline1(data2['delay0F']+3.10)-average, 'r', linewidth=2, label='Profile')
pl.plot(data2['delay0F']+6.10, spline2(data2['delay0F']+3.10)-average, 'r', linewidth=2)


#pl.legend( loc=1, fontsize=20)
pl.xlim([-65,65])
pl.ylim([-1.1,1.4])
pl.ylabel('Signal', fontsize=23)
pl.legend( loc=1, fontsize=20)
pl.tick_params(axis='both', which='major', labelsize=23)
pl.show()





