#Basic analysis (not including demodulation or deconvolution)
#Given a pickle file of an interferogram this plots the interferogram (using a power of 2 samples)
#it then plots the spectrum (the fourier transform of the interferogram). Mira

import numpy as np
import matplotlib
import matplotlib.pyplot as pl
import pickle
import sys
import numpy as np



filenames = {'With_filter':'20171013_1315_test.pkl', 'Without_filter':'20171013_1318_test_no_lpf.pkl' } #1mms


pl.figure()
for key in sorted(filenames.keys()):
 filename=filenames[key]
 with open('../../data/raw_data/' +str(filename) , 'rb') as f:
    d = pickle.load(f) #Add 'encoding = latin1' if using python 3.x  
 def analyze_spectrum(d): 
        #print d['speed']
        print d['sample freq']
	Nsize = int((120/d['speed']*d['sample freq'])/2)*2
        #print Nsize
	dt=(1/(d['sample freq'])) #period
	T1=dt*(Nsize) #full period
	v=(d['speed'])
        #print v
	X = v*T1 #full distance
	dx = dt*v #smallest amount of distance travelled
	total_t = (d['scan time']) #how long it ran
    
	total_s = (d['samples requested']) #number of samples 
	#startpt = ((total_s - Nsize)/2) #starting point


	#endpt = startpt + Nsize #ending point
	#startpt = int(startpt)
	#endpt = int(endpt)

	df = 1/T1
	f = df*np.arange(Nsize/2)+df/2.0
	fFull = df*np.arange((Nsize/2) + 1)+df/2.0


	y = (d['sig0F'])
	D = y#[startpt:endpt] #signal
	D = np.flipud(D)

	a = d['delay0F']/v
	t = a#[startpt:endpt] #time (used as x axis)

	#D = np.hanning(Nsize)*D #signal multiplied by a hanning function to improve FFT
	S = np.fft.rfft(D) #fourier transform
	S = S[0:-1]
	u = np.abs(S) #gets rid of imaginary part of fourier transform solely for the plot
	dNu = 1/(Nsize*dx)
	Nu = dNu*np.arange(Nsize/2)
        return t, D, Nu, u
 dt=(1/(d['sample freq']))
 Nsize = int((120/d['speed']*d['sample freq'])/2)*2
 t=np.arange(Nsize)*dt
 t, D, Nu, u= analyze_spectrum(d=d)
 #f, (ax1,ax2) = pl.subplots(2,1)
 x = t*d['speed']
 y = d['wlf0F']
 #startpt = ((len(y) - Nsize)/2) #starting point
 #endpt = startpt + Nsize 
 y=y#[startpt:endpt]
 #print np.where(y>2.5)
 x_trans = min(x[np.where(y>2.5)])
 #print 'xtrans is', x_trans
 #print x_trans
 x=x+x_trans
 pl.plot(d['delay0F'], D, label=key) #cut off
pl.title('Interferogram')
pl.legend(loc='best')
 #ax1.set_xlabel('Time(s) starting at WLF')



pl.figure()
for key in sorted(filenames.keys()):
 filename=filenames[key]
 with open('../../data/raw_data/' +str(filename) , 'rb') as f:
    d = pickle.load(f) #Add 'encoding = latin1' if using python 3.x  
 def analyze_spectrum(d): 
	Nsize = len(d['sig0F'])
	dt=(1/(d['sample freq'])) #period
	T1=dt*(Nsize) #full period
	v=(d['speed'])
        #print v
	X = v*T1 #full distance
	dx = dt*v #smallest amount of distance travelled
	total_t = (d['scan time']) #how long it ran
    
	total_s = (d['samples requested']) #number of samples 
	#startpt = ((total_s - Nsize)/2) #starting point


	#endpt = startpt + Nsize #ending point
	#startpt = int(startpt)
	#endpt = int(endpt)

	df = 1/T1
	f = df*np.arange(Nsize/2)+df/2.0
	fFull = df*np.arange((Nsize/2) + 1)+df/2.0


	y = (d['sig0F'])
	D = y#[startpt:endpt] #signal
	D = np.flipud(D)

	a = d['delay0F']/v
	t = a#[startpt:endpt] #time (used as x axis)

	D = np.hanning(Nsize)*D #signal multiplied by a hanning function to improve FFT
	S = np.fft.rfft(D) #fourier transform
	S = S[0:-1]
	u = np.abs(S) #gets rid of imaginary part of fourier transform solely for the plot
	dNu = 1/(Nsize*dx)
	Nu = dNu*np.arange(Nsize/2)
        return t, D, Nu, u

 t, D, Nu, u= analyze_spectrum(d=d)
 #f, (ax1,ax2) = pl.subplots(2,1)
 pl.plot(300*Nu, u/max(u),'-o', label=key)
pl.title('Fourier Transform of data')
#pl.xlim(0,2000)
pl.xlabel('GHz')
pl.ylabel('Arb')
pl.xlim([0,1000])
pl.legend()
pl.show()
