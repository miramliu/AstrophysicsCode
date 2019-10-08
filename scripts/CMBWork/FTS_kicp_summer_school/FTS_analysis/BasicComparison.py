#given two files, plots them in separate windows and then gives the option of comparing the two through division. Mira

#THIS HAS BEEN EDITTED SINCE TO MAKE PLOTS FOR CMB POSTER (see github for original)

import matplotlib.pyplot as pl
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import pickle
import numpy as np
import sys

file1N = raw_input('First File Name: ') #gets file 1

with open( '../../../../data/raw_data/' + str(file1N) , 'rb') as file1:
    d1=pickle.load(file1)
    
file2N = raw_input('Second File Name: ') #gets file 2
with open( '../../../../data/raw_data/' + str(file2N) , 'rb') as file2:
    d2=pickle.load(file2)



def analyze_spectrum(d): 
	i = 11
	Nsize = 2**i
	dt=(1/(d['sample freq'])) #period
	T1=dt*(Nsize) #full period
	v=(d['speed'])
	X = v*T1 #full distance
	dx = dt*v #smallest amount of distance travelled
	total_t = (d['scan time']) #how long it ran
    
	total_s = (d['samples requested']) #number of samples 
	startpt = ((total_s - Nsize)/2) #starting point


	endpt = startpt + Nsize #ending point
	startpt = int(startpt)
	endpt = int(endpt)

	df = 1/T1
	f = df*np.arange(Nsize/2)+df/2.0
	fFull = df*np.arange((Nsize/2) + 1)+df/2.0


	y = (d['sig0F'])
	D = y[startpt:endpt] #signal
	#D = np.flipud(D)

	a = d['delay0F']# just delay now, not time /v
	t = a[startpt:endpt] #time (used as x axis)

	D = np.hanning(Nsize)*D #signal multiplied by a hanning function to improve FFT
	S = np.fft.rfft(D) #fourier transform
	S = S[0:-1]
	u = np.abs(S) #gets rid of imaginary part of fourier transform solely for the plot
	dNu = 1/(Nsize*dx)
	Nu = dNu*np.arange(Nsize/2)
        return t, D, Nu, u


t1, D1, Nu1, u1= analyze_spectrum(d1)
t2, D2, Nu2, u2= analyze_spectrum(d2)

# FIRST FILE
#f, (ax1,ax2) = pl.subplots(1,2)
#ax1.plot(t1,D1, label = str(file1N)[:-4], color = 'blue')
#ax1.set_title('Interferogram')
#ax1.set_xlabel('Delay (mm)')
#ax1.set_ylabel('Sig (mV)')
#ax2.plot(300*Nu1,u1, label = str(file1N)[:-4], color = 'blue')
#ax2.set_title('Spectrum')
#ax2.set_xlabel('GHz')
#ax2.set_ylabel('Arb')
#ax1.legend()
#ax2.legend()
f, ax1 = pl.subplots(1,1)
ax1.plot(t1,D1*80, color = 'blueviolet', linewidth = .95, label = 'Through ' + str(file1N)[:-4] + ' filter')
ax1.set_xlabel('Delay (mm)')
ax1.set_ylabel('Signal (Arb)')
ax1.set_title('Interferogram')
ax1.legend()
pl.tight_layout()

#SECOND FILE
f,(ax3,ax4) = pl.subplots(1,2)
ax3.plot(t2,D2, label = str(file2N)[:-4] + 'erence\nInterferogram', color = 'darkcyan')
ax3.set_title('Interferogram')
ax3.set_xlabel('Delay (mm)')
ax3.set_ylabel('Sig (mV)')
ax4.plot(300*Nu2,u2, label = str(file2N)[:-4] + 'erence\nSpectrum', color = 'darkcyan')
ax4.set_title('Spectrum')
ax4.set_xlabel('Frequency (GHz)')
ax4.set_xlim(10,800)
ax4.set_ylabel('Power (Arb)')
ax3.legend()
ax4.legend()
pl.tight_layout()


#oper = raw_input('Divide [y]/[n] : ')

#ensures the number of samples used is the same to be able to divide corresponding values
Len1 = len(Nu1)
Len2 = len(Nu2)

Nsize = min(Len1,Len2)
startpt1 = ((Len1 - Nsize)/2)
endpt1 = startpt1 + Nsize

startpt2 = ((Len2-Nsize)/2)
endpt2 = startpt2 + Nsize
# divide compare the two spectra

#if str(oper) == 'y': 
    #u3 = np.divide(u1[startpt1:endpt1],u2[startpt2:endpt2])
    #pl.figure(3)
    #pl.plot(300*Nu1[startpt1:endpt1], u3)
    #pl.tight_layout()
    #pl.title('Divided spectra' )
f,ax5 = pl.subplots()
ax5.plot(300*Nu1[startpt1:endpt1],u1[startpt1:endpt1], label = 'Spectrum through ' + str(file1N)[:-4] + ' filter', color = 'blueviolet')
ax5.plot(300*Nu2[startpt2:endpt2],u2[startpt2:endpt2], label = str(file2N)[:-4] + 'erence Spectrum', color = 'darkcyan')
#ax5.set_title('Individual Spectra')
ax5.set_xlabel('Frequency (GHz)')
ax5.set_xlim(10,800)
ax5.set_ylim(0,2.0)
ax5.axvline(x = 270,linestyle = '--', color = 'deeppink', label = '270 Ghz')
ax5.annotate('Water Absorption', xy = (570,.09), xytext = (585, .2), arrowprops=dict(facecolor='black', shrink=0.05, headwidth = 3, width = 5),fontsize = 15)
ax5.set_ylabel('Power (Arb)')
u3 = np.divide(u1[startpt1:endpt1],u2[startpt2:endpt2])
#ax6.set_title('Divided Spectra')
left, bottom, width, height = [.65, .38, 0.2, 0.2]
ax6 = f.add_axes([left, bottom, width, height])
#matplotlib.rc('xtick', labelsize=10) 
ax6.tick_params(axis='both', which='major', labelsize=10)
ax6.plot(300*Nu1[startpt1:endpt1], u3, color = 'black')
ax6.set_xlabel('Frequency (GHz)', fontsize = 15)
ax6.set_title('Ratio of Spectra', fontsize = 15 )
ax6.set_xlim(150,400)
ax6.set_ylim(0,1)
#ax6.set_ylabel('Arb')
ax5.legend()
#ax6.legend()


pl.show()
