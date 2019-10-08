import numpy as np
from matplotlib import pylab as plt
import math
from scipy import interpolate as intp
from scipy import integrate as integrate
from matplotlib import pylab as plt
from math import pi
from scipy.special import digamma
from scipy.optimize import curve_fit
import pickle
import random


#FUNCTION dielect1, nu, t,a,b,n
#dielect returns the transmission of a sheet of
# dielectric material with parameters a,b,n,
# 
# nu is the frequency in icm
# t is the thickness in cm
# a is the coefficient for the absorption
# b is the exponent for the powerlaw in frequency
# n is the index of refraction
# The routine is an implementation of the equations in
# Halpern and Gush 1985.
#
# Version Who            date           comments
#  1      S. Meyer       23-Jun-1999    Original version
#  2                     24-Jun-1999    fixed math error exception
#  3      Z. Pan         16-Dec-2014    Adapted to python 
#  4      Z. Pan         08-Aug-2017    Added fit function 
# implement equations 6 and 7
#

data_file='../Data/fluorogold_band.pkl'
data_file_ref='../Data/fluorogold_ref_band.pkl'

Thickness=0.35306
data=pickle.load(open(data_file,'rb'))
freq=data['freq']
spectrum=data['real']
data_ref=pickle.load(open(data_file_ref,'rb'))
freq_ref=data_ref['freq']
spectrum_ref=data_ref['real']

spectrum_divided= spectrum/spectrum_ref
#choose only the useful frequency components
spectrum_divided=spectrum_divided[np.where(freq_ref<500)]
freq_ref=freq_ref[np.where(freq_ref<500)]
spectrum_divided=spectrum_divided[np.where(freq_ref>100)]
freq_ref=freq_ref[np.where(freq_ref>100)]

#Transmission model
def dielec(nu,a,b,n,A,t):

 alpha=a*nu**b             
 k=2*np.pi*n*nu
 gamma=alpha/2.0+k*1j
 eta=1.0/(n*np.sqrt(np.linspace(1,1,len(alpha/k))-1j*alpha/k))

 # calc the complex admittance
 y=(1.0/eta)*(eta*np.cosh(gamma*t)+np.sinh(gamma*t))/(np.cosh(gamma*t)+eta*np.sinh(gamma*t))
 # calc the amplitude reflectance
 rho=(1-y)/(1+y)
 # power reflectance
 r=(abs(rho))**2
 # return transmission
 return A*np.exp(-alpha*t)*(1-r)    



# We do not want to fit for thickness. We take a measurement.
def dielec_fit(nu,a,b,n,A):
  return dielec(nu, a, b, n, A,Thickness)



#do the fit
fit=curve_fit(dielec_fit, freq_ref/30.0, spectrum_divided, bounds=([2.5e-5,0.5,1.5, .9],[5e-1,6,1.7, 1.1]))

a, b, n, A= fit[0]

a_err, b_err, n_err, A_err= np.sqrt(fit[1][0][0]), np.sqrt(fit[1][1][1]),np.sqrt(fit[1][2][2]),np.sqrt(fit[1][3][3])

print 'a and error in a are', a, a_err
print 'b and error in b are', b, b_err
print 'n and error in n are', n, n_err
print 'A and error in A are', A, A_err
plt.figure()
plt.plot(freq_ref,spectrum_divided, color='b', label='Original_data')
plt.plot(freq_ref,dielec_fit(freq_ref/30.0, a, b, n, A), 'g', label='Fit_result, n=%.2f, a=%.4f, b=%.2f'%(n,a,b))
plt.xlabel('Frequency(GHz)')
plt.ylabel('Transmission')
plt.title('0.353 cm thick Fluorogold')
plt.legend()
plt.show()




















