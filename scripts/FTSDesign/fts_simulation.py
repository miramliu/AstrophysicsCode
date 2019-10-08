#This code generates the interferogram from a given input band.
#To get the band back, do the FTS analysis on the generated interferogram.
#Zhaodi Pan  05/04/2017 panz@uchicago.edu

import numpy as np

import pylab as pl

import math

def rad(deg):

 return deg/180.0*np.pi
#this is a top hat function for the band we want to generate interferogram from
def flatband(x,leftbound, rightbound):

 return (np.sign(x-leftbound)+1)/2*(np.sign(rightbound-x)+1)/2

#this is the polarizer function

def polarizer(theta, R, T):

 m11=-R*np.cos(theta)**2-(1-T)*np.sin(theta)**2

 m12=-R*np.sin(theta)*np.cos(theta)+(1-T)*np.sin(theta)*np.cos(theta)

 m13=T*np.sin(theta)**2+(1-R)*np.cos(theta)**2

 m14=-T*np.cos(theta)*np.sin(theta)+(1-R)*np.sin(theta)*np.cos(theta)

 m21=-R*np.cos(theta)*np.sin(theta)+(1-T)*np.sin(theta)*np.cos(theta)

 m22=-R*np.sin(theta)**2-(1-T)*np.cos(theta)**2

 m23=-T*np.sin(theta)*np.cos(theta)+(1-R)*np.cos(theta)*np.sin(theta)

 m24=T*np.cos(theta)**2+(1-R)*np.sin(theta)**2

 m31=T*np.sin(theta)**2+(1-R)*np.cos(theta)**2

 m32=-T*np.cos(theta)*np.sin(theta)+(1-R)*np.sin(theta)*np.cos(theta)

 m33=-R*np.cos(theta)**2-(1-T)*np.sin(theta)**2

 m34=-R*np.sin(theta)*np.cos(theta)+(1-T)*np.cos(theta)*np.sin(theta)

 m41=-T*np.sin(theta)*np.cos(theta)+(1-R)*np.cos(theta)*np.sin(theta)

 m42=T*np.cos(theta)**2+(1-R)*np.sin(theta)**2

 m43=-R*np.cos(theta)*np.sin(theta)+(1-T)*np.sin(theta)*np.cos(theta)

 m44=-R*np.sin(theta)**2-(1-T)*np.cos(theta)**2

 return np.matrix(((m11,m12,m13,m14),(m21,m22,m23,m24),(m31,m32,m33,m34),(m41,m42,m43,m44)))
#this is the center moving mirror
def delay(delta):

 imag=np.complex(0,1)

 return np.matrix(((np.exp(imag*delta/2),0,0,0),(0,np.exp(imag*delta/2),0,0),(0,0,np.exp(-imag*delta/2),0),(0,0,0,np.exp(-imag*delta/2))))

#reflection mirrors. Can define the mirror to be blocked in the top half or the bottom half

def mirror(blockup=0, blockdown=0):

 if blockup:

  return -1*np.matrix(((0,0,0,0),(0,0,0,0),(0,0,1,0),(0,0,0,1)))

 if blockdown:

  return -1*np.matrix(((1,0,0,0),(0,1,0,0),(0,0,0,0),(0,0,0,0)))

 return -1*np.matrix(((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)))
#This is the function for generating the output at the symmetric and anti-symmetric output ports
def optical_response(distance,freq,pol_angle=[45,90,0,45],R=1,T=1):

#distance in unit of mm

 delta=distance/1000/(2.9979e8/freq)*2*np.pi

 e_field=np.sqrt(flatband(freq,90e9,100e9))

 e_vector=[0,0,e_field,e_field]

 transfer_matrix= polarizer(rad(pol_angle[0]),R,T)*mirror()*polarizer(rad(pol_angle[1]),R,T)*mirror()*delay(delta)*mirror()*polarizer(rad(pol_angle[2]),R,T)*mirror()*polarizer(rad(pol_angle[3]),R,T)

 out_vector=[0,0,0,0]

 for i in range(4):

  for j in range(4):

   out_vector[i]=out_vector[i]+e_vector[j]*transfer_matrix.item(i,j)

 anti_sym_out=abs(out_vector[2])**2+abs(out_vector[3])**2

 sym_out=abs(out_vector[0])**2+abs(out_vector[1])**2

 return anti_sym_out, sym_out

#nominal inteferogram

freq=np.linspace(70e9,100e9,10000)

df=freq[1]-freq[0]

dist=np.linspace(-150,150,2000)

interferogram=[]

for i in range(len(dist)):

 amplitude=np.sum(optical_response(dist[i],freq)[1])

 interferogram.append(amplitude)

#change the angle of the first polarizer to be 48deg

#plot intefograms with different transmission/reflection coefficients

 
'''
interferogram2=[]

for i in range(len(dist)):

 amplitude=np.sum(optical_response(dist[i],freq, R=.9, T=1)[1])

 interferogram2.append(amplitude)

 interferogram3=[]

for i in range(len(dist)):

 amplitude=np.sum(optical_response(dist[i],freq, R=1, T=.9)[1])

 interferogram3.append(amplitude)

 

interferogram4=[]

for i in range(len(dist)):

 amplitude=np.sum(optical_response(dist[i],freq, R=.9, T=.9)[1])

 interferogram4.append(amplitude)

 

pl.figure()

pl.plot(dist,interferogram,color='b',label='Perfect')

pl.plot(dist,interferogram2,label='Reflection=0.9, Transmission=1',color='g')

pl.plot(dist,interferogram3,label='Reflection=1, Transmission=0.9',color='r')

pl.plot(dist,interferogram4,label='Reflection=0.9, Transmission=0.9',color='k')

 

pl.xlabel('Optical delay(mm)')

pl.ylabel('Amplitude(arbitrary unit)')

pl.title('Non-perfect reflection and transmission')

pl.legend(loc='best')

pl.show()

 

'''

#plot interferograms with different alignment angles
interferogram2=[]

for i in range(len(dist)):

 amplitude=np.sum(optical_response(dist[i],freq,[45,90,0,60])[0])

 interferogram2.append(amplitude)

interferogram3=[]

for i in range(len(dist)):

 amplitude=np.sum(optical_response(dist[i],freq,[45,90,20,45])[0])

 interferogram3.append(amplitude)

interferogram4=[]

for i in range(len(dist)):

 amplitude=np.sum(optical_response(dist[i],freq,[45,115,0,45])[0])

 interferogram4.append(amplitude)

interferogram5=[]

for i in range(len(dist)):

 amplitude=np.sum(optical_response(dist[i],freq,[80,90,0,45])[0])

 interferogram5.append(amplitude)

interferogram6=[]

for i in range(len(dist)):

 amplitude=np.sum(optical_response(dist[i],freq,[50,95,-5,40])[0])

 interferogram6.append(amplitude)

pl.figure()

pl.plot(dist,interferogram,color='b',label='Perfect')

pl.plot(dist,interferogram2,label='First polarizer 15 deg off',color='g')

pl.plot(dist,interferogram3,label='Second polarizer 20 deg off',color='r')

pl.plot(dist,interferogram4,label='Third polarizer 25 deg off',color='k')

pl.plot(dist,interferogram5,label='Fourth polarizer 30 deg off',color='y')

pl.plot(dist,interferogram6,label='All 5 deg off',color='m')

pl.xlabel('Optical delay(mm)')

pl.ylabel('Amplitude(arbitrary unit)')

pl.title('Non-perfect polarization angle, anti-symmetric output')

pl.legend(loc='best')

pl.show()

