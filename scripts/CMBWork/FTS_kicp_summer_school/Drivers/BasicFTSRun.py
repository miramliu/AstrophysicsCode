#Basic code to run the FTS. Give name, number of scans, and speed. Will run, save the output as a pickle file, and plot the interferogram. Mira

import sys
import time
import pickle
import serial
import matplotlib.pyplot as pl
import zaber_commands as zc
import labjack_commands as lj
from fts_scan import fts_scan
import numpy as np

d = lj.init_labjack()

Answ = raw_input('Connect [y]/[n] : ')
if str(Answ) == 'y':
    d
else: 
    print ('end')

#choose based on which ports are used on the computer running this code.
#DONT DELETE THIS
#dev = ('/dev/tty.usbmodem1421')#Mac port number
dev = '/dev/tty.usbmodem1411'
#dev = '/dev/ttyACM2' # Linux port number
#dev = "COM4" # Window port number
ser,err = zc.init_zaber(dev)

run_name = raw_input('Save Name: ') #saved with this name as a pickle file
niter = int(raw_input('Niter: '))
max_d = 60. #mm
#max_d = raw_input('Max Distance: ')
speed = int(raw_input('Speed: '))  #mm/s
max_nu = 300. # GHz
accel = 10000. # mm/s^2
oversample = 16
gain = 1
#max_nu = raw_input('Max Frequency (GHz): ') #300. # GHz
#accel = raw_input('Acceleration (mm/s^2): ') #10000. # mm/s^2
#oversample = raw_input('Oversample: ') #16
#gain = raw_input('Gain: ') 
            
            
Answ2 = raw_input('Run [y]/[n] : ')
if str(Answ2) == 'y':
    err,data=fts_scan(run_name,
                  niter,
                  max_d,
                  speed,
                  max_nu,
                  oversample,
                  gain,
                  ser,
                  d)
    pl.plot(data['delay0F'],data['sig0F'])
    pl.title(str(run_name))
    pl.ylabel('Signal')
    pl.xlabel('Delay (mm)')
    pl.show()
else: 
    print('end')
