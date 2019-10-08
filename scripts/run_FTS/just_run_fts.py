import serial
import time
import sys
import zaber_commands as zc
from fts_scan_zc import fts_scan_zc

print('!! Port for zaber (FTS mirror motor) is hardcoded. Adjust it for your computer first !! \n')

### open the port where zaber is plugged in, some examples below
#dev = '/dev/ttyACM1'
dev = '/dev/tty.usbmodem1421'
#dev = "COM3"

print('Opening port connection .... \n')
ser,err = zc.init_zaber(dev)

print('Parameters to run FTS---\n')

yn = raw_input("Custom Parameters [y/n] ? Defualts used otherwise. ")

if yn == 'n':
    # default values:
    niter = 1
    max_d = 75. #mm
    speed = 10. #mm/s
    accel = 10000. # mm/s^2
else:
    niter_str = raw_input("Number of scans ? ")
    niter = int(niter_str)
    max_d_str = raw_input("Distance to traverse [mm] ? ")
    max_d = float(max_d_str)
    speed_str = raw_input("Speed [mm/s] ? ")
    speed = float(speed_str)
    accel_str = raw_input("Acceleration [mm/s^2] ? ")
    accel = float(accel_str)

print("\n Moving FTS mirror with these parameters now .... \n")
print("N scans: ",niter, "Distance: ", max_d, "Speed: ", speed, "Acceleration: ", accel)
#fts_scan_zc(niter,max_d,speed,ser)
time.sleep(2)
print('\n Now check bolometer time stream / raw dump for interferograms!')
