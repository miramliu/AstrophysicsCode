import serial, time
import numpy as np
ser = serial.Serial()
ser.port = "/dev/ttyUSB0"
ser.baudrate = 9600
ser.bytesize = serial.EIGHTBITS #number of bits per bytes
ser.parity = serial.PARITY_NONE #set parity check: no parity
ser.stopbits = serial.STOPBITS_ONE #number of stop bits
ser.timeout = 2           #non-block read
ser.writeTimeout = 2   #timeout for write
ser.xonxoff = False
ser.rtscts= False
ser.dsrdtr = False

# constants for use
# home on which switch (1 for positive switch [default], zero for negative switch)
global home_sw,advpstp
home_sw = 1
advpstp = 0.00025 # advance in inches per step [physical to motor units]

def usrdist2stp(mm):
    # convert the user distance to steps
    # User eneters this in mm
    # function outputs string to write to velmex with correct direction
    step2take = np.abs(mm/(advpstp*25.4)) # 25.4 is mm/inch
    if home_sw:
        vf = int(-1*np.sign(mm)*np.floor(step2take))
         # -1 since we are moving "back" from positive home switch.
         # So if the user wants to move forward, they enter a positive distance, which we make negative and vice versa
    else:
        vf = int(1*np.sign(mm)*np.floor(step2take))
    return(str(vf))

ser.open()
print "Serial connection Ok --- \n"
ser.flushInput() #flush input buffer, discarding all its contents
ser.flushOutput()#flush output buffer, aborting current output

def home_motors():
    print('--- Homing motors to positive limit switch first ---\n')
    if home_sw == 0:
        # home to negative limit switch only when requested
        ser.write('I1M-0, I2M-0, IA1M-0, IA2M-0, R, C,')
    else:
        # home to positive limit switch by default
        ser.write('I1M0, I2M0, IA1M-0, IA2M-0, R, C,')
    ser.write('N')
    time.sleep(1)
    print('-------------- @home!\n')

# Configure motor 2
#ser.write('res ')
time.sleep(0.2)
ser.write('C, F,')
ser.write('SA2M4125, ')
ser.write('A2M25, ')
# motor 2 (PK268) in this design has low load <10 lbs
# keeping conservative settings identical to motor 1

# Configure motor 1
#ser.write('res ')
time.sleep(0.2)
ser.write('C, F,')
ser.write('SA1M4125, ')
ser.write('A1M25, ')
# motor 1 (PK296): with 61 lbs (vertical) motor starts to stall at speed of 5500 (max is 6000)
# Setting max speed to floor(5700*0.75) = 4000 (conservative since real load is <61 lbs)
# with speed at 4125, max accleration is 25 (it works at 30, but sounds stressed with 61 lbs vertical)

#home motor 1 and 2
time.sleep(0.2)
h = home_motors()

print('---> To quit anytime enter q \n---> Or proceed with motor controls \n')

while 2>1:
    ser.write('F,')
    time.sleep(0.2)

    abs_or_rel = raw_input("Absolute or relative motion, or home (enter a / r / h)?")

    if abs_or_rel == 'h':
        h = home_motors()
        continue
    elif abs_or_rel =='q':
        break

    r1 = raw_input("Distance for motor#1 [mm] ? (if homing, entry is not used)")
    if str(r1)== 'q':
        break
    r2 = raw_input("Distance for motor#2 [mm] ? ")
    if str(r2)=='q':
        break

    r1m= usrdist2stp(float(r1))
    r2m= usrdist2stp(float(r2))
    ser.write('C,')# this clear is very important
    print('Moving motors .... \n')

    if abs_or_rel == 'h':
        h = home_motors()
    elif abs_or_rel == 'a':
        print('..... absolute motion in progress ...\n')
        ser.write('IA1M'+r1m+', IA2M'+r2m+',R,C <cr>')
    else:
        print('..... relative motion in progress ...\n')
        ser.write('I1M'+r1m+', I2M'+r2m+',R,C <cr>')

ser.close()
