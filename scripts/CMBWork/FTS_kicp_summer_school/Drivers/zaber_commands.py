"""
Module zaber_commands

This contains the commands needed to move the FNAL_FTS mirror

"""

import numpy as np
import serial
import time

#####
# Constants having to do with the FTS setup and stage parameters
#####

opt_per_physical = 4.0 * np.cos(14.*np.pi/180.)# conversion from physical to optical delay
mm_per_step = 1.905e-4# stepper motor mm/step
acc_conversion = 1.6384/1.e4
vel_conversion = 1.6384
white_light_fringe = 99.55+15.265-3.335+16.156-23.76+8.89# Position of white light fringe in optical mm near
#white_light_fringe = 120.# Position of white light fringe in optical mm far
max_speed = 104.# maximum speed (mm/s physical) given by manufacturer
max_speed_o = max_speed * opt_per_physical# conversion to optical units

def init_zaber(dev):#initialze the serial port, return port handle
    # set zabor device ID (controller) and peripheral ID (stage)
    ser=serial.Serial(
        port = dev,
        baudrate = 115200,
        parity = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        bytesize = serial.EIGHTBITS)
    if ser.isOpen():
        print ('Serial port ', dev, ' was opened')
        output_command_wait(ser,'/1 set peripheralid 43022\r\n')	#set the ID for the LSM050B-T4 stage
        output_command_wait(ser,'/1 set deviceid 30211\r\n')		#set the ID for the X-MCB1 controller
        err = 0
    else:
        print ('Serial port ', dev, ' did not open')
        err = 10
    return ser,err

#####
# Functions to run whole sets of scans
#
# Two different ways to run a set of scans. Each scan runs from -max_delay
# to +max_delay on either side of the white light fringe. A full cycle is run
# niter times. Digital output 1 is true during the scan motion, Digital output
# 2 is true when the stage is on the + side of the white light fringe.
#####

def run_scans(port,max_d,speed,niter):# run a set of scans with specifiec maxdelay
                                    # and speed
    unpark(port)
    set_do(port,1,0)
    set_do(port,2,0)
    set_trigger_do_pos(port,white_light_fringe,2,1,1)
    set_max_speed(port,max_speed_o)
    move_abs_wait(port,0.2)
    s = home(port)
    startpos = white_light_fringe-max_d
    stoppos = white_light_fringe+max_d
    set_max_speed(port,max_speed_o)
    move_abs_wait(port,startpos)
    set_max_speed(port,speed)
    for iiter in np.arange(niter):
        set_do(port,1,1)
        move_abs_wait(port,stoppos)
        set_do(port,1,0)
        set_do(port,1,1)
        move_abs_wait(port,startpos)
        set_do(port,1,0)
    set_max_speed(port,max_speed_o)
    move_abs_wait(port,white_light_fringe)
    clear_trigger_do_pos(port,1)
    park(port)

def run_scans2(port,max_d,speed,niter):# run a set of scans with limits and move vel
    unpark(port)
    set_do(port,1,0)
    set_do(port,2,0)
    set_trigger_do_pos(port,white_light_fringe,2,1,1)
    err,limit_home_pos=get_home_pos(port)
    err,limit_away_pos=get_away_pos(port)
    set_max_speed(port,max_speed_o)
    move_abs_wait(port,0.2)
    s = home(port)
    startpos = white_light_fringe-max_d
    stoppos = white_light_fringe+max_d
    set_max_speed(port,max_speed_o)
    move_abs_wait(port,startpos)
    set_lim_max(port,stoppos)
    set_lim_min(port,startpos)
    for iiter in np.arange(niter):
        set_do(port,1,1)
        move_vel_wait(port,speed)
        set_do(port,1,0)
        set_do(port,1,1)
        move_vel_wait(port,-speed)
        set_do(port,1,0)
    set_max_speed(port,max_speed_o)
    move_abs_wait(port,white_light_fringe)
    set_lim_max(port,limit_away_pos)
    set_lim_min(port,limit_home_pos)
    clear_trigger_do_pos(port,1)
    park(port)

#####
# Commands to move the stage
####

def home(port):# Move the stage to the home position
    return output_command_wait(port,'/1 home \r\n')
    
def move_abs_wait(port,pos):# Move to absolute position (in mm optical)
    ipos = int(pos/(mm_per_step * opt_per_physical))
    return output_command_wait(port,'/1 move abs ' + str(ipos) + '\r\n')

def move_abs(port,pos):# Move to absolute position (in mm optical)
    ipos = int(pos/(mm_per_step * opt_per_physical))
    return output_command(port,'/1 move abs ' + str(ipos) + '\r\n')

def move_vel_wait(port,speed):# set speed for scan
    ispeed = int((speed * vel_conversion)/(mm_per_step * opt_per_physical))
    return output_command_wait(port,'/1 move vel ' + str(ispeed) + '\r\n')

def move_vel(port,speed):# set speed for scan
    ispeed = int((speed * vel_conversion)/(mm_per_step * opt_per_physical))
    return output_command(port,'/1 move vel ' + str(ispeed) + '\r\n')

def set_accel(port,accel):# set the acceleration mm/s^2
    iaccel=int(accel * acc_conversion / (mm_per_step * opt_per_physical))
    return output_command_wait(port,'/1 set accel ' + str(iaccel) + '\r\n')

#####
# Commands to set stage parameters
#####

def set_max_speed(port,speed):# Set max speed (in mm/s optical)
    ispeed = int(speed/(mm_per_step * opt_per_physical))
    return output_command_wait(port,'/1 set maxspeed ' + str(ispeed) + '\r\n')

def set_lim_max(port,limit):# set the maximum limit for travel
    ilimit = int(limit/(mm_per_step * opt_per_physical))
    return output_command(port,'/1 set limit.max ' + str(ilimit) + '\r\n')

def set_lim_min(port,limit):# set the maximum limit for travel
    ilimit = int(limit/(mm_per_step * opt_per_physical))
    return output_command(port,'/1 set limit.min ' + str(ilimit) + '\r\n')


#####
# Commands to set the digital output signals
#####

def set_do(port,chan,state):# set the digital output chan to state
    if chan == 1 or chan == 2:
        if int(state) == 0 or int(state) == 1:
            return output_command(port,'/1 io set do ' + 
                    str(int(chan)) + ' ' + str(int(not state))+'\r\n')
        else:
            print ('Set_do: state must be a logical')
            return 'state must be logical',0
    else:
        print ('Set_do: Channel must be 1 or 2')
        return 'channel must be 1 or 2',0

def set_trigger_do_pos(port,pos,chan,trig,up):# set trigger trig and trig+1 to make digital output chan true 
    #                                    when position is greater than pos. means it goes up on + travel
    ipos = int(pos/(mm_per_step * opt_per_physical))
    err,response = output_command(port,'/1 trigger ' + str(trig) + ' when 1 pos >= ' 
            + str(ipos) + '\r\n')
    err,response = output_command(port,'/1 trigger ' + str(trig+1) + ' when 1 pos < ' 
            + str(ipos) + '\r\n')
    err,response = output_command(port,'/1 trigger ' + str(trig) + ' action a io do '
             +str(chan) + ' ' + str(int(not up)) + '\r\n')
    err,response = output_command(port,'/1 trigger ' + str(trig+1) + ' action a io do '
             +str(chan) + ' ' + str(int(up)) + '\r\n')
    err,response = output_command(port,'/1 trigger ' + str(trig) + ' enable\r\n')
    err,response = output_command(port,'/1 trigger ' + str(trig+1) + ' enable\r\n')
    return err,response

def clear_trigger_do_pos(port,trig):# clear trigger trig and trig+1
    err,response = output_command(port,'/1 trigger ' + str(trig) + ' disable\r\n')
    err,response = output_command(port,'/1 trigger ' + str(trig+1) + ' disable\r\n')
    err,response = output_command(port,'/1 trigger 3 disable\r\n')
    err,response = output_command(port,'/1 trigger 4 disable\r\n')
    err,response = output_command(port,'/1 trigger 5 disable\r\n')
    err,response = output_command(port,'/1 trigger 6 disable\r\n')
    return err,response

#####
# Commands to get information from the stage
#####

def get_away_pos(port):# Get the position of the away limit
    err,response = output_command(port,'/1 get limit.away.pos\r\n')
    pos = mm_per_step * opt_per_physical * int(response[17:23])
    return err,pos

def get_home_pos(port):# Get the position of the away limit
    err,response = output_command(port,'/1 get limit.home.pos\r\n')
    pos = mm_per_step * opt_per_physical * int(response[17:23])
    return err,pos

#####
# Commands to park and unpark the stage
#####

def park(port):# set device into park state
    return output_command_wait(port,'/1 tools parking park\r\n')

def unpark(port):# set device into park state
    return output_command_wait(port,'/1 tools parking unpark\r\n')

#####
# Low level commands that talk to the stage, wait and parse responses
#####

def output_command(port,command):# Put a command out the port
    if port.isOpen():
        err = port.write(command)
        return err,port.readline()
    else:
        return 'port not open',0

def output_command_wait(port,command):#Put a command out the port and wait for IDLE
    if port.isOpen():
        port.write(command)
    else:
        return 'port not open',0
    time.sleep(0.1)
    response = wait_for_idle(port)
    return 0,response

def wait_for_idle(port):# keep poking and wait until IDLE return
    response = port.readline()
    err = 0
    (r,d,a,f,s,w) = parse_reply(response)
    while (s == 'BUSY'):
        err = port.write('/1 tools echo 1 \r\n')
        time.sleep(0.1)
        response = port.readline()
        (r,d,a,f,status,w)  =parse_reply(response)
        s = status
    return err,response

def parse_reply(reply):# divide the reply from stage into sections
    rtype = reply[0:1]
    dev = reply[1:3]
    axis = reply[4:5]
    flag = reply[6:8]
    status = reply[9:13]
    warn = reply[14:18]
    return (rtype,dev,axis,flag,status,warn)
