"""
Module fts_scan

This routine runs an FTS scan and writes the data to a pkl file

"""
import numpy as np
from datetime import datetime
import time
import pickle as pk
import zaber_commands as zc
import labjack_commands as lj

def fts_scan(name,niter,max_d,speed,max_nu,oversample,gain,ser,lj_hand):
	"""
		name		run name and will be the name of the pkl file
		iter		number of scans to run
		max_d		maximum delay. Scan length is 2 x this length (units of optical delay, mm)
		speed		scan speed (units of optical delay mm/s)
		max_nu		maximum desired frequency (GHz)
		oversample	adc sample rate multiplier over what is needed for max_nu
		gain		ADC gain 0 => +10V to -10V, 1 is x10, 2 is x100, 3 is x1000
		ser		serial port handle for zaber stage
		lj_hand		handle for connection to ADC
		
		Returns error code and dict containing data
	"""
	#
	# write the input parameters to the data dict.
	#
	accel=2000. # acceleration to use in mm/s^2 optic
	data = {'run':name,
	        'max_d':max_d,
	        'speed':speed,
	        'iterations':niter,
	        'max_nu':max_nu,
	        'oversample':oversample,
	        'ADC gain': gain,
	        'acceleration': accel}
	#
	# clear the zaber position triggers
	#
	zc.clear_trigger_do_pos(ser,1)	#this is the white light fringe trigger
	zc.clear_trigger_do_pos(ser,3)	#this is the start end "scan" trigger
	zc.clear_trigger_do_pos(ser,5)	#this is the stop end "scan" trigger
	#
	# calculate scan parameters
	#
	d_accel=speed**2/(2.*accel)	#the distance you go in acceleration phase
	startscan = zc.white_light_fringe - max_d	#the start of the constant velocity part of scan
	stopscan = zc.white_light_fringe + max_d	#the end of the constant velocity part of scan
	startpos = startscan - d_accel	#the starting position of the carriage before a forward scan
	stoppos = stopscan + d_accel	#the ending position of the carriage after a forward scan
	#
	# calculate ADC parameters and setup the labjack adc
	#
	freq = lj.setup_scan(lj_hand,speed,max_nu,oversample,gain)	#labjac setup returns actual frequency
	scan_time = 2 * (max_d/speed + speed/accel)	#scan time including acceleration and deceleration
	n = int(scan_time * freq)	#number of ADC samples requested
	dx = speed/freq				#distance between samples
	#
	# unpark the stage, clear the scan and wlf signals move the stage home and then to start position
	#
	zc.unpark(ser)
	zc.set_do(ser,1,0)			#clear digital output 1
	zc.set_do(ser,2,0)			#clear digital output 2
	zc.set_accel(ser,15000.)	#set fast acceleration for move home
	zc.set_max_speed(ser,zc.max_speed_o)	#set speed to max
	zc.move_abs_wait(ser,0.2)	#go 0.2mm of the home position fast
	zc.home(ser)				#home the stage
	zc.move_abs_wait(ser,startpos)	#move to the start position
	#
	# setup triggers for digital outputs
	#
	zc.set_trigger_do_pos(ser,zc.white_light_fringe,2,1,1)# set digital position trigger at wlf for channel 2
	zc.set_trigger_do_pos(ser,startscan,1,3,1)	#set digital position trigger at constant velocity scan start
	zc.set_trigger_do_pos(ser,stopscan,1,5,0)	#set digital postiion trigger at constant velocity scan end
	#
	# set parameters for scan and write into dict
	#
	zc.set_accel(ser,accel)			#set acceleration
	zc.set_lim_max(ser,stoppos)		#set end position for forward constant velocity scan
	zc.set_lim_min(ser,startpos)	#set end position for reverse constatn velocity scan
	zc.set_do(ser,1,0)			#clear digital output 1
	zc.set_do(ser,2,0)			#clear digital output 2
	data['sample freq'] = freq
	data['dx'] = dx
	data['scan time'] = 2 * max_d/speed
	data['acc time'] = 2 * speed/accel
	data['samples requested'] = n
	data['scan start struct_time'] = time.localtime()
	now=datetime.now()
	#
	# iterate over scans - move forwared and then back
	#
	for iiter in np.arange(niter):
		key = str(iiter)+'F'		#make keyword be iteration number and 'F'
		zc.move_vel(ser,speed)	#move foward at command speed
		lj.start_adc_one(lj_hand,n,dx,data,key)	#start ADC data taking
		time.sleep(.5)		#give it an extra 500 ms to finish moving
		key = str(iiter)+'R'		#make keyword be iteration number and 'F'
		zc.move_vel(ser,-speed)		#move reverse at command speed
		lj.start_adc_one(lj_hand,n,-dx,data,key)	#start ADC data taking
		time.sleep(0.5)		#give it an extra 500 ms to finish moving
	#
	# clear digital outputs move to the wlf, park and return dict
	lj.close_labjack(lj_hand)
	zc.set_do(ser,1,0)			#clear digital output 1 (white light fringe signal)
	zc.set_do(ser,2,0)			#clear digital output 2 (scan signal)
	zc.clear_trigger_do_pos(ser,1)	#clear white light fringe position trigger
	zc.clear_trigger_do_pos(ser,3)	#clear lower limit of scan region position trigger
	zc.clear_trigger_do_pos(ser,5)	#clear upper limit of scan region position trigger
	zc.set_accel(ser,10000.)		#set acceleration to high value for return to WLF position
	zc.set_max_speed(ser,zc.max_speed_o)	#set max speed for return to WLF position
	zc.move_abs_wait(ser,zc.white_light_fringe)# move to wlf
	zc.park(ser)				#park
	#
	# write the data to file
	#
	
	file=open('../../data/raw_data/'+now.strftime("%Y%m%d_%H%M")+'_'+name+'.pkl','wb')
	pk.dump(data,file)
	file.close()
	return 0,data