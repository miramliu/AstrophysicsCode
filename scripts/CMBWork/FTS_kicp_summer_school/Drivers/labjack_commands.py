"""
Module labjack_commands

contains the commands to take data for the FNAL_FTS.

"""
import numpy as np
import traceback
from datetime import datetime
import u6

SAMPLESPERPACKET = 25

def init_labjack(): #initialize the U6 and return the handle
    d = u6.U6()
    d.getCalibrationData()
    return d

def setup_scan(d,speed,max_nu,oversample,gain): # setup the scan frequency. Return the actual frequency
    """
        d is device handle
        speed is optical speed
        nu_max is biggest frequency desired (GHz)
        oversample is the number of samples you want per actual sample needed
        gain is the analog input gain 0=>1 1=>10 2=>100 3=>1000
        returns sample freqency
    """
    dx = 300. / (2. * max_nu) # the sample spacing from the max frequency (300 to convert to mm^-1) 
    dt = dx/(speed * oversample) # the sample frequency including an oversample factor
    SCAN_I = int(4.e6 *dt) # the clock frequency divider
    Options1 = int(128 + 16 * int(gain)) # set the options for the signal channel
    d.streamConfig(NumChannels = 3, ChannelNumbers = [0, 2, 3],
        SamplesPerPacket = SAMPLESPERPACKET,
        ChannelOptions = [Options1, 0, 0], SettlingFactor = 0,
        ResolutionIndex = 0, InternalStreamClockFrequency = 0,
        DivideClockBy256 = False, ScanInterval = SCAN_I)
    return 4.e6 / float(SCAN_I) # return the actual sample frequency

def start_adc_one(d,n,dx,data,key):
    """
        d is device handle
        n is the number of samples needed
        dx is the distance between samples
        data is the data dictionary
        key is the keyword specifier
    """
    d.streamStart()
    missed = 0
    try:
        for r in d.streamData():
            if r is not None:
                #if dataCount >= packetRequest:
                    #break
                if r['errors'] != 0:
                    print ("Error: %s ; " % r['errors'])
                if r['numPackets'] != d.packetsPerRequest:
                    print ("----- UNDERFLOW : %s : " % r['numPackets'])
                if r['missed'] !=0:
                    missed += r['missed']
                try:
                    wlf
                except NameError:
                    sig = r['AIN0']
                    scan = r['AIN2']
                    wlf = r['AIN3']
                else:
                    sig = np.append(sig, r['AIN0'])
                    scan = np.append(scan, r['AIN2'])
                    wlf = np.append(wlf, r['AIN3'])
                    if sig.size >= n:
                        break
            else:
                print ("No data - stream sample rate slower than USB timeout")
    except:
        print "".join(i for i in traceback.format_exec())
    finally:
        d.streamStop()
        if missed != 0: print 'missed ',missed,' samples!!'
        #wlf=np.append(wlf,wlf[-1])
        la=np.array([sig.size,wlf.size,scan.size])
        l=la.min()
        sig=sig[:l-1]
        wlf=wlf[:l-1]
        scan=scan[:l-1]
        iidex=np.where(scan > 2.5)
        #print 'before cut ',sig.size
        #sig=sig[iidex]
        #print 'after cut' ,sig.size
        #wlf=wlf[iidex]
        #scan=scan[iidex]
        delay = np.arange(len(wlf))*dx
        dwlf=wlf-np.roll(wlf,1)
        dwlf[0] = dwlf[1]
        dwlf[-1]=dwlf[-2]
        dwlf = abs(dwlf)
        pos = delay[np.where(dwlf == np.max(dwlf))]
        delay = delay - pos[0]
        data['sig'+key] = sig
        data['scan'+key] = scan
        data['wlf'+key] = wlf
        data['delay'+key] = delay
    return

def close_labjack(d):
    return d.close()
