# Zhaodi Pan  Oct 31
import pylab as pl
import sptpol_software.util.files as files
import cPickle as pkl
from glob import glob
from pyfts_package import *
import numpy as np
from netCDF4 import Dataset
import pickle

# For chopped scans, you might have to shift the frequencies by 60 Hz. I'm not sure why this is yet.

def fts(netcdf='W98_114px_20161023_0442',boloname='W98_114.150.x',band=150,speed=0.5,chop=False,optics_correction=False,save=True,return_data=True,plot=True, num=1):

  data_dir='./'
  rate = 152.587890625   #sampling rate in Hz
  #speed = optical velocity in cm/s

  bandpass = [band-30,band+30]   #approximate band of the detector, used to find interferrorgrams in code
  
  bolo_data=glob(data_dir+netcdf+'.nc')[0]
  bolo_data=Dataset(bolo_data,mode='r')
  #print bolo_data.variables.keys()
  data_i=pl.array(bolo_data.variables[boloname+'_I'])
  data_q=pl.array(bolo_data.variables[boloname+'_Q'])
    
  cts = np.sqrt(abs(abs(np.array(data_i))**2+abs(np.array(data_q))**2))
  pl.plot(cts)
  if chop:
    cts=-1*cts
  #pl.figure()
  #pl.clf()
  #pl.plot(range(len(cts)),cts)   #plot it
  #cts = cts[1.5e4:*]  # if you want to cut some data points off the beginning


  out = analyze_fts_scan(cts, speed, band=bandpass, rate=rate, chop=chop,hann=True, pband = [10, 400],absv=False,plot=plot,length=7.5)  #process the FTS data# if it's high enough signal-to-noise drop the /  abs at the end
  #print cts

  avg = add_fts_scan(out)  #average the fts scans
  if avg!=0: 
   figure=pl.figure()
   pl.clf()
   pl.plot(avg['freq'], avg['real']/max(avg['real'])) # plot the average
   pl.xlim(10, 300)
   pl.xlabel('Frequency (GHz)')
   pl.ylabel('Normalized Response')
   pl.plot(avg['freq'], avg['im']/max(avg['real']), 'b--') #overplot the imagninary component to get an estimate of the noise
   pl.grid(b=True)
   pl.title('Averaged Spectra for '+boloname)
   pl.savefig(data_dir+'bolos/'+boloname+'%d'%num+'.png')
   figure2=pl.figure()
   pl.clf()
   pl.plot(cts )
   pl.title(boloname)
   pl.savefig(data_dir+'bolos/timestream/'+boloname+'%d'%num+'.png')
   if optics_correction==True:
    optics=pkl.load(open('/home/cryo/Data/spectra/fts_analysis/Optics_chain.pkl','r'))
    ind1=pl.where((optics[0]>60) & (optics[0]<240))
    ind2=pl.where((avg['freq']>60) & (avg['freq']<240))
    ind3=pl.where((avg['freq']>60) & (avg['freq']<300))
    
    pl.figure(4)
    pl.plot(optics[0][ind1],(avg['real'][ind2]/optics[1][ind1])/max(avg['real'][ind2]/optics[1][ind1]))
    pl.plot(optics[0][ind1],(avg['im'][ind2]/optics[1][ind1])/max(avg['real'][ind2]/optics[1][ind1]),'b--')
    pl.grid(b=True)
    pl.xlabel('Frequency (GHz)')
    pl.ylabel('Normalized Response')
    pl.title('Optics-Corrected Avg Spectra for '+sq+'Ch'+str(chan))

   if return_data:
    if save:
#      pkl.dump([avg['freq'][ind3],avg['real'][ind3]/max(avg['real'][ind3])],open(data_dir+parser_dir+'/'+parser_dir.rpartition('/')[-1]+'_'+sq+ch+'_raw_spectrum.pkl','wb'))
#      pkl.dump([optics[0][ind1],(avg['real'][ind2]/optics[1][ind1])/max(avg['real'][ind2]/optics[1][ind1])],open(data_dir+parser_dir+'/'+parser_dir.rpartition('/')[-1]+'_'+sq+ch+'_corrected_spectrum.pkl','wb'))
#      file2=open(data_dir+parser_dir+'/'+parser_dir.rpartition('/')[-1]+'_'+sq+ch+'spectrum.txt','w')
#      for i in range(len(avg['freq'])):
#             file2.write("%s  %s\n" % (avg['freq'][i], avg['real'][i]/max(avg['real'])))
#      file2.close()
#      pkl.dump(avg,open(data_dir+parser_dir+'/'+parser_dir.rpartition('/')[-1]+'_'+sq+ch+'_spectrum.pkl','wb'))
      pkl.dump(avg,open(data_dir+'bolos/'+boloname+'%d'%num+'.pkl','wb'))
    if optics_correction:  
      return [optics[0][ind1],(avg['real'][ind2]/optics[1][ind1])/max(avg['real'][ind2]/optics[1][ind1])]
    else:
      return avg #[(avg['freq'], avg['real']/max(avg['real']))]

def netcdf_plot(netcdf_files={'W94_254_20161020_2340.nc':'94.197',  'W94_231px_20161021_0017.nc':'94.197', 'W94_218px_20161021_0035.nc':'94.197','W95_249px_20161022_0510.nc':'95.197', 'W95_250px_20161022_0534.nc':'95.197','W95_245px_20161022_1727.nc':'95.197','W95_219px_20161022_1835.nc':'95.197'}, distance_range=10):

    #netcdf_files={'W98_px3_20161020_2047.nc':'98.3','W98_px13_20161020_2205.nc':'98.13','W98_px36_20161020_2237.nc':'98.36', 'W94_254_20161020_2340.nc':'94.254',  'W94_231px_20161021_0017.nc':'94.231', 'W94_218px_20161021_0035.nc':'94.218','W99_178px_20161021_0126.nc':'99.178', 'W99_179px_20161021_0149.nc': '99.179', 'W99_161px_20161021_0219.nc':'99.161' }
    # 20161020/20161020_232102_drop_bolos/data/IceBoard_0117.Mezz_2.ReadoutModule_1_OUTPUT.pkl
    #
    #'W95_249px_20161022_0510.nc':'95.249', 'W95_250px_20161022_0534.nc':'95.250'
    ########20161022_200208_drop_bolos
    #
    #'W98_px36_unsaturated_20161020_2247.nc':'98.36','W94_254px_unsaturated_20161020_2347.nc':'94.254', 'W99_179px_unsaturated_20161021_0149.nc':'99.179', 'W99_161px_unsaturated_20161021_0219.nc':'99.161'
    # 20161020/20161020_232102_drop_bolos/data/IceBoard_0117.Mezz_2.ReadoutModule_1_OUTPUT.pkl
    #
    #'W95_245px_20161022_1727.nc':'95.245','W95_219px_20161022_1835.nc':'95.219', 'W98_114px_20161022_2021.nc':'98.114'
    #20161022_200208_drop_bolos
    #
    #'W98_17px_20161023_0527.nc': '98.17' #,'W98_114px_20161023_0442.nc':'98.114'
    #fts/20161023/RELATIVE_OPTICAL_EFFICIENCY_TESTS/PLATE_ON_DROP_BOLOS/20161023_085308_drop_bolos
    '''
    Arguments: netcdf_files, distance_range.
    netcdf_files should be a dictionary in the format of {'location of the netcdf file': centered pixel number(int)}
    distance_range is the distance to the centered pixel you want to get spectrum on.
    '''
    pixel_position=pickle.load(open('pixel_positions.pkl','rb'))
    #print pixel_position[5]
    for file_name in netcdf_files.keys():
       center_pixel= netcdf_files[file_name].split('.')[1]
       waf=netcdf_files[file_name].split('.')[0]
       #print center_pixel
       #print center_pixel
       center_position= pixel_position[int(center_pixel)]
       #print center_pixel
       neighbor_pixel=[]
       all_pixel=[]
       centered_pixel=[]
       second_pixel=[]
       far_pixel=[]
       for pixel in pixel_position.keys():
          coord= pixel_position[pixel]
          dist= ((coord[0]-center_position[0])**2+(coord[1]-center_position[1])**2)**(0.5)
          if dist<10 and dist>=1:
              neighbor_pixel.append(pixel)
          if dist<10:
              all_pixel.append(pixel)
          if dist<1:
              centered_pixel.append(pixel)
          if dist>10 and dist<12:
              second_pixel.append(pixel)
          if dist>20 and dist<25:
              far_pixel.append(pixel)
       #print centered_pixel
       bololist=['90.x','90.y','150.x', '150.y','220.x', '220.y']
       band=[90,90,150,150,220,220]
       k=1
       for i in range(len(centered_pixel)):
            for j in range(len(bololist)):
                   bolo_data=glob(file_name)[0]
                   bolo_data=Dataset(bolo_data,mode='r')
                   if 'W'+str(waf)+'_'+str(centered_pixel[i])+'.'+bololist[j]+'_I' in bolo_data.variables.keys():
                    k=k+1
                    fts_result= fts(netcdf=file_name.split('.')[0],boloname='W'+str(waf)+'_'+str(centered_pixel[i])+'.'+bololist[j],band=band[j],speed=0.5,chop=False,optics_correction=False,save=True,return_data=True,plot=False, num=k)
    return 0

#netcdf_plot()   
   
       

