def Analysis():

    import pickle
    import glob
    import matplotlib.pyplot as pl
    import numpy as np
    import os
    
    figlist = []
    
   
    default_directory = '../software/' 
    stri = raw_input('file pathway: ')
    if len(stri)>1:
        default_directory = stri #change directory
    default_extension = '2016*.pkl'
    
    q=os.path.join(default_directory, default_extension)
    print q
    
    for name in glob.glob(q): 
        print name[12:]
        Na = str(name[12:-4])
        
    while True:
        
    
        f = raw_input ('Pickle File Name: ')
        if len(f) == 0:
             break
        
        g = os.path.join(default_directory, f)
        X = str(g)
    
        date = f[0:13]
        year = date[0:4]
        month = date[4:6]
        day = date[6:8]
        hr = date[9:11]
        sc = date[11:13]

        time= year + ' ' + month + '/' + day + ' ' + hr + ':' + sc
    
    
    
        file=open( X , 'rb')
        d=pickle.load(file)
        file.close()
    

        g = raw_input('Signal(0F or 0R): ')
        Y = str(g)
        
        xx,yy = d['delay' + Y], d['sig' + Y]
    
        pl.ion()
    
        
    
        k = raw_input('plot number: ')
        thisfig =int(k) #plot number
        
        g =  time + ' ' + g
        G = str(g)
        
       
        r = []
        if len(figlist)>0: #figlist = list of list of attributes
            for i in np.arange(len(figlist)):
                r.append(figlist[i][0]) 
                
                #the figure number of each pre-existing plot
        new = thisfig not in r
        if new:
            fignum = thisfig
            fig = pl.figure(fignum)
            
            ax2 = pl.subplot2grid((2,1), (0,0), colspan=1,)
            ax1 = pl.subplot2grid((2,1), (1,0),colspan =1)
        
            figattr = [fignum,fig,ax1,ax2] #list of attributes: plot#, figure, axes
            
            figattr[3].plot (d['delay'+Y], d['sig'+ Y])
            figattr[3].set_xlabel ("Delay0F")
            figattr[3].set_ylabel ("Sig0F")
            figattr[3].set_title (str(d['run']))
            
            #np.savetxt(Na +' ' + 'Delay'+ Y + '.txt',d['delay'+Y])
            #np.savetxt(Na + ' ' + 'Sig' + Y + '.txt',d['sig'+ Y])
        
        
    
            line, = figattr[3].plot (d['delay'+ Y], d['sig'+Y])
            line.set_label(G)
            figattr[3].legend()
    

            i = 13
            Nsize = 2**i
            dt=(1/(d['sample freq']))
            t=dt*np.arange(Nsize)
            v=(d['speed'])
            x = v*t
            dx = dt*v
            total_t = (d['scan time'])
    
            total_s = (d['samples requested'])
            startpt = ((total_s - Nsize)/2) 
            endpt = startpt + Nsize
    
            F = (d['sig'+Y])
            A = np.hanning(Nsize)*F[startpt:endpt]
            S = np.fft.rfft(A)
    
            S = S[:-1]
            u = np.abs(S)
            dNu = 1/(Nsize*dx)
            Nu = dNu*np.arange(Nsize/2)
            center = (300*(Nu[0] + Nu[-1]))/2
            top = max(np.abs(S))


            figattr[2].plot (300*Nu,np.abs(S)) 
            figattr[2].set_xlabel ("Frequency, GHz")
            figattr[2].set_ylabel ('Spectrum (arb)')
    
            m = time + ' Spectrum'
            M = str(m)
    
            s = 'sample frequency: %5.0f\r\nvelocity: %2.1f\r\ntime: %2.1f'% ((d['sample freq']), v, total_t)
        
            figattr[2].text(center,top, s, verticalalignment='top')
            
            figlist.append(figattr)#figlist: list of figattributes
            print figlist
            
            
            std_wlf = np.std(yy[(xx >= 0) & (xx <= 3.2)])
            std_symminus = np.std(yy[(xx >= -10) & (xx <= -5)])
            std_symplus = np.std(yy[(xx >= 5) & (xx <= 13)])
            std_plf = np.std(yy[(xx >= 17) & (xx <= 20)])
            std_bln = np.std([yy[-50:-35], yy[45:60]])
            
            result = 'wlf = '+ str(std_wlf)
            result = result + '\r\nsymminus = ' + str(std_symminus)
            result = result + '\r\nsymplus = '+ str(std_symplus)
            result = result + '\r\nplf = ' + str(std_plf)
            result = result + '\r\nbln = ' + str(std_bln)
            
            
            print result
            np.savetxt(Na + ' ' + 'Variance' + '.txt',result)
                    
        else:
            figmarker = r.index(thisfig)
            figattr = figlist[figmarker]
            
            
            fig = pl.figure(figattr[0])
            figattr[3].plot (d['delay'+ Y], d['sig'+Y])
            
            F = (d['sig'+Y])
            A = np.hanning(Nsize)*F[startpt:endpt]
            S = np.fft.rfft(A)
            S = S[:-1]
            u = np.abs(S)
            dNu = 1/(Nsize*dx)
            Nu = dNu*np.arange(Nsize/2)
            figattr[2].plot (300*Nu,np.abs(S)) 
            
            #np.savetxt(Na + 'Spectrum-Xaxis' + '.txt',300*Nu)
            #np.savetxt(Na + 'Spectrum-Yaxis' + '.txt',np.abs(S))
            
            std_wlf = np.std(yy[(xx >= 0) & (xx <= 3.2)])
            std_symminus = np.std(yy[(xx >= -10) & (xx <= -5)])
            std_symplus = np.std(yy[(xx >= 5) & (xx <= 13)])
            std_plf = np.std(yy[(xx >= 17) & (xx <= 20)])
            std_bln = np.std([yy[-50:-35], yy[45:60]])
            
            result = 'wlf = '+ str(std_wlf)
            result = result + '\r\nsymminus = ' + str(std_symminus)
            result = result + '\r\nsymplus = '+ str(std_symplus)
            result = result + '\r\nplf = ' + str(std_plf)
            result = result + '\r\nbln = ' + str(std_bln)
            
            
            print result
            np.savetxt(Na + 'Variance' + '.txt',result)
            
            
            
            

    

        
    
        
        
    
    
    

    