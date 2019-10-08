def Analysis():

    import pickle
    import glob
    import matplotlib.pyplot as pl
    import numpy as np
    import os
    
    default_directory = '../software/' 
    stri = raw_input('file pathway: ')
    if len(stri)>1:
        default_directory = stri
    default_extension = '*.pkl'
    
    q=os.path.join(default_directory, default_extension)
    print q
    
    for name in glob.glob(q): 
        print name
    
    f = raw_input ('Pickle File Name: ')
    g = os.path.join(default_directory, f)
    X = str(g)
    
    date = f[0:13]
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    hr = date[9:11]
    sc = date[11:13]

    time= year + ' ' + month + '/' + day + ' ' + hr + ':' + sc
    
    print X
    
    file=open( X , 'rb')
    d=pickle.load(file)
    file.close()
    

    g = raw_input('Signal(0F or 0R): ')
    Y = str(g)
    
    pl.ion()
    
    k = raw_input('plot number: ')
    K = str(k)
    g =  time + ' ' + g
    G = str(g)

    fig = pl.figure(k)
    ax2 = pl.subplot2grid((2,1), (0,0), colspan=1,)
    ax2.plot (d['delay'+ Y], d['sig'+Y])
    ax2.set_xlabel ("Delay0F")
    ax2.set_ylabel ("Sig0F")
    ax2.set_title (str(d['run']))
    
    
    
    line, = ax2.plot (d['delay'+ Y], d['sig'+Y])
    line.set_label(G)
    ax2.legend()
    

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
    A = F[startpt:endpt]
    S = np.fft.rfft(A)
    
    S = S[:-1]
    u = np.abs(S)
    dNu = 1/(Nsize*dx)
    Nu = dNu*np.arange(Nsize/2)
    center = (300*(Nu[0] + Nu[-1]))/2
    top = max(np.abs(S))


    ax1 = pl.subplot2grid((2,1), (1,0),colspan =1)
    ax1.plot (300*Nu,np.abs(S)) 
    ax1.set_xlabel ("Frequency, GHz")
    ax1.set_ylabel ('Spectrum (arb)')
    
    m = time + ' Spectrum'
    M = str(m)
    
    s = 'sample frequency: %5.0f\r\nvelocity: %2.1f\r\ntime: %2.1f'% ((d['sample freq']), v, total_t)
    
    #L = 'sample frequency: ' + str((d['sample freq']))
    #V = 'velocity:' + str(v) + ','
    #T = 'time: ' + str(total_t)
    
  
   
    
    pl.text(center,top, s, verticalalignment='top')
