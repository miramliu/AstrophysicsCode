def error():
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import glob
    import os

    
    
    default_directory = '../software/' 
    stri = raw_input('file pathway: ')
    if len(stri)>1:
        default_directory = stri #change directory
    default_extension = '2016*.pkl'
    
    q=os.path.join(default_directory, default_extension)
    print q
    
    for name in glob.glob(q): 
        print name[12:]
        
    while True:
        
        f = raw_input('pickle file name: ')
        if len(f) == 0:
             break
            
            
        g = os.path.join(default_directory, f)
        X = str(g)
        
        file=open( X , 'rb')
        d=pickle.load(file)
        file.close()
        
        
        
        
        
        
        xx = d['delay0F']
        yy= d['sig0F']
          
    # add WLF at center with amplitude 20

        fig = plt.figure(1)
        yflip = np.flipud(yy)
        plt.plot(xx,yy,'-b')
        plt.plot(xx,yflip,'-r')
        
        
        fig = plt.figure(2)
        ysim = 0.5*(yy+yflip)
        yasym=0.5*(yy-yflip)
        plt.plot(xx,ysim,'-b',label='sim')
        plt.plot(xx,yasym,'-r',label='anti sym')
        plt.legend()
        plt.show()
        

    # define the three regions, side band for noise, center for WLF and as for asym term
        