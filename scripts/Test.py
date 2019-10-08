def glob():

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


def pickle():    
    import pickle
    
    
    f = raw_input ('Pickle File Name: ')
    X = str(f)

    file=open( X , 'rb')
    d=pickle.load(file)
    file.close()
    
    print d