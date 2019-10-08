#Basic analysis GUI for the FTI using simple deconvolution. Mira
import os
import glob
import itertools
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import numpy as np
import pickle 

#from Tkinter import * for python 2.x
from tkinter import *

root = Tk()
root.wm_title("Basic Operations")

class App:

    def __init__(self, parent):

        frame = Frame(parent)
        frame.pack()
        
        self.scrollbar = Scrollbar(frame,orient = "vertical")
        self.listbox1 = Listbox(frame, width = 40, height = 20, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command = self.listbox1.yview)
        self.scrollbar.config(command = self.listbox1.yview)
        
        self.scrollbar.pack(side = "right", fill = "y")
        self.listbox1.pack(side = "left", fill = "both", expand = True)

        self.listbox1.insert(END, "File Names: ")

        default_directory = '../../data/raw_data/' 
        default_extension = '2016*.pkl'
    
        q=os.path.join(default_directory, default_extension)
    

        for name in glob.glob(q):
            self.listbox1.insert(END, name[20:-4])
            
            
        self.f =IntVar()
        def get_list(event):
            self.f = str(self.listbox1.get(self.listbox1.curselection()))
            
            
        self.listbox1.bind('<<ListboxSelect>>', get_list)
        
        
        self.buttonC = Button(
            parent, text = 'Choose Specific Date', fg = "blue", command = self.choice)
        self.buttonC.pack()
        
        self.button = Button(
            parent, text="Load", fg="red", command=self.load
            )
        self.button.pack()
        
        self.w = Label(parent, text="Choose Scan Direction:", fg = "blue")
        self.w.pack()
        
        #Direction frame
        Direc= Frame(parent)
        Direc.pack()
        #variable for radiobutton (Forward or Reverse)
        self.T = IntVar()
        self.K = StringVar()
        self.RadF = Radiobutton(Direc, text = "Forward", variable = self.T, command = self.choice1, value = 1)
        self.RadF.pack(side = LEFT, anchor ="w")
        self.RadR = Radiobutton(Direc, text = "Reverse", variable = self.T, command = self.choice2, value = 2)
        self.RadR.pack(side = LEFT, anchor = "w")
        
        self.kk = Label(parent, text="Choose Scan Number:", fg = "blue")
        self.kk.pack()
        
        b = Entry(parent, textvariable = self.K)
        b.pack()
        self.K.set("0")
        
        self.ww = Label(parent, text = "Choose Actions:", fg = "blue")
        self.ww.pack()
        
        #variables for the analysis commands 
        self.x = IntVar()
        self.y = IntVar()
        self.z = IntVar()
        self.a = IntVar()
        self.b = IntVar()
        self.c = IntVar()
        
        
        #variables for run
        self.X = IntVar()
        self.time = IntVar()
        self.Analysis = IntVar()
        self.Interferogram = IntVar()
        self.Spectrum = IntVar()
        self.SaveInt = IntVar()
        self.SaveSpec = IntVar()
        self.SaveOut = IntVar()
        
        
        #variables in run
        self.xint = IntVar()
        self.yint = IntVar()
        self.xspec = IntVar()
        self.yspec = IntVar()
        self.p = IntVar()
        self.j = IntVar()
        self.total_t = IntVar()
        
        #variables for Advanced commands
        self.A = IntVar()
        self.B = IntVar()
        self.C = IntVar()
        self.D = IntVar()
        self.E = IntVar()
        self.F = IntVar()
        self.G = IntVar()
        self.H = IntVar()
        self.I = IntVar()
        self.O = IntVar()
        
        # variables for Calculate
        self.WLF = IntVar()
        self.SYMMINUS = IntVar()
        self.SYMPLUS  = IntVar()
        self.PLF = IntVar()
        self.BLN = IntVar()
        self.SAVE = IntVar()
        self.SWLF = IntVar()
        self.SPLF = IntVar()
        self.ratio = IntVar()
        
        #Analysis frame
        Act = Frame(parent)
        Act.pack()
        Analysis = Checkbutton(Act, text = "Do Analysis", variable = self.x, command = self.Analysis_check)
        Analysis.pack(side = TOP)
        
        ShowInterferogram = Checkbutton(Act, text = "Plot Inteferogram", variable = self.y, command = self.PlotInt_check)
        ShowInterferogram.pack(side = TOP)
        
        ShowSpectrum = Checkbutton(Act, text = "Plot Spectrum", variable = self.z, command = self.PlotSpec_check)
        ShowSpectrum.pack(side = TOP)
        
        SaveInterferogram = Checkbutton(Act, text = "Save Inteferogram", variable = self.a, command = self.savePlotInt_check)
        SaveInterferogram.pack(side = TOP)
        
        SaveSpectrum = Checkbutton(Act, text = "Save Spectrum", variable = self.b, command = self.savePlotSpec_check)
        SaveSpectrum.pack(side = TOP)
        
        SaveOut = Checkbutton(Act, text = "Save Output", variable = self.c, command = self.saveOut_check)
        SaveOut.pack(side = TOP)
        
        In = Frame(parent)
        In.pack()
        
        self.www = Label(In, text = "Plot Number:", fg = "blue")
        self.www.pack(side = LEFT)
        
        self.Number = Entry(In, textvariable = self.j, width = 3)
        self.Number.pack(side = LEFT)
        
        self.j.set("1")
        
        self.s = Label(In, text = "Plot Color:", fg = "blue")
        self.s.pack(side = LEFT)
        
        self.color = StringVar()
        self.SetColor = Entry(In, textvariable = self.color, width = 3)
        self.SetColor.pack(side = LEFT)
        self.color.set("r")
        
        pp = Label(In, text = "Set i:", fg = 'blue')
        pp.pack(side = LEFT)
        
        self.W = StringVar()
        seti = Entry(In, textvariable = self.W, width = 3)
        seti.pack(side = LEFT)
        self.W.set("8")
        
        #final actions
        self.button1 = Button(
            parent, text="Run", fg="red", command=self.run,
            )
        self.button1.pack()
        
        self.button2 = Button(
            parent, text="Quit", fg="red", command=self.quit
            )
        self.button2.pack(side = LEFT)
        
        self.button3 = Button(
            parent, text = 'Advanced', fg = 'green', command = self.window
            )
        self.button3.pack(side = RIGHT)
        
        #Functions
    def load(self):
        #self.f = str(self.listbox1.get(self.listbox1.curselection()))
        default_directory = '../../data/raw_data/' 
        g = os.path.join(default_directory, self.f)
        self.X = str(g) +'.pkl'
        
        print (self.f + ' loaded')
        
        
        date = self.f[0:13]
        year = date[0:4]
        month = date[4:6]
        day = date[6:8]
        hr = date[9:11]
        sc = date[11:13]

        self.time= year + ' ' + month + '/' + day + ' ' + hr + ':' + sc
        
    def choice (self): 
        choice = self.choice = Toplevel()
        choice.title ('Input Date')
        
        default_directory = '../../data/raw_data/' 
        default_extension = '2016*.pkl'
        
        self.CH=StringVar()
        bb = Entry(choice, textvariable = self.CH)
        bb.pack()
        
        
        Appl= Button(
            choice, text = 'Apply', fg = 'blue',command = self.Appl)
        Appl.pack()
    
        labelnote = Label(choice, text = 'Input it as year,month,day.\n For example, may 12th 2014 is 20140512',fg = "black")
        labelnote.pack()
        
    def Appl(self): 
        self.listbox1.delete(0, END)
        self.CHO=StringVar()
        self.CHO = self.CH.get()
        default_directory = '../../data/raw_data/'
        default_extension = str(self.CHO) + '*.pkl'
        
        qq=os.path.join(default_directory, default_extension)
        
        for name in glob.glob(qq):
            self.listbox1.insert(END, name[20:-4])
        
    def choice1(self):
        self.T = "F"
        print ('you chose Forward')
    def choice2(self):
        self.T = "R"
        print ('you chose Reverse')
        
        
    def Analysis_check(self):
        if self.x.get() == 1:
            self.Analysis = 'on'
        else:
            self.Analysis = 'off'
    def PlotInt_check(self):
        if self.y.get() == 1:
            self.Interferogram = 'on'
        else:
            self.Interferogram = 'off'
    def PlotSpec_check(self):
        if self.z.get() == 1:
            self.Spectrum = 'on'
        else:
            self.Spectrum = 'off'
    def savePlotInt_check(self):
        if self.a.get() == 1:
            self.SaveInt = 'on'
        else: 
            self.saveInt = 'off'
    def savePlotSpec_check(self):
        if self.b.get() ==1:
            self.SaveSpec = 'on'
        else:
            self.SaveSpec = 'off'
    def saveOut_check(self):
        if self.c.get() ==1:
            self.SaveOut = 'on'
        else:
            self.SaveOut = 'off'
            
    
    def run(self):
        B = str(self.X)
        file=open(B , 'rb')
        d=pickle.load(file)
        file.close()
        self.k = IntVar()
        self.seti=IntVar()
        
        self.k = self.K.get()
        self.t = self.k + self.T
        self.J = self.j.get()
        self.seti = self.W.get()
        if self.Analysis == 'on':
            i = int(self.seti)#adjust to number of data points
            Nsize = 2**i
            dt=(1/(d['sample freq']))
            t=dt*np.arange(Nsize)
            self.p=(d['speed'])
            x = self.p*t
            dx = dt*self.p #self.p is velocity
            self.total_t = (d['scan time'])
    
            total_s = (d['samples requested'])
            startpt = ((total_s - Nsize)/2) 
            endpt = startpt + Nsize
    
            F = (d['sig'+self.t])
            A = np.hanning(Nsize)*F[startpt:endpt]
            S = np.fft.rfft(A)
            
            a = d['delay' + self.t]
            b = a[startpt:endpt]
            
    
            s = S[:-1]
            u = np.abs(s)
            dNu = 1/(Nsize*dx)
            Nu = dNu*np.arange(Nsize/2)
            NuFull = dNu*np.arange((Nsize/2) + 1)
            center = (300*(Nu[0] + Nu[-1]))/2
            top = max(np.abs(s))
            
            tau = .04 #ms
            k_v = 1/((1/tau)+1j*2*np.pi*300*NuFull)
            
            
            n = 3 #greater number filters out more high frequencies
            sig = n/(tau*(2*np.pi)) 
            
            f = (1/(sig*np.sqrt(np.pi)))*np.exp(-(NuFull**2)/(sig**2)) #gaussian
            B = f*np.fft.rfft(A)/k_v*tau  #fourier transform of interferogram/kernel
            c = np.fft.irfft(B) #inverse fourier transform of deconvolved spectrum (deconvolved interferogram)
            
            # normalize gaussian to get rid of noise (divide by integral of itself) (in B)
            
            g = (np.abs(np.fft.rfft(c))/k_v*tau)[:2**7] #CHANGE depending on number of data points
            L=np.sum(g) #integral of spectrum
            
            self.xint, self.yint = b, (1/L)*c

            self.xspec, self.yspec = 300*NuFull,np.abs(np.fft.rfft(c))/k_v*tau
        if self.SaveInt == 'on': #saves plot of interferogram in processed data folder
            pl.plot (self.xint, self.yint)
            pl.title(str(d['run']))
            k= self.f + 'Int' + '.png'
            K = '../../data/processed_data/'+k
            pl.savefig(K)
            
        if self.SaveSpec == 'on': #saves plot of spectrum in processed data folder
            pl.plot (self.xspec, self.yspec)
            pl.title(str(d['run']))
            #pl.xlim(0,900)
            k= self.f +'Spec' + '.png'
            K = '../../data/processed_data/'+k
            pl.savefig(K)
        if self.SaveOut == 'on': #saves all of data in a textfile in processed data folder
            Int = '../../data/processed_data/' + self.f + '_Interferogram' + '.txt'
            np.savetxt(Int,np.c_[self.xint,self.yint])
            Spec = '../../data/processed_data/' + self.f + '_Spectrum' + '.txt'
            np.savetxt(Spec, np.c_[self.xspec,self.yspec])
            
        if self.Interferogram == 'on' and self.Spectrum == 'on':
            pl.ion
            fig = pl.figure(self.J)
            ax2 = pl.subplot2grid((2,1), (0,0), colspan=1,)
            ax1 = pl.subplot2grid((2,1), (1,0),colspan =1)
            
            figattr = [self.J,fig,ax1,ax2] #list of attributes: plot#, figure, axes
            
            figattr[3].plot (self.xint,self.yint)
            figattr[3].set_xlabel ("Delay0F")
            figattr[3].set_ylabel ("Sig0F")
            figattr[3].set_title (str(d['run']))
            
            line, = figattr[3].plot (self.xint, self.yint)
            g = self.time + ' ' + self.t
            G = str(g)
            line.set_label(G)
            figattr[3].legend()
            
            figattr[2].plot (self.xspec, self.yspec) 
            n = 3
            sig = n/(tau*(2*np.pi))
            f = (1/(sig*np.sqrt(np.pi)))*np.exp(-(NuFull**2)/(sig**2))
            #f = f*np.max(self.yspec)/np.max(f) make into a parameter?
            
            figattr[2].plot (self.xspec, np.abs(f))
            figattr[2].set_xlabel ("Frequency, GHz")
            figattr[2].set_ylabel ('Spectrum (arb)')
    
            m = self.time + ' Spectrum'
            M = str(m)
    
            s = 'sample frequency: %5.0f\r\nvelocity: %2.1f\r\ntime: %2.1f'% ((d['sample freq']), self.p, self.total_t)
        
            figattr[2].text(center,top, s, verticalalignment='top')
            
            pl.show()
            
        else: 
            if self.Interferogram == 'on':
                
                self.Color = self.color.get()
                
                fig = pl.figure(self.J)
                #pl.plot (self.xint, self.yint)
                pl.title(str(d['run']))
                line, = pl.plot (self.xint, self.yint,self.Color , linewidth = .5)
                
                
                g = self.time + ' ' + self.t
                G = str(g)
                line.set_label(G)
                
                
                
                pl.legend()
                pl.show()
                
            if self.Spectrum == 'on':
                
                self.Color = self.color.get()
                
                fig = pl.figure(self.J)
                #pl.plot (self.xspec, self.yspec)
                pl.title(str(d['run']))
                line, = pl.plot (self.xspec, self.yspec, self.Color)
                m = self.time + ' Spectrum'
                M = str(m)
                line.set_label(M)
                
                
                pl.legend()
                pl.show()
    
                #s = 'sample frequency: %5.0f\r\nvelocity: %2.1f\r\ntime: %2.1f'% ((d['sample freq']), self.p, self.total_t)
        
                #pl.text(center,top, s, verticalalignment='top')
                
        
            
            
    def quit(self):
        root.quit()
        
    def window(self): 
        top = self.top = Toplevel()
        top.title ('Advanced Operations')
        #self.top.geometry("%dx%d%+d%+d" % (500, 500, 250, 125))
        label1 = Label(top, fg = 'blue', text = 'Standard Deviation')
        label1.pack()
        
        wlf = Checkbutton(top, text = 'White Light Fringe', variable = self.A, command = self.wlf)
        wlf.pack(side = TOP)
        
        symminus = Checkbutton(top, text = "Deviation on negative side", variable = self.B, command = self.symminus)
        symminus.pack(side = TOP)
        
        symplus = Checkbutton(top, text = "Deviation on positive side", variable = self.C, command = self.symplus)
        symplus.pack(side = TOP)
        
        plf = Checkbutton(top, text = "Phantom Light Fringe", variable = self.D, command = self.plf)
        plf.pack(side = TOP)
        
        bln = Checkbutton(top, text = "Edges", variable = self.E, command = self.bln)
        bln.pack(side = TOP)
        
        label2 = Label(top, fg = 'blue', text = 'Symmetry')
        label2.pack()
        
        swlf = Checkbutton(top, text = 'White Light Fringe Symmetry', variable = self.G, command = self.swlf)
        swlf.pack(side = TOP)
        
        splf = Checkbutton(top, text = 'Phantom Light Fringe Symmetry', variable = self.H, command = self.splf)
        splf.pack(side = TOP)
        
        tsm = Checkbutton(top, text = 'Total Symmetry Measure', variable = self.O, command = self.tsm)
        tsm.pack(side = TOP)
        
        #label3 = Label(top, fg = 'blue', text = 'Ratio of spectrums')
        #label3.pack()
        
        #ref = Button(top, text = 'load reference', command = self.loadref)
        #ref.pack(side = TOP)
        
        #filt= Button(
            #top, text = 'load filter', command = self.loadfilt)
        #filt.pack(side = TOP)
        
        #Ratio = Checkbutton(top, text = "Ratio of Spectrum", variable = self.I, command = self.CalcRatio)
        #Ratio.pack(side = TOP)
        
        
        saveall = Checkbutton(top, text = 'Save All', variable = self.F, command = self.saveall)
        saveall.pack(side = TOP)
        
        self.button4 = Button(
            top, text="Calculate", fg="red", command=self.Calculate,
            )
        self.button4.pack()
        
        self.button5 = Button(
            top, text="Close", fg="red", command=self.quitAdvanced
            )
        self.button5.pack(side = LEFT)
        
        
        
    def wlf(self):
        if self.A.get() == 1:
            self.WLF = 'on'
        else:
            self.WLF = 'off'
    def symminus(self):
        if self.B.get() == 1:
            self.SYMMINUS = 'on'
        else:
            self.SYMMINUS = 'off'
    def symplus(self):
        if self.C.get() ==1:
            self.SYMPLUS = 'on'
        else:
            self.SYMPLUS = 'off'
    def plf(self):
        if self.D.get() ==1:
            self.PLF = 'on'
        else:
            self.PLF = 'off'
    def bln(self):
        if self.E.get() ==1:
            self.BLN = 'on'
        else:
            self.BLN = 'off'
    def saveall(self):
        if self.F.get() ==1:
            self.SAVE = 'on'
        else:
            self.SAVE = 'off'
    def swlf(self):
        if self.G.get() ==1:
            self.SWLF= 'on'
        else:
            self.SWLF = 'off'
    def splf(self):
        if self.H.get() ==1:
            self.SPLF = 'on'
        else:
            self.SPLF = 'off'
    def tsm(self):
        if self.O.get() ==1:
            self.TSM = 'on'
        else:
            self.SPLF = 'off'
    #def loadref(self):
        #self.Ref = IntVar()
        #self.Ref = str(self.listbox1.get(self.listbox1.curselection()))
        #print 'Reference is ' + self.Ref
    #def loadfilt(self):
        #self.Filt = IntVar()
        #self.Filt = str(self.listbox1.get(self.listbox1.curselection()))
        #print 'Filter is ' + self.Filt
    #def CalcRatio(self):
        #if self.I.get()==1:
            #self.ratio = 'on'
        #else:
            #self.ratio = 'off'
        
        
            
    def Calculate(self):
        xx,yy = self.xint,self.yint
        result = ''
        if self.TSM =='on':
            x1,y1 = self.xint[:-35],self.yint[:-35]
            sym_tsm = (y1[(xx>=0)] - y1[(xx<=0)])/(y1[(xx>=0)] + y1[(xx<=0)])
            print (sym_tsm)
        if self.WLF == 'on':
            std_wlf = np.std(yy[(xx >= 0) & (xx <= 3.2)])
            print ('white light fringe = ' + str(std_wlf))
            result = 'white light fringe = '+ str(std_wlf)
        if self.SYMMINUS== 'on':
            std_symminus = np.std(yy[(xx >= -10) & (xx <= -5)])
            print ('standard deviation on negative side = ' + str(std_symminus))
            result = result + '\r\nstandard deviation on negative side = ' + str(std_symminus)
        if self.SYMPLUS== 'on':
            std_symplus = np.std(yy[(xx >= 5) & (xx <= 13)])
            print ('standard deviation on positive side = ' + str(std_symplus))
            result = result + '\r\nstandard deviation on positive side = '+ str(std_symplus)
        if self.PLF== 'on':
            std_plf = np.std(yy[(xx >= 17) & (xx <= 20)])
            print ('phantom light fringe = ' + str(std_plf))
            result = result + '\r\nphantom light fringe = ' + str(std_plf)
        if self.BLN== 'on':
            std_bln = np.std([yy[-50:-35], yy[45:60]])
            print ('edges = ' + str(std_bln))
            result = result + '\r\nedges = ' + str(std_bln)
        #if self.SWLF == 'on':
            #print 'working on it'
            #y_plus = yy
            #y_minus = np.flipup(y_plus) #Must be at least 2-d? 
            #y_sym = (y_plus + y_minus)/2
            #y_antisym = (y_plus - y_minus)/2
            #var1
            #result = result + '\r\nsymmetry = ' + str(y_sym)
        #if self.SPLF == 'on':
            #print 'still working on it'
            #y_anti = (yy[(xx<=0)] - yy[(xx>=0)])/2
            #print 'Y Antisymmetry = ' + str(y_anti)
            #result = result + '\r\nantisymmetry = ' + str(y_anti)
        if self.SAVE == 'on':
            print (result)
            #STDS = '../../data/processed_data/' + self.f + 'STDS' + '.txt'
            #np.savetxt(STDS,result)
            # get error "IndexError: tuple index out of range" why??
        #if self.ratio == 'on':
            #self.k = IntVar()
        
            #self.k = self.K.get()
            #self.t = self.k + self.T
            #Reference
            
            #default_directory = '../../data/raw_data/' 
            #g = os.path.join(default_directory, self.Ref)
            #self.Ref = str(g) +'.pkl'
            #b = str(self.Ref)
            #file=open(b , 'rb')
            #T=pickle.load(file)
            #file.close()
            
            #Nsize = 2**13
            #total_s = (T['samples requested'])
            #startpt = ((total_s - Nsize)/2) 
            #endpt = startpt + Nsize
            
            #F1 = (T['sig'+self.t])
            #A1 = np.hanning(Nsize)*F1[startpt:endpt]
            #S1 = np.fft.rfft(A1)
    
            #S1 = S1[:-1]
            #u1 = np.abs(S1)
            
            #filter
            #default_directory = '../../data/raw_data/' 
            #g = os.path.join(default_directory, self.Filt)
            #self.Filt = str(g) +'.pkl'
            #c = str(self.Filt)
            #file=open(c , 'rb')
            #t=pickle.load(file)
            #file.close()
        
            #F2 = (t['sig'+self.t])
            #A2 = np.hanning(Nsize)*F2[startpt:endpt]
            #S2 = np.fft.rfft(A2)
    
            #S2 = S2[:-1]
            #u2 = np.abs(S2)
            
            #print 'Ratio is' + u2/u1
            
            
            
    def quitAdvanced(self):
        self.top.destroy()
        
        
        
app = App(root)
root.mainloop()

