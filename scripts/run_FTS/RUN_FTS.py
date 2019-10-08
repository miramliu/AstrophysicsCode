import numpy as np
import scipy
import serial
import time
import glob
import os
import pickle
import sys
import zaber_commands as zc
import labjack_commands as lj
from fts_scan import fts_scan

from Tkinter import *

root = Tk()
root.wm_title("Basic Operations")

class App:

    def __init__(self, parent):

        frame = Frame(parent)
        frame.pack()
        parent.minsize(width = 300, height = 300)
        
        
        self.button = Button(
            frame, text="Initialize Labjack", fg="blue", command=self.Init
            )
        self.button.pack()
        
        self.j = StringVar()
        self.k = IntVar()
        self.l = IntVar()
        self.m = IntVar()
        self.n = IntVar()
        self.o = IntVar()
        self.p = IntVar()
        self.q = IntVar()
        self.SaveName = StringVar()
        
        self.dev = StringVar()
        self.show_1 = IntVar()
        
        self.I = StringVar()
        self.run_name = StringVar()
        self.niter = IntVar()
        self.max_d = IntVar()
        self.speed = IntVar()
        self.max_nu = IntVar()
        self.accel = IntVar()
        self.oversample = IntVar()
        self.gain = IntVar()
        
        
        #Serial Port Fram
        Sir= Frame(parent)
        Sir.pack()
        self.P = IntVar()
        
        self.kk = Label(Sir, text="Choose the zaber port:", fg = "black")
        self.kk.pack()
        
        self.RadP1 = Radiobutton(Sir, text = "port 1411", variable = self.P, command = self.choice1, value = 1)
        self.RadP1.pack(side = TOP, anchor ="w")
        
        self.RadP2 = Radiobutton(Sir, text = "port 1421", variable = self.P, command = self.choice2, value = 2)
        self.RadP2.pack(side = TOP, anchor = 'w')
        
        self.RadP3 = Radiobutton(Sir, text = 'COM3', variable = self.P, command = self.choice3, value = 3)
        self.RadP3.pack(side = TOP, anchor = "w")
        
        mam = Frame(parent)
        mam.pack()
        
        self.button = Button(
            mam, text="Open Serial Port", fg="blue", command=self.Port
            )
        self.button.pack()
        
        
        
        self.label1 = Label(mam, text = 'Enter parameters or load existing', fg = 'black')
        self.label1.pack(side = TOP)
        
        self.button3 = Button(
            mam, text = "Existing parameters", fg = "blue", command = self.LoadEx
            )
        self.button3.pack(side = TOP)
        
        
        In1 = Frame(parent)
        In1.pack() 
        self.name = Label(In1, text = "Run Name:", fg = "blue")
        self.name.pack(side = LEFT)
        
        self.NameEnt = Entry(In1, textvariable = self.j, width = 15)
        self.NameEnt.pack(side = LEFT)
        
       
        self.j.set("Name")
        
        In2 = Frame(parent)
        In2.pack() 
        
        self.niter = Label(In2, text = "niter:", fg = "blue")
        self.niter.pack(side = LEFT)
        
        self.NiterEnt = Entry(In2, textvariable = self.k, width = 5)
        self.NiterEnt.pack(side = LEFT)
        
        self.k.set("1")
        
        In3 = Frame(parent)
        In3.pack() 
        
        self.Max_d = Label(In3, text = "Max_d, mm:", fg = "blue")
        self.Max_d.pack(side = LEFT)
        
        self.Max_dEnt = Entry(In3, textvariable = self.l, width = 5)
        self.Max_dEnt.pack(side = LEFT)
        
        self.l.set("60")
        
        In4 = Frame(parent)
        In4.pack() 
        
        self.Speed = Label(In4, text = "Speed, mm/s:", fg = "blue")
        self.Speed.pack(side = LEFT)
        
        self.SpeedEnt = Entry(In4, textvariable = self.m, width = 5)
        self.SpeedEnt.pack(side = LEFT)
        
        self.m.set("10")
        
        In5 = Frame(parent)
        In5.pack() 
        
        self.Max_nu = Label(In5, text = "Max_nu, Ghz:", fg = "blue")
        self.Max_nu.pack(side = LEFT)
        
        self.Maxnuent = Entry(In5, textvariable = self.n, width = 5)
        self.Maxnuent.pack(side = LEFT)
        
        self.n.set("300")
        
        In6 = Frame(parent)
        In6.pack() 
        
        self.Accel = Label(In6, text = "Acceleration, mm/s^2:", fg = "blue")
        self.Accel.pack(side = LEFT)
        
        self.AccelEnt = Entry(In6, textvariable = self.o, width = 5)
        self.AccelEnt.pack(side = LEFT)
        
        self.o.set("10000")
        
        In7 = Frame(parent)
        In7.pack() 
        
        
        self.Oversample = Label(In7, text = "Oversample:", fg = "blue")
        self.Oversample.pack(side = LEFT)
        
        self.OversampleEnt = Entry(In7, textvariable = self.p, width = 5)
        self.OversampleEnt.pack(side = LEFT)
        
        self.p.set("16")
        
        #self.T = Message(In, text = "0 is -10V to 10V, 1 is -1V to 1V, 2 is -0.1V to 0.1V, 3 is -10 mV to 10 mV")
        #self.T.pack(side = RIGHT)
        
        In8 = Frame(parent)
        In8.pack() 
        
        self.Gain = Label(In8, text = "Gain:", fg = "blue")
        self.Gain.pack(side = LEFT)
        
        self.GainEnt = Entry(In8, textvariable = self.q, width = 5)
        self.GainEnt.pack(side = LEFT)
        
        self.q.set("2")
        
        
        In = Frame(parent)
        In.pack() 
        
        self.button1 = Button(
            In, text="Save", fg="blue", command=self.SaveParams
            )
        self.button1.pack(side = LEFT)
        
        self.button2 = Button(
            In, text = "Load", fg = "red", command = self.LoadNew
            )
        self.button2.pack(side = RIGHT)
        
        
        self.a = IntVar()
        
        
        
        done=Frame(parent)
        done.pack()
        
        button = Checkbutton(
            done, text="Show Keys", fg="black", variable = self.a, command=self.Show
            )
        button.pack(side = TOP)
        
        self.button = Button(
            done, text="Run", fg="red", command=self.Run
            )
        self.button.pack(side = TOP)
        
        
        self.button2 = Button(
            parent, text="Quit", fg="red", command=self.quit
            )
        self.button2.pack(side = LEFT)
        
        
        
        
    
    
        
    def Init(self):
        print 'init labjack'
        
    def choice1(self):
        self.P = "1411"
        print 'port 1411'
    def choice2(self):
        self.P = "1421"
        print 'port 1421'
    def choice3(self):
        self.P = "COM3"
        print "COM3"
        
    def Port(self):
        self.dev = '/dev/tty.usbmodem' + self.P
        dev = self.dev
        ser,err = zc.init_zaber(dev)
        
       
        
    def LoadEx(self):
        Pa = self.Pa = Toplevel()
        Pa.title ('Available Parameters')
        
        self.scrollbar = Scrollbar(Pa,orient = "vertical")
        self.listbox1 = Listbox(Pa, width = 20, height = 20, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command = self.listbox1.yview)
        self.scrollbar.config(command = self.listbox1.yview)
        
        self.scrollbar.pack(side = "right", fill = "y")
        self.listbox1.pack(side = "left", fill = "both", expand = True)

        self.listbox1.insert(END, "Existing Parameters: ")

        default_directory = '../../scripts/Parameters/' 
        default_extension = '*.txt'
    
        q=os.path.join(default_directory, default_extension)
        
        for name in glob.glob(q):
            self.listbox1.insert(END, name[25:-4])
            
        self.f =StringVar()
        
        def get_list(event):
            self.f = str(self.listbox1.get(self.listbox1.curselection()))
            
        self.listbox1.bind('<<ListboxSelect>>', get_list)
        
        self.button = Button(
            Pa, text="Load", fg="red", command=self.load
            )
        self.button.pack()
        
    def load(self):
        default_directory = '../../scripts/Parameters/' 
        g = os.path.join(default_directory, self.f)
        self.X = str(g) +'.txt'
        
        B = str(self.X)
        with open(B,"r") as f:
            lines=[line.strip() for line in f]
        self.run_name = lines[0]
        self.niter = lines[1]
        self.max_d = lines[2]
        self.speed = lines[3]
        self.max_nu = lines[4]
        self.accel = lines[5]
        self.oversample = lines[6]
        self.gain = lines[2]
        
        print B[25:-4] + ' parameters loaded'
    def LoadNew(self):
        self.I = self.SaveName.get()
        self.run_name = self.j.get()
        self.niter = self.k.get()
        self.max_d = self.l.get()
        self.speed = self.m.get()
        self.max_nu = self.n.get()
        self.accel = self.o.get()
        self.oversample = self.p.get()
        self.gain = self.q.get()
        
        print 'loaded run name: ' + str(self.run_name) + ' niter: ' + str(self.niter) + '  max_d ' + str(self.max_d) + ' speed ' + str(self.speed) + ' max_nu: ' + str(self.max_nu) + ' accel: ' + str(self.accel) + ' oversample: ' + str(self.oversample) + ' gain: ' + str(self.gain)
        
    def SaveParams(self):
        Ma = self.Ma = Toplevel()
        Ma.title ('Save Parameters')
        
        self.PNam = Label(Ma, text = "Set Parameter Name:", fg = "blue")
        self.PNam.pack(side = TOP)
        
        self.PNamEnt = Entry(Ma, textvariable = self.SaveName, width = 15)
        self.PNamEnt.pack(side = TOP)
        
        self.SaveName.set(self.j.get())
        
        self.button1 = Button(
            Ma, text="Upload", fg="blue", command=self.SaveParams1
            )
        self.button1.pack()
        
    def SaveParams1(self):
        self.I = self.SaveName.get()
        self.run_name = self.j.get()
        self.niter = self.k.get()
        self.max_d = self.l.get()
        self.speed = self.m.get()
        self.max_nu = self.n.get()
        self.accel = self.o.get()
        self.oversample = self.p.get()
        self.gain = self.q.get()
            
        
        w = ('run_name:', 'niter:', 'max_d:', 'speed:', 'max_nu:', 'accel:', 'oversample:', 'gain:')
        x = (self.run_name, self.niter, self.max_d, self.speed, self.max_nu, self.accel, self.oversample, self.gain)
        ParamName = '../../scripts/Parameters/' + self.I + '.txt'
        np.savetxt(str(ParamName),x, fmt="%s")
        np.c_[w,x]    
        g = self.I + ' parameters saved'
        print g
        
    def Show(self):
        if self.a.get() == 1:
            self.show_1 = 'on'
        else:
            self.show_1 = 'off'
        
    def Run(self): 
        
        d = lj.init_labjack()
        d
        
        run_name = self.run_name
        niter = self.niter
        max_d = self.max_d
        speed = int(self.speed)
        max_nu = self.max_nu
        accel = self.accel
        oversample = self.oversample
        gain = self.gain
        
        ser,err = zc.init_zaber(self.dev)
        
        
        err,data=fts_scan(run_name,
                  niter,
                  max_d,
                  speed,
                  max_nu,
                  oversample,
                  gain,
                  ser,
                  d)
        
        if self.show_1 == 'on':
            data.keys()
        
        fig1=figure(1,figsize=(15.,9.))
        #fig1.subplots_adjust(bottom=0.13,top=0.95,left=0.09,right=.98)
        ax1=fig1.add_subplot(211)
        ax2=fig1.add_subplot(212)
        ax1.plot(data['sig0F'])
        ax1.set_title('Run '+ data['run'] +' Signal vs data index')
        ax1.set_ylabel('Signal (V)')
        ax1.set_xlabel('Sample Number')
        ax1.grid()
        ax2.plot(data['delay0F'],data['sig0F'],label='Signal')
        ax2.plot(data['delay0F'],data['scan0F']/45.,label='Scan')
        ax2.plot(data['delay0F'],data['wlf0F']/45.,label='WLF')
        ax2.set_title('Run '+ data['run'] +' Signal vs Delay')
        ax2.set_ylabel('Signal (V)')
        ax2.set_xlabel('Delay (mm)')
        ax2.legend(loc='upper right')
        ax2.grid()
    
        fig2=figure(2,figsize=(15.,9.))
        #fig1.subplots_adjust(bottom=0.13,top=0.95,left=0.09,right=.98)
        ax1=fig2.add_subplot(211)
        ax2=fig2.add_subplot(212)
        ax1.plot(data['sig0R'])
        ax1.set_title('Run '+ data['run'] +' Signal vs data index')
        ax1.set_ylabel('Signal (V)')
        ax1.set_xlabel('Sample Number')
        ax1.grid()
        ax2.plot(data['delay0R'],data['sig0R'],label='Signal')
        ax2.plot(data['delay0R'],data['scan0R']/45.,label='Scan')
        ax2.plot(data['delay0R'],data['wlf0R']/45.,label='WLF')
        ax2.set_title('Run '+ data['run'] +' Signal vs Delay')
        ax2.set_ylabel('Signal (V)')
        ax2.set_xlabel('Delay (mm)')
        ax2.legend(loc='upper right')
        ax2.grid()
        ser.close()
        
    def quit(self):
        root.quit()
        
        
        
        
        
app = App(root)
root.mainloop()
