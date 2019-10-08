import numpy as np
import matplotlib.pyplot as plt

Ns=10**3
xx = np.arange(0,Ns,1)
yy=np.random.randn(len(xx))
# add WLF at center with amplitude 20
yy=yy-20.0*np.exp(-0.005*(xx-0.5*Ns)**2)*np.cos(0.2*(xx-0.5*Ns))
# add PLF at 80% to the right with amplitude of 10
yy=yy-10.0*np.exp(-0.01*(xx-0.8*Ns)**2)*np.sin(0.1*(xx-0.8*Ns))

yflip = np.flipud(yy)

plt.plot(xx,yy,'-b')
plt.plot(xx,yflip,'-r')
plt.show()

ysim = 0.5*(yy+yflip)
yasym=0.5*(yy-yflip)

plt.plot(xx,ysim,'-b',label='sim')
plt.plot(xx,yasym,'-r',label='anti sym')
plt.legend()
plt.show()

# define the three regions, side band for noise, center for WLF and as for asym term
reg_sb = xx[0:0.1*Ns]
reg_cent = xx[0.45*Ns:0.55*Ns]
reg_as = xx[0.75*Ns:0.85*Ns]

print('Sym sig, noise = ', np.std(ysim[reg_cent]), np.std(ysim[reg_sb]))
print('ASym sig, noise = ', np.std(yasym[reg_as]), np.std(yasym[reg_sb]))
