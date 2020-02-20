import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy.stats import gaussian_kde
from scipy import stats
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['axes.linewidth'] = 1.

gaia     =ascii.read('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/DR2PapTable1.txt',delimiter='&')

gaia_stars=np.loadtxt('pande_final_sample_full.txt',usecols=[0],dtype=str)
ast_stars=np.loadtxt('astero_final_sample_full.txt',usecols=[0],dtype=str)

gaia_teff=[]
gaia_lum=[]
for file in gaia_stars:
	kic=file[0:-3].split('/')[-1].split('-')[0].split('kplr')[-1]
	kic=int(kic.lstrip('0'))
	idx=np.where(gaia['KIC']==kic)[0]
	t=gaia['teff'][idx][0]
	r=gaia['rad'][idx][0]
	l=r**2.*(t/5777.)**4.
	gaia_teff.append(t)
	gaia_lum.append(l)


ast_teff=[]
ast_lum=[]
for file in ast_stars:
	kic=file[0:-3].split('/')[-1].split('-')[0].split('kplr')[-1]
	kic=int(kic.lstrip('0'))
	idx=np.where(gaia['KIC']==kic)[0]
	t=gaia['teff'][idx][0]
	r=gaia['rad'][idx][0]
	l=r**2.*(t/5777.)**4.
	ast_teff.append(t)
	ast_lum.append(l)

print(len(ast_teff),len(gaia_teff))


fig=plt.figure(figsize=(7,10))
ax = fig.add_subplot(111)    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('Luminosity (Solar)',fontsize=15,labelpad=30)

ax1 = fig.add_subplot(2,1,1)
numBins=35

hb = ax1.hexbin(gaia_teff,gaia_lum,gridsize=numBins,bins='log',yscale='log',lw=0.5,edgecolor='face',cmap='viridis',mincnt=1)

#im1=ax.scatter(gaia_teff,gaia_lum,c=z_gaia)
ax1.set_ylim([0.15,200])
ax1.set_xlim([4420,7300])
ax1.set_yscale("log")
ax1.text(7200,90,'Gaia: '+str(len(gaia_teff))+' stars',fontsize=15)
cb = plt.colorbar(mappable=hb,pad=0.03)#,format='%.0e')
cb.set_label('$\log_{10}(\mathrm{Count})$',fontsize=15)
cb.ax.tick_params(labelsize=15)
ax1.tick_params(which='major',axis='both',labelsize=15)
plt.gca().invert_xaxis()
#plt.savefig('Gaia_HR_Diagram.pdf',dpi=50)

ax2 = fig.add_subplot(2,1,2)
numBins=70

hb = ax2.hexbin(ast_teff,ast_lum,gridsize=numBins,bins='log',yscale='log',lw=1,edgecolor='face',cmap='viridis',mincnt=1)

ax2.set_ylim([0.04,4200])
ax2.set_yscale("log")
#ax2.set_ylabel('Luminosity (Solar)',fontsize=20,labelpad=10)
ax2.set_xlabel('Effective Temperature (K)',fontsize=15)
ax2.text(7000,1000,'Asteroseismic: '+str(len(ast_teff))+' stars',fontsize=15)
cb = plt.colorbar(mappable=hb,pad=0.03)#,format='%.0e')
# cb=fig.colorbar(hb,ax=ax2)
cb.set_label('$\log_{10}(\mathrm{Count})$',fontsize=15)
cb.ax.tick_params(labelsize=15)
ax2.tick_params(which='major',axis='both',labelsize=15)
plt.gca().invert_xaxis()
plt.tight_layout()
#plt.savefig('Both_HR_Diagrams.pdf',dpi=50)
# plt.show(True)


# just in case:
# xy = np.vstack([gaia_teff,gaia_lum])
# z_gaia = gaussian_kde(xy)(xy)

