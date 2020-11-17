import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import ascii
import glob, re
from astropy.io import fits


gcat=pd.read_csv('LLR_gaia/Gaia_Sample_v3.csv',index_col=False)
scat=pd.read_csv('LLR_seismic/Seismic_Sample_v3.csv',index_col=False)

gaia   =pd.read_csv('LLR_gaia/pande_final_sample_full.txt',index_col=False,delimiter=' ')
seismic=pd.read_csv('LLR_seismic/astero_final_sample_full.txt',index_col=False,delimiter=' ')
gaia=np.array(gaia)
seismic=np.array(seismic)
files=np.concatenate([gaia[:,0],seismic[:,0]])
loggs=np.concatenate([gaia[:,1],seismic[:,1]])
print(type(loggs[0]))
kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
allkps   =list(df['kic_kepmag'])
gaia     =ascii.read('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/DR2PapTable1.txt',delimiter='&')
kics   =[]
labels =[]
amp20  =[]
amp50  =[]
amp100 =[]
amp150 =[]
amp200 =[]
amp250 =[]

print('Getting spectral power...')
for i in range(0,len(files)): #
	file=files[i]
	kic   = int(file.split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
	b     = ascii.read(file)
	freq  = b['freq'] 
	power = b['power']
	labels.append(loggs[i])
	kics.append(kic)
	
	f=np.where(freq>20.)[0][0]
	p=power[f]
	amp20.append(p)
	
	f=np.where(freq>50.)[0][0]
	p=power[f]
	amp50.append(p)
	
	f=np.where(freq>100.)[0][0]
	p=power[f]
	amp100.append(p)
	
	f=np.where(freq>200.)[0][0]
	p=power[f]
	amp200.append(p)
	
	print(i)

print('Getting stellar parameters...')
rads=[]
lums=[]
teffs=[]
kps=[]
for kic in kics:
	row=kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
	try:
		t  =row['iso_teff'].item()
		r  =row['iso_rad'].item()
	except:
		idx=np.where(gaia['KIC']==kic)[0]
		t     =gaia['teff'][idx][0]
		r     =gaia['rad'][idx][0]
	l  =r**2.*(t/5777.)**4.
	kp =allkps[kp_kics.index(kic)]
	rads.append(r)
	lums.append(l)
	teffs.append(t)
	kps.append(kp)


plt.rc('font', size=15)                  # controls default text sizes
plt.rc('axes', titlesize=15)             # fontsize of the axes title
plt.rc('axes', labelsize=15)             # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)            # fontsize of the tick labels
plt.rc('ytick', labelsize=15)            # fontsize of the tick labels
plt.rc('axes', linewidth=1)  
plt.rc('legend', fontsize=15)
plt.rc('lines',markersize=1)

def getplot(y,a,b,c,d):
	ax1=plt.subplot(4,4,a)
	plt.scatter(labels,y,c='k',alpha=0.5)
	plt.ylim(5,2e5)
	plt.yscale('log')
	ax2=plt.subplot(4,4,b)
	plt.scatter(teffs,y,c='k',alpha=0.5)
	plt.ylim(5,2e5)
	plt.xlim(3200,7800)
	plt.yscale('log')
	plt.yticks([])
	plt.gca().invert_xaxis()
	ax3=plt.subplot(4,4,c)
	plt.scatter(lums,y,c='k',alpha=0.5)
	plt.ylim(5,2e5)
	plt.yscale('log')
	plt.xscale('log')
	plt.yticks([])
	ax4=plt.subplot(4,4,d)
	plt.scatter(kps,y,c='k',alpha=0.5)
	plt.ylim(5,2e5)
	plt.yscale('log')
	plt.yticks([])
	if a==1:
		ax1.set_ylim([5,2e5])
		ax2.set_ylim([5,2e5])
		ax3.set_ylim([5,2e5])
		ax4.set_ylim([5,2e5])
	if a==5:
		ax1.set_ylim([5,5e4])
		ax2.set_ylim([5,5e4])
		ax3.set_ylim([5,5e4])
		ax4.set_ylim([5,5e4])
	if a==9:
		ax1.set_ylim([1e0,1e4])
		ax2.set_ylim([1e0,1e4])
		ax3.set_ylim([1e0,1e4])
		ax4.set_ylim([1e0,1e4])
	if a==13:
		ax1.set_ylim([0.5,5e3])
		ax2.set_ylim([0.5,5e3])
		ax3.set_ylim([0.5,5e3])
		ax4.set_ylim([0.5,5e3])

	if a==1:
		STR='20 $\mu$Hz'
		t=ax1.text(0.03,0.05,s=STR,color='k',ha='left',va='bottom',transform = ax1.transAxes,fontsize=20)
		t.set_bbox(dict(facecolor='white',edgecolor='k'))#, alpha=0.5, edgecolor='red'))
	if a==5:
		STR='50 $\mu$Hz'
		t=ax1.text(0.03,0.05,s=STR,color='k',ha='left',va='bottom',transform = ax1.transAxes,fontsize=20)
		t.set_bbox(dict(facecolor='white',edgecolor='k'))#, alpha=0.5, edgecolor='red'))
	if a==9:
		STR='100 $\mu$Hz'
		t=ax1.text(0.03,0.05,s=STR,color='k',ha='left',va='bottom',transform = ax1.transAxes,fontsize=20)
		t.set_bbox(dict(facecolor='white',edgecolor='k'))#, alpha=0.5, edgecolor='red'))
	if a==13:
		STR='200 $\mu$Hz'
		t=ax1.text(0.03,0.05,s=STR,color='k',ha='left',va='bottom',transform = ax1.transAxes,fontsize=20)
		t.set_bbox(dict(facecolor='white',edgecolor='k'))#, alpha=0.5, edgecolor='red'))
		ax1.set_xlabel('Logg [dex]')
	if b==14:
		STR='T$_{eff}$ [K]'
		ax2.set_xlabel(STR)
	if c==15:
		ldot='L$_{\\odot}$'
		STR='Luminosity [{}]'.format(ldot)
		ax3.set_xlabel(STR)
	if d==16:
		STR='Kp'
		ax4.set_xlabel(STR)
	
# 1st ROW
fig=plt.figure(figsize=(20,12))

getplot(amp20,1,2,3,4)

# 2nd ROW
getplot(amp50,5,6,7,8)

# 3rd ROW
getplot(amp100,9,10,11,12)

# 4th ROW
getplot(amp200,13,14,15,16)

plt.tight_layout()
plt.subplots_adjust(wspace=0,hspace=0)
STR='PSD [ppm$^2/\mu$Hz]'
fig.text(-0.01, 0.5, STR, va='center', rotation='vertical')
plt.savefig('info_content.png',dpi=100,bbox_inches='tight')
# plt.savefig('Maryum_2.png',dpi=100,bbox_inches='tight')
# plt.show()













