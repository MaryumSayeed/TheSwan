import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from astropy.stats import mad_std
from astropy.io import ascii

pcat=ascii.read('LLR_gaia/Gaia_Sample_v5.csv')#,skiprows=1,usecols=[0,19],index_col=False,names=['KIC','Outlier'])
acat=ascii.read('LLR_seismic/Seismic_Sample_v5.csv')#,skiprows=1,usecols=[0,19],index_col=False,names=['KIC','Outlier'])

acat=acat[acat['Outlier']==0]
pcat=pcat[pcat['Outlier']==0]

pcat,acat=np.array(pcat['KICID']),np.array(acat['KICID'])
print(len(pcat),len(acat))

afile1=np.loadtxt('baseline_cuts/astero_final_sample_1.txt',dtype=str,delimiter=' ')#,usecols=[0])
afile2=np.loadtxt('baseline_cuts/astero_final_sample_2.txt',dtype=str,delimiter=' ')#,usecols=[0])
afile3=np.loadtxt('baseline_cuts/astero_final_sample_3.txt',dtype=str,delimiter=' ')#,usecols=[0])
afile4=np.loadtxt('baseline_cuts/astero_final_sample_4.txt',dtype=str,delimiter=' ')#,usecols=[0])

pfile1=np.loadtxt('baseline_cuts/pande_pickle_1.txt',dtype=str,delimiter=' ')#,usecols=[0])
pfile2=np.loadtxt('baseline_cuts/pande_pickle_2.txt',dtype=str,delimiter=' ')#,usecols=[0])
pfile3=np.loadtxt('baseline_cuts/pande_pickle_3.txt',dtype=str,delimiter=' ')#,usecols=[0])

astero=np.concatenate([afile1,afile2,afile3,afile4],axis=0)
pande =np.concatenate([pfile1,pfile2,pfile3],axis=0)

astero_files,astero_true=astero[:,0],(astero[:,1]).astype(np.float)
pande_files,pande_true  =pande[:,0],(pande[:,1]).astype(np.float)

# get KICIDs in Pande sample:
pande_index=[]
for i in range(0,len(pande_files)): 
	file=pande_files[i]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))

	if kic in pcat:
		pande_index.append(i)

# get KICIDs in Astero. sample:
astero_index=[]
for i in range(0,len(astero_files)): 
	file=astero_files[i]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic in acat:
		astero_index.append(i)

def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms

def get_og_ivals():
	# load inferred values for full light curve analysis:
	adir='LLR_seismic/'
	pdir='LLR_gaia/'

	ia=np.load(adir+'labels_m1.npy')[astero_index]
	ip=np.load(pdir+'labels_m1.npy')[pande_index]
	return ia,ip


def get_inferred_vals(frac):
	adir='baseline_cuts/{}_days/LLR_seismic/'.format(frac,frac)
	pdir='baseline_cuts/{}_days/LLR_gaia/'.format(frac,frac)

	ia=np.load(adir+'labels_m1.npy')[astero_index]
	ip=np.load(pdir+'labels_m1.npy')[pande_index]

	return ia,ip


def get_text(ax,inferred,true):
	bias,rms=returnscatter(inferred-true)
	STR1='RMS: '+str('{0:.2f}'.format(rms))
	STR2='Bias: '+str('{0:.2f}'.format(bias))
	t=ax.text(0.03,0.92,s=STR1,color='k',ha='left',va='center',transform = ax.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
	t=ax.text(0.03,0.52,s=STR2,color='k',ha='left',va='center',transform = ax.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
	return bias,rms

def get_plot(true,infer,labelname,n):
	# ax=plt.subplot(2,3,n)
	# plt.plot(true,true,c='k',linestyle='--')
	# plt.scatter(true,infer,s=10,label=labelname)
	# bf,rf=get_text(ax,infer,true)
	std=mad_std(infer-true)
	bias,rms=returnscatter(infer-true)
	print(labelname,std,rms)
	# plt.legend(loc='lower right')
	return bias,rms,std

af,pf=get_og_ivals()
a14,p14=get_inferred_vals('14')
a27,p27=get_inferred_vals('27')
a48,p48=get_inferred_vals('48')
a65,p65=get_inferred_vals('65')

pt=pande_true[pande_index]
at=astero_true[astero_index]

# PANDE PLOTS:

plt.rc('font', size=15)                  # controls default text sizes
plt.rc('axes', titlesize=12)             # fontsize of the axes title
plt.rc('axes', labelsize=12)             # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
plt.rc('figure', titlesize=12)           # fontsize of the figure title
plt.rc('axes', linewidth=1)    
plt.rc('axes', axisbelow=True)
sig_rms=r'$\sigma$'
sig_mad=r'$\sigma_{\mathrm{mad}}$'
ystr=sig_rms+' [dex]'

rmsc  ='#404788FF'
stdc = '#55C667FF'
	
xls=['14','27','48','65','96'] 

# PANDE:
print('\n','GAIA')
pbf,prf,psf=get_plot(pt,pf,'Full',1)
pb14,pr14,ps14=get_plot(pt,p14,'14',5)
pb27,pr27,ps27=get_plot(pt,p27,'27',2)
pb48,pr48,ps48=get_plot(pt,p48,'48',3)
pb65,pr65,ps65=get_plot(pt,p65,'65',4)

#print(pr27,prf,(pr27-prf)/prf)
#print(ps27,psf,(ps27-psf)/psf)

# ASTERO
print('\n','SEISMIC')
bf,arf,asf=get_plot(at,af,'Full',1)
b14,ar14,as14=get_plot(at,a14,'14',5)
b27,ar27,as27=get_plot(at,a27,'27',2)
b48,ar48,as48=get_plot(at,a48,'48',3)
b65,ar65,as65=get_plot(at,a65,'65',4)

#print(ar27,arf,(ar27)/arf)
#print(as27,asf,(as27)/asf)


print(ar14,as14,ar27,as27)
print(pr14,ps14,pr27,ps27)
exit()
fig=plt.figure(figsize=(5,8))
xls=['14','27','48','65','96'] 

ax3=plt.subplot(211)
plt.grid(which='major', axis='y', linestyle='--')
plt.bar(xls,[pr14,pr27,pr48,pr65,prf],width=0.7,color=rmsc,edgecolor='k',label=sig_rms,fill='False',hatch='/',alpha=0.7)
plt.bar(xls,[ps14,ps27,ps48,ps65,psf],width=0.7,color=stdc,edgecolor='k',label=sig_mad,hatch='\\',alpha=0.7)
plt.ylabel(ystr)

STR='Gaia'
t=ax3.text(0.05,0.05,s=STR,color='k',ha='left',va='bottom',transform = ax3.transAxes)
t.set_bbox(dict(facecolor='white',edgecolor='k'))#, alpha=0.5, edgecolor='red'))
plt.minorticks_on()
ax3.tick_params(axis='x', which='minor',bottom=False)

ax4=plt.subplot(212)
plt.grid(which='major', axis='y', linestyle='--')
plt.bar(xls,[ar14,ar27,ar48,ar65,arf],width=0.7,color=rmsc,edgecolor='k',label=sig_rms,fill='False',hatch='/',alpha=0.7)
plt.bar(xls,[as14,as27,as48,as65,asf],width=0.7,color=stdc,edgecolor='k',label=sig_mad,hatch='\\',alpha=0.7)
plt.ylabel(ystr)

STR='Seismic'
t=ax4.text(0.05,0.05,s=STR,color='k',ha='left',va='bottom',transform = ax4.transAxes,zorder=1)
t.set_bbox(dict(facecolor='white',edgecolor='k'))#, alpha=0.5, edgecolor='red'))
plt.minorticks_on()
ax4.tick_params(axis='x', which='minor',bottom=False)
plt.legend(loc='upper right')
#ax4.tick_params(axis='x', which='major',top='on',direction='inout',length=6)

ax4.set_xlabel('Time-series length [days]')#,labelpad=10)
fig.tight_layout()
pys=[pr14,pr27,pr48,pr65,prf]
ays=[ar14,ar27,ar48,ar65,arf]
for i in range(0,len(xls)):
	print(xls[i],'P',pys[i],'A',ays[i])
plt.show(False)



# plt.savefig('timeseries_paper_plot.png',dpi=100,bbox_inches='tight')
# plt.savefig('Maryum_1.png',dpi=100,bbox_inches='tight')
exit()