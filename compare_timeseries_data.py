import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

pcat=pd.read_csv('Pande_Catalogue.txt',skiprows=1,usecols=[0,16],index_col=False,delimiter=';',names=['KIC','Outlier'])
acat=pd.read_csv('Astero_Catalogue.txt',skiprows=1,usecols=[0,16],index_col=False,delimiter=';',names=['KIC','Outlier'])
# print(pcat,acat)

acat=acat[acat['Outlier']==0]
pcat=pcat[pcat['Outlier']==0]
# print(pcat,acat)
pcat,acat=np.array(pcat['KIC']),np.array(acat['KIC'])
print(len(pcat),len(acat))
# exit()
afile1=np.loadtxt('twothirds_astero/astero_final_sample_1.txt',dtype=str,delimiter=' ')#,usecols=[0])
afile2=np.loadtxt('twothirds_astero/astero_final_sample_2.txt',dtype=str,delimiter=' ')#,usecols=[0])
afile3=np.loadtxt('twothirds_astero/astero_final_sample_3.txt',dtype=str,delimiter=' ')#,usecols=[0])
afile4=np.loadtxt('twothirds_astero/astero_final_sample_4.txt',dtype=str,delimiter=' ')#,usecols=[0])

pfile1=np.loadtxt('twothirds_pande/pande_pickle_1.txt',dtype=str,delimiter=' ')#,usecols=[0])
pfile2=np.loadtxt('twothirds_pande/pande_pickle_2.txt',dtype=str,delimiter=' ')#,usecols=[0])
pfile3=np.loadtxt('twothirds_pande/pande_pickle_3.txt',dtype=str,delimiter=' ')#,usecols=[0])

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
	adir='jan2020_astero_sample'
	pdir='jan2020_pande_sample'

	ia=np.load(adir+'/labels_m1.npy')[astero_index]
	ip=np.load(pdir+'/labels_m1.npy')[pande_index]
	return ia,ip


def get_inferred_vals(frac):
	adir='{}_astero/{}_of_data'.format(frac,frac)
	pdir='{}_pande/{}_of_data'.format(frac,frac)

	ia=np.load(adir+'/labels_m1.npy')[astero_index]
	ip=np.load(pdir+'/labels_m1.npy')[pande_index]
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
	bias,rms=returnscatter(infer-true)
	# plt.legend(loc='lower right')
	return bias,rms

af,pf=get_og_ivals()
a2,p2=get_inferred_vals('half')
a3,p3=get_inferred_vals('third')
a4,p4=get_inferred_vals('fourth')
a23,p23=get_inferred_vals('twothirds')

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

xls=['24','32','48','65','97'] #[24,32,48,97]

# PANDE:
pbf,prf=get_plot(pt,pf,'Full',1)
pb23,pr23=get_plot(pt,p23,'Two Thirds',5)
pb2,pr2=get_plot(pt,p2,'Half',2)
pb3,pr3=get_plot(pt,p3,'Third',3)
pb4,pr4=get_plot(pt,p4,'Fourth',4)

# ASTERO
bf,arf=get_plot(at,af,'Full',1)
b23,ar23=get_plot(at,a23,'Two Thirds',5)
b2,ar2=get_plot(at,a2,'Half',2)
b3,ar3=get_plot(at,a3,'Third',3)
b4,ar4=get_plot(at,a4,'Fourth',4)

fig=plt.figure(figsize=(5,10))
xls=['24','32','48','65','97'] #[24,32,48,97]

ax3=plt.subplot(211)
plt.grid(which='major', axis='y', linestyle='--')
plt.bar(xls,[pr4,pr3,pr2,pr23,prf],width=0.7,color='grey',edgecolor='k')
plt.ylabel('RMS [dex]',labelpad=10)
STR='Pande'
t=ax3.text(0.79,0.92,s=STR,color='k',ha='left',va='center',transform = ax3.transAxes)
t.set_bbox(dict(facecolor='white',edgecolor='k'))#, alpha=0.5, edgecolor='red'))
plt.minorticks_on()
ax3.tick_params(axis='x', which='minor',bottom='off')

ax4=plt.subplot(212)
plt.grid(which='major', axis='y', linestyle='--')
plt.bar(xls,[ar4,ar3,ar2,ar23,arf],width=0.7,color='grey',edgecolor='k')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax4.yaxis.offsetText.set_visible(False)
plt.ylabel('RMS [dex]',labelpad=2)#  ')
STR='Asteroseismic'
t=ax4.text(0.57,0.92,s=STR,color='k',ha='left',va='center',transform = ax4.transAxes)
t.set_bbox(dict(facecolor='white',edgecolor='k'))#, alpha=0.5, edgecolor='red'))
plt.minorticks_on()
ax4.tick_params(axis='x', which='minor',bottom='off')
ax4.tick_params(axis='x', which='major',top='on',direction='inout',length=6)

ax4.set_xlabel('Time-series length [days]')#,labelpad=10)
fig.tight_layout()

print(pr3/prf,pr4/prf)
plt.show(False)
# plt.savefig('timeseries_paper_plot.pdf',dpi=100)
exit()