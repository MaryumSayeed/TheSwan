import pickle, glob, re
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.stats import mad_std
from sklearn.metrics import r2_score
import matplotlib.image as mpimg
from scipy.stats import chisquare
import matplotlib.gridspec as gridspec


plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['font.size']=15
plt.rcParams['mathtext.default']='regular'
plt.rcParams['xtick.major.pad']='3'
plt.rcParams['ytick.major.pad']='4'
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('font', size=15)                  # controls default text sizes
plt.rc('axes', titlesize=15)             # fontsize of the axes title
plt.rc('axes', labelsize=15)             # fontsize of the x and y labels
plt.rc('axes', labelpad=3)               # padding of x and y labels
plt.rc('xtick', labelsize=15)            # fontsize of the tick labels
plt.rc('ytick', labelsize=15)            # fontsize of the tick labels
plt.rc('axes', linewidth=1)  
plt.rc('legend', fontsize=15)


dd='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'


def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias,rms


def get_plot(sample,n):
	teststars=np.loadtxt(dd+'{}/Cannon_test_stars.txt'.format(sample),usecols=[0],dtype='str')

	pf=open(dd+'{}/testmetaall_cannon_logg.pickle'.format(sample),'rb') 
	testmetaall,truemetaall=pickle.load(pf)
	pf.close() 
	true=truemetaall[:,0]
	infer=testmetaall[:,0]
	std=mad_std(infer-true)
	r2=r2_score(true,infer)
	bias,rms=returnscatter(true-infer)

	print('Max/Min',np.max(true),np.min(true))
	print('# stars')

	if n ==1:
		a1,a2,x1,x2=0,3,0,4
		b1,b2=3,4
	if n==2:
		a1,a2,x1,x2=0,3,4,8
		b1,b2=3,4

	gs = gridspec.GridSpec(ncols=8, nrows=4,hspace=0)
	ax1 = plt.subplot(gs[a1:a2, x1:x2])
	idx=np.where(abs(true-infer)<0.05)[0]
	print(len(true),len(infer),len(teststars))
	kics=[]
	for file in teststars:
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		kics.append(kic)
	kics=np.array(kics)
	# print(list(kics[idx]))
	# exit()
	ax1.plot(true,true,c='k',linestyle='dashed')
	ax1.scatter(true,infer,facecolors='grey', edgecolors='k',s=10)
	# ax1.scatter(true[idx],infer[idx],c='r',s=10)
	ax1.minorticks_off()
	locs, labels = plt.yticks()
	newlabels=[2.0,2.5,3.0,3.5,4.0,4.5,5.0]
	plt.yticks(newlabels, newlabels)
	ax1.set_xticklabels([]*len(newlabels))
	plt.minorticks_on()
	plt.yticks(newlabels,newlabels)
	plt.xticks([])		
	ax1.set_xlim([1.9,4.8])
	ax1.set_ylim([1.9,4.8])
	
	ax2 = plt.subplot(gs[b1:b2, x1:x2])
	ax2.scatter(true,true-infer,edgecolor='k',facecolor='grey',s=10)
	ax2.axhline(0,c='k',linestyle='dashed')
	ax2.set_xlabel('True Logg [dex]')
	
	plt.minorticks_on()
	ax2.tick_params(which='major',axis='y',pad=5)
	ax2.set_xlim([1.9,4.75])
	ax2.set_ylim([-0.7,0.7])

	if n==1:
		ax1.set_ylabel('Inferred Logg [dex]')
		ax2.set_ylabel('$\Delta$ Logg [dex]')
	if n==2:
		ax1.set_yticklabels([])
		ax2.set_yticklabels([])
	
	str1=r'$\sigma$ = '+str('{0:.2f}'.format(rms))
	str2=r'$\sigma_{\mathrm{mad}}$ = '+str('{0:.2f}'.format(std))
	str3='Offset = '+str('{0:.2f}'.format(bias))
	STR=str1+'\n'+str2+'\n'+str3
	t=ax1.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	diff=infer-true
	a01=np.where(true<3.5)[0]
	a02=np.where(true>=3.5)[0]
	a1=np.where((true<3.5) & (diff>0))[0]
	a2=np.where((true>=3.5) & (diff>0))[0]
	#print('pred>true','logg < 3.5',len(a1)/len(a01))
	#print('pred>true','logg > 3.5',len(a2)/len(a02))

	below=np.where(true<3.5)[0]
	bias_below,rms_below=returnscatter(true[below]-infer[below])
	std_below=mad_std(true[below]-infer[below])
	above=np.where(true>=3.5)[0]
	bias_above,rms_above=returnscatter(true[above]-infer[above])
	std_above=mad_std(true[above]-infer[above])
	ax1.text(2.9,2.2,r'$\log g < 3.5: \sigma$ = '+str(round(rms_below,2))+' Offset = '+str(round(bias_below,2)), fontsize=12, ha='left',va='bottom')
	ax1.text(2.9,2.,r'$\log g \geq 3.5: \sigma$ = '+str(round(rms_above,2))+' Offset = '+str('{0:.2f}'.format(bias_above)), fontsize=12, ha='left',va='bottom')

	mjlength=5
	mnlength=3
	ax1.tick_params(which='both', # Options for both major and minor ticks
	                top=False, # turn off top ticks
	                left=True, # turn off left ticks
	                right=False,  # turn off right ticks
	                bottom=True,# turn off bottom ticks)
	                length=mjlength,
	                width=1)
	ax1.tick_params(which='minor',length=mnlength) 
	ax2.tick_params(which='both', top=True, left=True, bottom=True,length=mjlength,width=1) 
	ax2.tick_params(which='minor',axis='y',length=mnlength) 

	mjlength=7
	mnlength=4
	ax2.tick_params(which='major',axis='x',direction='inout',length=mjlength) 
	ax2.tick_params(which='minor',axis='x',direction='inout',length=mnlength)
	ax2.tick_params(which='major',axis='y',direction='inout',length=mjlength) 
	ax2.tick_params(which='minor',axis='y',direction='inout',length=mnlength)
	
	ax1.tick_params(which='major',axis='y',direction='inout',length=mjlength) 
	ax1.tick_params(which='minor',axis='y',direction='inout',length=mnlength)
	print('\n')

plt.figure(figsize=(12,8))
get_plot('cannon_vs_LLR/one_label',1)
# get_plot('dwarfs_only',2)
plt.tight_layout()
plt.subplots_adjust(wspace=0)
# plt.show(True)
plt.savefig(dd+'1.png',dpi=100,bbox_inches='tight')
	