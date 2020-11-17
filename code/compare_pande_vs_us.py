import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from astropy.stats import mad_std

psdir='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'
# pande_results=np.loadtxt(psdir+'Pande_2018.txt',delimiter=',',skiprows=1,usecols=[8,9],dtype=float)
pande18  =pd.read_csv(psdir+'Pande_2018.txt',delimiter=',',skiprows=1,usecols=[0,8,9],names=['KIC','Logg','Logg_err'])
pande_kics=pande18['KIC']
pande_pred=np.array(pande18['Logg'])

mathur_header=['KIC','loggi','e_loggi','r_loggi','n_loggi','logg','E_logg','e_logg','Mass','E_Mass','e_Mass']
mathur_2017 =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/mathur_2017.txt',delimiter=';',skiprows=54,names=mathur_header)
mathur_2017 =mathur_2017[mathur_2017['n_loggi']=='AST'] #include only asteroseismic measurements
mathur_kics=np.array(mathur_2017['KIC'])
mathur_loggs=np.array(mathur_2017['loggi'])

yu_header   =['KICID','Teff','err','logg','logg_err','Fe/H','err','M_noCorrection','M_nocorr_err','R_noCorrection','err','M_RGB','M_RGB_err','R_RGB','err','M_Clump','M_Clump_err','R_Clump','err','EvoPhase']
yu_2018     =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/rg_yu.txt',delimiter='|',names=yu_header,skiprows=1,index_col=False)#,names=yu_header)
yu_kics=np.array(yu_2018['KICID'])
yu_loggs=np.array(yu_2018['logg'])
pande_true=np.zeros(len(pande_kics))
for i in range(0,len(pande_kics)):
	kic=pande_kics[i]
	# print(type(kic),type(yu_kics[0]))
	if kic in yu_kics:
		idx=np.where(yu_kics==kic)[0]
		tlogg=yu_loggs[idx]
		pande_true[i]=tlogg
	elif kic in mathur_kics:
		idx=np.where(mathur_kics==kic)[0]
		tlogg=mathur_loggs[idx]
		pande_true[i]=tlogg
	else:
		pande_true[i]=-999


idx=np.where(pande_true>0)[0]
pande_true,pande_pred=pande_true[idx],pande_pred[idx]
print(mad_std(pande_pred-pande_true))
print(np.std(pande_pred-pande_true))
# exit()

# print(pande18[0:3])
# print(np.min(pande18[:,0]))
# print(np.max(pande18[:,0]))
# exit()
adir=psdir+'LLR_seismic/'
astero_files1=np.loadtxt(adir+'astero_final_sample_1.txt',usecols=[0],dtype='str')
astero_files2=np.loadtxt(adir+'astero_final_sample_2.txt',usecols=[0],dtype='str')
astero_files3=np.loadtxt(adir+'astero_final_sample_3.txt',usecols=[0],dtype='str')
astero_files4=np.loadtxt(adir+'astero_final_sample_4.txt',usecols=[0],dtype='str')

# Get Inferred Logg:

# astero_infer=np.load(psdir+'jan2020_astero_sample/labels_m1.npy')
# astero_true=np.load(psdir+'jan2020_astero_sample/testlabels.npy')[6135:]
filename='LLR_seismic/Seismic_Sample_v4.csv'
df=pd.read_csv(filename,index_col=False)
df=df[df['Outlier']==0]
astero_true=np.array(df['True_Logg'])
astero_pred=np.array(df['Inferred_Logg'])
astero_kics=np.array(df['KICID'])
print(len(astero_pred),len(astero_true))

# idx_good_stars=np.load(psdir+'jan2020_astero_sample/index_of_good_stars.npy')
# my_files=np.concatenate([astero_files1,astero_files2,astero_files3,astero_files4])#[idx_good_stars]

# exit()
# my_true=astero_true[idx_good_stars]
# my_infer=astero_infer[idx_good_stars]

# print('0',len(my_files))
# print('1',len(my_true))
# print('2',len(my_infer))

# my_kics=[]

# for file in my_files:
# 	kic=re.search('kplr(.*)-', file).group(1)
# 	kic=int(kic.lstrip('0'))
# 	my_kics.append(kic)

common_kics=list(set(astero_kics) & set(pande_kics))
print('stars common in my Astero. sample & Pande paper sample:',len(common_kics))
# exit()
# Grab stars that are common in both our samples from our results:
final_kics =[]
ast_true =[]
ast_infer=[]
pande_infer=[]

for i in range(0,len(astero_kics)):
	kic=int(astero_kics[i])
	if kic in common_kics:
		final_kics.append(astero_kics[i])
		ast_true.append(astero_true[i])
		ast_infer.append(astero_pred[i])
		#kic=pande_results['KIC'].iloc[i]
		pande_idx=np.where(pande_kics==kic)[0]
		pande_infer.append(np.array(pande18)[pande_idx,1][0])

print(len(pande_infer))
def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms

ast_true,ast_infer,pande_infer=np.array(ast_true),np.array(ast_infer),np.array(pande_infer)

# exit()
# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
plt.rcParams['font.size']=15
fs=15
fig=plt.figure(figsize=(6,10))
# plt.subplots_adjust(hspace=None)
ax1=plt.subplot(211)
plt.plot(ast_true,ast_true,c='k',linestyle='dashed')
plt.scatter(astero_true,astero_pred,c='lightcoral',s=10,label='This Work')
plt.scatter(ast_true,ast_infer,c='k',s=10,label='Overlapping')
my_outliers=np.where(abs(ast_true-ast_infer)>0.5)[0]

b,r=returnscatter(ast_infer-ast_true)
below=np.where(ast_true<=3.5)[0]
bias_below,rms_below=returnscatter(ast_infer[below]-ast_true[below])
above=np.where((ast_true>3.5) & (abs(ast_true-ast_infer)<0.5))[0]
bias_above,rms_above=returnscatter(ast_infer[above]-ast_true[above])
b,r=returnscatter(ast_infer-ast_true)
std  =mad_std(ast_infer-ast_true)

plt.text(1.95,4.5,r'$\log g$$< 3.5: \sigma$ = '+str(round(rms_below,2))+' Offset = '+str('{0:.2f}'.format(bias_below)), fontsize=12, ha='left',va='bottom')
plt.text(1.95,4.3,r'$\log g \geq 3.5: \sigma$ = '+str(round(rms_above,2))+' Offset = '+str('{0:.2f}'.format(bias_above)), fontsize=12, ha='left',va='bottom')
plt.text(1.95,4.1,'$\sigma$ = '+str(round(r,2))+r' $\sigma_{\mathrm{mad}}$'+' = {0:.2f}'.format(std)+' Offset = '+str('{0:.2f}'.format(b)), fontsize=12, ha='left',va='bottom')

plt.xlim([1.9,4.7])
plt.ylim([1.9,4.7])
#plt.xlabel('True Asteroseismic Logg [dex]',fontsize=fs)

plt.ylabel('Inferred Logg [dex]',fontsize=fs)
locs,labels=plt.xticks()
ax1.set_xticklabels([]*len(labels))
plt.minorticks_on()
# plt.xticks([])
lgnd=plt.legend(fontsize=12,loc='lower right')
lgnd.legendHandles[0]._sizes = [40]
lgnd.legendHandles[1]._sizes = [40]

ax2=plt.subplot(212)
plt.minorticks_on()
plt.plot(astero_true,astero_true,c='k',linestyle='dashed',label='Asteroseismic Logg')
plt.scatter(ast_true,pande_infer,c='k',s=10)

above=np.where(ast_true>=3.5)[0]
bias_below,rms_below=returnscatter(pande_infer[below]-ast_true[below])
bias_above,rms_above=returnscatter(pande_infer[above]-ast_true[above])
b,r=returnscatter(pande_infer-ast_true)
std  =mad_std(pande_infer-ast_true)

# +str('{0:.2f}'.format(rmsa))
plt.text(1.95,4.5,r'$\log g$$< 3.5: \sigma$ = '+str('{0:.2f}'.format(rms_below))+' Offset = '+str('{0:.2f}'.format(bias_below)), fontsize=12, ha='left',va='bottom')
plt.text(1.95,4.3,r'$\log g$$\geq 3.5: \sigma$ = '+str(round(rms_above,2))+' Offset = '+str('{0:.2f}'.format(bias_above)), fontsize=12, ha='left',va='bottom')
plt.text(1.95,4.1,'$\sigma$ = '+str('{0:.2f}'.format(r))+r' $\sigma_{\mathrm{mad}}$'+' = {0:.2f}'.format(std)+' Offset = '+str('{0:.2f}'.format(b)), fontsize=12, ha='left',va='bottom')
plt.xlim([1.9,4.7])
plt.ylim([1.9,4.7])
plt.xlabel('Asteroseismic Logg [dex]',fontsize=fs)
plt.ylabel('Pande Inferred Logg [dex]',fontsize=fs)
#plt.legend(fontsize=10,loc='lower right')
mjlength=5
mnlength=3

ax2.tick_params(which='both', top=True, left=True, bottom=True,length=mjlength)#,width=1) 
ax2.tick_params(which='minor',axis='y',length=mnlength) 

mjlength=7
mnlength=4
ax2.tick_params(which='major',axis='x',direction='inout',length=mjlength) 
ax2.tick_params(which='minor',axis='x',direction='inout',length=mnlength)

# plt.subplot(133)
# plt.xlim([1.9,4.7])
# plt.ylim([1.9,4.7])
# plt.plot(ast_true,ast_true,c='k',linestyle='dashed',label='Asteroseismic Logg')
# plt.scatter(pande_infer,ast_infer,edgecolor='k',facecolor='none',s=10)
# plt.xlabel('Pande Inferred Logg [dex]',fontsize=fs)
# plt.ylabel('Our Inferred Logg [dex]',fontsize=fs)
# plt.legend(fontsize=10,loc='upper left')

savedir='/Users/maryumsayeed/Desktop/HuberNess/iPoster/'
fig.tight_layout()
plt.subplots_adjust(hspace=0)

# plt.savefig(savedir+'pande2018_vs_us_new.png',dpi=100,bbox_inches='tight')
plt.savefig('pande2018_vs_us_new.png',dpi=100,bbox_inches='tight')


plt.show(False)
