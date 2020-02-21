import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
psdir='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'
pande_results=np.loadtxt(psdir+'pande_results_table.txt',delimiter=',',skiprows=1,usecols=[8,9],dtype=float)
pande_kics   =np.loadtxt(psdir+'pande_results_table.txt',delimiter=',',skiprows=1,usecols=[0],dtype=int)

astero_files1=np.loadtxt(psdir+'astero_final_sample_1.txt',usecols=[0],dtype='str')
astero_files2=np.loadtxt(psdir+'astero_final_sample_2.txt',usecols=[0],dtype='str')
astero_files3=np.loadtxt(psdir+'astero_final_sample_3.txt',usecols=[0],dtype='str')
astero_files4=np.loadtxt(psdir+'astero_final_sample_4.txt',usecols=[0],dtype='str')

# Get Inferred Logg:
astero_infer=np.load(psdir+'jan2020_astero_sample/labels_m2.npy')
astero_true=np.load(psdir+'jan2020_astero_sample/testlabels.npy')[6135:]
print(len(astero_infer),len(astero_true))

idx_good_stars=np.load(psdir+'jan2020_astero_sample/index_of_good_stars.npy')
my_files=np.concatenate([astero_files1,astero_files2,astero_files3,astero_files4])[idx_good_stars]
my_true=astero_true[idx_good_stars]
my_infer=astero_infer[idx_good_stars]

print('0',len(my_files))
print('1',len(my_true))
print('2',len(my_infer))

my_kics=[]

for file in my_files:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	my_kics.append(kic)

common_kics=list(set(my_kics) & set(pande_kics))
print('stars common in my Astero. sample & Pande paper sample:',len(common_kics))

# Grab stars that are common in both our samples from our results:
final_kics =[]
ast_true =[]
LLR_infer=[]
pande_infer=[]

for i in range(0,len(my_files)):
	file=my_files[i]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic in common_kics:
		final_kics.append(my_kics[i])
		ast_true.append(my_true[i])
		LLR_infer.append(my_infer[i])
		#kic=pande_results['KIC'].iloc[i]
		pande_idx=np.where(pande_kics==kic)[0]
		pande_infer.append(pande_results[pande_idx,0][0])

		
def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms

ast_true,LLR_infer,pande_infer=np.array(ast_true),np.array(LLR_infer),np.array(pande_infer)

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['font.size']=15
fs=15
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.plot(ast_true,ast_true,c='k',linestyle='dashed')
plt.scatter(my_true,my_infer,c='lightcoral',alpha=0.2,s=10,label='Our Sample')
plt.scatter(ast_true,LLR_infer,edgecolor='k',facecolor='none',s=10,label='Overlapping')
my_outliers=np.where(abs(ast_true-LLR_infer)>0.5)[0]


b,r=returnscatter(LLR_infer-ast_true)
below=np.where(ast_true<=3.5)[0]
bias_below,rms_below=returnscatter(ast_true[below]-LLR_infer[below])
above=np.where((ast_true>3.5) & (abs(ast_true-LLR_infer)<0.5))[0]
bias_above,rms_above=returnscatter(ast_true[above]-LLR_infer[above])

plt.text(2.,4.5,r'$< 3.5$: RMS = '+str(round(rms_below,2))+' Bias = '+str(round(bias_below,2)), fontsize=10, ha='left',va='bottom')
plt.text(2.,4.3,r'$> 3.5$: RMS = '+str(round(rms_above,2))+' Bias = '+str(round(bias_above,2)), fontsize=10, ha='left',va='bottom')
plt.xlim([1.9,4.7])
plt.ylim([1.9,4.7])
plt.xlabel('True Asteroseismic Logg (dex)',fontsize=fs)
plt.ylabel('Our Inferred Logg (dex)',fontsize=fs)
plt.legend(fontsize=7,loc='lower right')

plt.subplot(132)
plt.plot(my_true,my_true,c='k',linestyle='dashed',label='Asteroseismic Logg')
plt.scatter(ast_true,pande_infer,edgecolor='k',facecolor='none',s=10)

above=np.where(ast_true>3.5)[0]
bias_below,rms_below=returnscatter(ast_true[below]-pande_infer[below])
bias_above,rms_above=returnscatter(ast_true[above]-pande_infer[above])

plt.text(2.,4.5,r'$< 3.5$: RMS = '+str(round(rms_below,2))+' Bias = '+str(round(bias_below,2)), fontsize=10, ha='left',va='bottom')
plt.text(2.,4.3,r'$> 3.5$: RMS = '+str(round(rms_above,2))+' Bias = '+str(round(bias_above,2)), fontsize=10, ha='left',va='bottom')
b,r=returnscatter(pande_infer-ast_true)
plt.xlim([1.9,4.7])
plt.ylim([1.9,4.7])
plt.xlabel('True Asteroseismic Logg (dex)',fontsize=fs)
plt.ylabel('Pande Inferred Logg (dex)',fontsize=fs)
plt.legend(fontsize=10,loc='lower right')

plt.subplot(133)
plt.xlim([1.9,4.7])
plt.ylim([1.9,4.7])
plt.plot(ast_true,ast_true,c='k',linestyle='dashed',label='Asteroseismic Logg')
plt.scatter(pande_infer,LLR_infer,edgecolor='k',facecolor='none',s=10)
plt.xlabel('Pande Inferred Logg (dex)',fontsize=fs)
plt.ylabel('Our Inferred Logg (dex)',fontsize=fs)
plt.legend(fontsize=10,loc='upper left')

savedir='/Users/maryumsayeed/Desktop/HuberNess/iPoster/'
plt.tight_layout()
plt.savefig(savedir+'pande2018_vs_us_new.png')#,dpi=50)


plt.show()
