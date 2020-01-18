import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
psdir='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'
pande_results=pd.read_csv(psdir+'pande_results_table.txt',delimiter=',')
pande_kics   =list(pande_results['KIC'])

astero_files1=np.loadtxt(psdir+'train_stars_ast_1.txt',usecols=[0],dtype='str')
astero_files2=np.loadtxt(psdir+'train_stars_ast_2.txt',usecols=[0],dtype='str')
#pande_files  =np.loadtxt(psdir+'train_stars_pande.txt',usecols=[0],dtype='str')

# Get Inferred Logg:
astero_infer=np.load(psdir+'astero/labels_m2.npy')
astero_true=np.load(psdir+'astero/testlabels.npy')
idx_good_stars=np.load(psdir+'astero/index_final_good_stars.npy')
supposed_true=np.load(psdir+'astero/supposed_true.npy')
supposed_pred=np.load(psdir+'astero/supposed_pred.npy')
supposed_files=np.load(psdir+'astero/supposed_files.npy')
#pande_infer=np.load(psdir+'pande/labels_m1.npy')
#pande_true=np.load(psdir+'pande/testlabels.npy')[len(astero_true):]
my_files=np.concatenate([astero_files1,astero_files2])[idx_good_stars]
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

common_kics=list(set(my_kics) & set(pande_results['KIC']))
print(len(common_kics))
# Grab stars that are common in both our samples from our results:
my_kics1 =[]
my_true1 =[]
my_infer1=[]

for i in range(0,len(my_files)):
	file=my_files[i]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic in common_kics:
		my_kics1.append(my_kics[i])
		my_true1.append(my_true[i])
		my_infer1.append(my_infer[i])


# Grab stars that are common in both our samples from Pande's table:
pande_results_logg    =list(pande_results[' logg'])
pande_results_logg_err=list(pande_results[' logg err '])
idx_star_from_pande_results=[]

pande_infer1=[]
for i in range(0,len(pande_results)):
	kic=pande_results['KIC'].iloc[i]
	if kic in common_kics:
		pande_infer1.append(pande_results_logg[i])

def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms

my_true1,my_infer1,pande_infer1=np.array(my_true1),np.array(my_infer1),np.array(pande_infer1)

# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
plt.rcParams['font.size']=15
fs=15
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.plot(supposed_true,supposed_true,c='k',linestyle='dashed')
plt.scatter(supposed_true,supposed_pred,c='lightcoral',alpha=0.2,s=10,label='Our sample')
#plt.plot(my_true1,my_true1,c='k',linestyle='dashed')
plt.scatter(my_true1,my_infer1,edgecolor='k',facecolor='none',s=10,label='Overlapping in Astero. & Pande')
my_outliers=np.where(abs(my_true1-my_infer1)>0.5)[0]
plt.scatter(my_true1[my_outliers],my_infer1[my_outliers],c='r',s=20,label='My Outliers',zorder=10)
b,r=returnscatter(my_infer1-my_true1)
below=np.where(my_true1<=3.5)[0]
bias_below,rms_below=returnscatter(my_true1[below]-my_infer1[below])
above=np.where((my_true1>3.5) & (abs(my_true1-my_infer1)<0.5))[0]
bias_above,rms_above=returnscatter(my_true1[above]-my_infer1[above])
plt.title('note: $>3.5$ stats not including outliers',fontsize=10)
plt.text(2.1,4.1,r'$< 3.5$: RMS = '+str(round(rms_below,2))+' Bias = '+str(round(bias_below,2)), fontsize=10, ha='left',va='bottom')
plt.text(2.1,3.9,r'$> 3.5$: RMS = '+str(round(rms_above,2))+' Bias = '+str(round(bias_above,2)), fontsize=10, ha='left',va='bottom')
# plt.text(1,4.3,'RMS: {}'.format(round(r,3)),fontsize=fs,va='bottom')
# plt.text(1,3.9,'Bias: {}'.format(round(b,3)),fontsize=fs)
plt.xlim([2,4.7])
plt.ylim([2,4.7])
plt.xlabel('True Asteroseismic Logg (dex)',fontsize=fs)
plt.ylabel('Our Inferred Logg (dex)',fontsize=fs)
plt.legend(fontsize=7,loc='best')

plt.subplot(132)
plt.plot(my_true1,my_true1,c='k',linestyle='dashed',label='Asteroseismic Logg')
plt.scatter(my_true1,pande_infer1,edgecolor='k',facecolor='none',s=10)
plt.scatter(my_true1[my_outliers],pande_infer1[my_outliers],c='r',s=20,label='My Outliers',zorder=10)
above=np.where(my_true1>3.5)[0]
bias_below,rms_below=returnscatter(my_true1[below]-pande_infer1[below])
bias_above,rms_above=returnscatter(my_true1[above]-pande_infer1[above])
plt.text(2.1,4.5,r'$< 3.5$: RMS = '+str(round(rms_below,2))+' Bias = '+str(round(bias_below,2)), fontsize=10, ha='left',va='bottom')
plt.text(2.1,4.2,r'$> 3.5$: RMS = '+str(round(rms_above,2))+' Bias = '+str(round(bias_above,2)), fontsize=10, ha='left',va='bottom')
b,r=returnscatter(pande_infer1-my_true1)
plt.xlim([2,4.7])
plt.ylim([2,4.7])
# plt.text(2,4.3,'RMS: {}'.format(round(r,3)),fontsize=fs,va='bottom')
# plt.text(2,4.0,'Bias: {}'.format(round(b,3)),fontsize=fs)
plt.xlabel('True Asteroseismic Logg (dex)',fontsize=fs)
plt.ylabel('Pande Inferred Logg (dex)',fontsize=fs)
plt.legend(fontsize=10,loc='lower right')

plt.subplot(133)
plt.xlim([2,4.7])
plt.ylim([2,4.7])
plt.plot(my_true1,my_true1,c='k',linestyle='dashed',label='Asteroseismic Logg')
plt.scatter(pande_infer1,my_infer1,edgecolor='k',facecolor='none',s=10)
plt.xlabel('Pande Inferred Logg (dex)',fontsize=fs)
plt.ylabel('Our Inferred Logg (dex)',fontsize=fs)
plt.legend(fontsize=10,loc='upper left')

savedir='/Users/maryumsayeed/Desktop/HuberNess/iPoster/'
plt.tight_layout()
# plt.savefig(savedir+'pande2018_vs_us_new.png')#,dpi=50)


plt.show()
