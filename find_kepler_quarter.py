import numpy as np
import pandas as pd
from astropy.io import fits
import glob,re
import matplotlib.pyplot as plt

pande=pd.read_csv('LLR_gaia/pande_final_sample_full.txt',index_col=False,delimiter=' ',names=['file','logg'])
astero=pd.read_csv('LLR_seismic/astero_final_sample_full.txt',index_col=False,delimiter=' ',names=['file','logg'])
data=pande#astero

# d='/Users/maryumsayeed/Desktop/pande/pande_lcs/'
# d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/large_train_sample/'

# ===== find quarter of data
quarter=[]
files=np.array(data['file'])
for file in files:
	file=file[0:-3]
	kicid=int(file.split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
	hdul = fits.open(file)
	hdr = hdul[0].header
	q=hdr['QUARTER']
	quarter.append(q)

q11=quarter.count(11)
q10=quarter.count(10)
q9=quarter.count(9)
q8=quarter.count(8)
q7=quarter.count(7)

print(q7,q8,q9,q10,q11)
print(len(quarter),np.sum([q7,q8,q9,q10,q11]))
exit()
# not downloaded light curves:
nd7=pd.read_csv('/Users/maryumsayeed/Downloads/astero_not_downloaded/7kepler_wget.sh',index_col=False,delimiter=' ',usecols=[2])#names=['a','b','file'])
nd8=pd.read_csv('/Users/maryumsayeed/Downloads/astero_not_downloaded/8kepler_wget.sh',index_col=False,delimiter=' ',usecols=[2])#,names=['a','b','file'])
nd10=pd.read_csv('/Users/maryumsayeed/Downloads/astero_not_downloaded/10kepler_wget.sh',index_col=False,delimiter=' ',usecols=[2])#,names=['a','b','file'])
nd11=pd.read_csv('/Users/maryumsayeed/Downloads/astero_not_downloaded/11kepler_wget.sh',index_col=False,delimiter=' ',usecols=[2])#,names=['a','b','file'])

nd7,nd8,nd10,nd11=np.array(nd7),np.array(nd8),np.array(nd10),np.array(nd11)
print(type(nd7[0]))
kics=[]

for i in [nd7,nd8,nd10,nd11]:
	print(len(i))
for group in [nd7,nd8,nd10,nd11]:
	for file in group:
		file=file[0]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		kics.append(kic)
kics_unique=list(set(kics))
print(len(kics),len(kics_unique))
