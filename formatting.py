import numpy as np
import pandas as pd
import glob,os,csv,re,math
import shutil, time
from astropy.io import ascii
import matplotlib.pyplot as plt

"""
Used for finding stars in Pande+2018 and Yu+2018 for Gaia and asteroseismic stars, respectively.
It finds the stars in both catalogues and cross-references them against other catalogues to remove 
any non-granulation stars. Finally, it saves a text file with location of .ps file and star logg.
"""

# Load all data files:
psdir    ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'
hrdir    ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/'
pande_dir='/Users/maryumsayeed/Desktop/pande/pande_lcs/'
ast_dir  ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/large_train_sample/'
pande_lcs     =glob.glob(pande_dir+'*.ps')
ast_lcs       =glob.glob(ast_dir+'*.ps')
print('# of Pande .ps files:',len(glob.glob(pande_dir+'*.ps')))
print('# of Pande .fits files:',len(glob.glob(pande_dir+'*.fits')))
print('# of Astero. .ps files:',len(glob.glob(ast_dir+'*.ps')))
print('# of Astero. .fits files:',len(glob.glob(ast_dir+'*.fits')))


gaia     =ascii.read('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/DR2PapTable1.txt',delimiter='&')
gaia     =gaia[gaia['binaryFlag']==0] #remove any binaries

# Get Kps for all stars:
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
kp_all   =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])

kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])

# Load asteroseismic samples:
astero1=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/labels_full.txt',delimiter=' ',skiprows=1,usecols=[0,2])
astero1_kics=[int(i) for i in list(astero1[:,0])]
print('# of stars in labels_full:',len(astero1))
astero2=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/rg_yu.txt',delimiter='|',skiprows=1,usecols=[0,3])
astero2_kics=[int(i) for i in list(astero2[:,0])] 
print('# of stars in Yu+(2018):',len(astero2))
astero=list(set(astero2_kics+astero1_kics))
astero=[int(i) for i in astero]
print('# total astero. stars:',len(astero))

# Load pande sample:
pande   =pd.read_csv('/Users/maryumsayeed/Desktop/pande/pande_granulation.txt')#,skiprows=1,usecols=[0],dtype=int,delimiter=',')
pande_kics=list(pande['#KIC'])
print('# of stars in Pande+(2018):',len(pande))

# If star in both sample, treat it as asteroseismic star to increase ast. sample.
# If star only in Pande sample, keep it there. 
# If star only in ast. sample, keep it there. 
pande_stars0=(set(pande_kics) - set(astero))

# Get catalogues of non-granulation stars:
not_dir='not_granulation_star/'
dscutis    =np.loadtxt(not_dir+'murphy_dscuti.txt',usecols=[0,-1],delimiter=',',skiprows=1,dtype=int)
idx=np.where(dscutis[:,1]==1)[0] #stars that have dSct flag
dscutis    =dscutis[idx][:,0]
binaries   =np.loadtxt(not_dir+'ebinary.txt',usecols=[0],dtype=int,delimiter=',')
exoplanets =pd.read_csv(not_dir+'koi_planethosts.csv',skiprows=53,usecols=['kepid','koi_disposition','koi_pdisposition'])
#exoplanets=exoplanets[exoplanets['koi_pdisposition']!='FALSE POSITIVE'] # Remove false positive exoplanets:
exoplanets =[int(i) for i in list(exoplanets['kepid'])] 
superflares=np.loadtxt(hrdir+'superflares_shibayama2013.txt',skiprows=33,usecols=[0],dtype=int)
superflares=[int(i) for i in list(superflares)]
flares     =list(np.loadtxt(not_dir+'flares_davenport2016.txt',usecols=[0],skiprows=1,delimiter=',',dtype=int))
rotating   =list(np.loadtxt(not_dir+'mcquillan_rotation.txt',usecols=[0],skiprows=1,delimiter=',',dtype=int))
clas       =ascii.read(not_dir+'debosscher2011.dat')
gdor       =clas[(clas['V1'] == 'GDOR') | (clas['V1'] == 'SPB')]
gdor       =[int(i) for i in list(gdor['KIC'])]
dscutis2   =clas[clas['V1'] == 'DSCUT']
dscutis2   =[int(i) for i in list(dscutis2['KIC'])]
rrlyr      =pd.read_csv(not_dir+'rrlyr.txt')
rrlyr      =[int(i) for i in list(rrlyr['kic'])]

# print(type(astero1_kics[0]))
# print(3646127 in astero1_kics)
# print(astero1_kics[0:10])
# print(np.where(np.asarray(astero1_kics) == 3646127))
# exit()

# Remove non-granulation stars:
start=time.time()
print(len(pande_stars0))
# pande_stars=list(set(pande_stars0)-set(binaries)-set(exoplanets['kepid'])-set(flares['#kicnum'])-set(rotating['KID']) \
# -set(superflares)-set(dscutis)-set(dscutis2['KIC'])-set(gdor['KIC'])-set(rrlyr['kic']))
pande_stars=list(set(pande_stars0)-set(binaries)-set(exoplanets)-set(flares)-set(rotating) \
-set(superflares)-set(dscutis)-set(dscutis2)-set(gdor)-set(rrlyr))

print('total time taken:',time.time()-start)

astero_stars=list(set(astero)-set(binaries)-set(exoplanets)-set(flares)-set(rotating) \
-set(superflares)-set(dscutis)-set(dscutis2)-set(gdor)-set(rrlyr))

# Only get stars in Gaia catalogue (Berger+2018):
pande_stars = list((set(pande_stars) & set(gaia['KIC'])))
astero_stars = list((set(astero_stars) & set(gaia['KIC'])))
print('# of Pande stars:',len(pande_stars))
print('# of asteroseismic stars:',len(astero_stars))

# Check if all Pande stars have a light curve downloaded : 
pande_kics_downloaded=[]
for file in pande_lcs:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	pande_kics_downloaded.append(kic)

pande_not_downloaded=list(set(pande_stars) - set(pande_kics_downloaded))
# Only use Pande stars we have downloaded:
pande_stars = list(set(set(pande_stars)-set(pande_not_downloaded)))
print('# Pande not downloaded',len(pande_not_downloaded))
print(len(pande_stars))

# Check if all astero. stars have a light curve downloaded : 
ast_kics_downloaded=[]
for file in ast_lcs:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	ast_kics_downloaded.append(kic)

ast_not_downloaded=list(set(astero_stars) - set(ast_kics_downloaded))
astero_stars = list(set(set(astero_stars)-set(ast_not_downloaded)))
print('# astero not downloaded',len(ast_not_downloaded))
print(len(astero_stars))

fn='/Users/maryumsayeed/Downloads/'
#np.savetxt(fn+'pande_not_downloaded.txt',pande_not_downloaded,fmt='%s')
np.savetxt(fn+'ast_not_downloaded.txt',ast_not_downloaded,fmt='%s')

# Find logg for Pande:
pande_final_sample=[]
pande_loggs=[]
check_dups=[]
for file in pande_lcs:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic in pande_stars:
		row=kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		logg=row['iso_logg'].item()
		if math.isnan(logg) is False: # check to see there are no nan loggs
			logg_pos_err=row['iso_logg_err1']
			logg_neg_err=row['iso_logg_err2']	
			pande_final_sample.append([file,logg])
			pande_loggs.append(logg)
	else:
		continue
plt.hist(pande_loggs)
plt.show()

# Double check all these stars are in Pande: 
kic_not_in_pande=[]
for i in pande_final_sample:
	file=i[0]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic not in pande_kics:
		kic_not_in_pande.append(kic)

print(len(kic_not_in_pande))
print('# Pande stars to save:',len(pande_final_sample))

np.savetxt(psdir+'pande_final_sample_full.txt',pande_final_sample,fmt='%s')
diff=3000
np.savetxt(psdir+'pande_pickle_1.txt',pande_final_sample[0:3000],fmt='%s')
np.savetxt(psdir+'pande_pickle_2.txt',pande_final_sample[3000:6000],fmt='%s')
np.savetxt(psdir+'pande_pickle_3.txt',pande_final_sample[6000:],fmt='%s')

# Find logg for Astero.:
astero_final_sample=[]
astero_loggs=[]
for file in ast_lcs:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic in astero_stars:
		if kic in astero1:
			idx=np.where(np.asarray(astero1_kics)==kic)[0]
			logg=astero1[idx,1][0]
		elif kic in astero2:
			idx=np.where(np.asarray(astero2_kics)==kic)[0]
			logg=astero2[idx,1][0]
		#logg=row['logg'].item()
		# logg_pos_err=row['err']
		# logg_neg_err=row['err']
		astero_final_sample.append([file,logg])
		astero_loggs.append(logg)
	else:
		continue

plt.hist(astero_loggs)
plt.show()

# Double check all these stars are in Pande: 
kic_not_in_astero=[]
for i in astero_final_sample:
	file=i[0]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic not in astero:
		kic_not_in_astero.append(kic)

print(len(kic_not_in_astero))
print('# Astero. stars to save:',len(astero_final_sample))


#exit()
diff=4000
np.savetxt(psdir+'astero_final_sample_full.txt',astero_final_sample,fmt='%s')
np.savetxt(psdir+'astero_final_sample_1.txt',astero_final_sample[0:4000],fmt='%s')
np.savetxt(psdir+'astero_final_sample_2.txt',astero_final_sample[4000:4000+diff],fmt='%s')
np.savetxt(psdir+'astero_final_sample_3.txt',astero_final_sample[8000:8000+diff],fmt='%s')
np.savetxt(psdir+'astero_final_sample_4.txt',astero_final_sample[12000:12000+diff],fmt='%s')

exit()



