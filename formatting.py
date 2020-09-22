#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob,os,csv,re,math
import shutil, time
from astropy.io import ascii
import matplotlib.pyplot as plt


# Load all data files:
psdir    ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'
hrdir    ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/'

# Original directories:
pande_dir='/Users/maryumsayeed/Desktop/pande/pande_lcs/'
ast_dir  ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/large_train_sample/'
# Directories when testing sections of lightcurves:
# pande_dir='/Users/maryumsayeed/Desktop/pande/pande_lcs_third/'
# ast_dir  ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/large_train_sample_third/'

pande_lcs     =glob.glob(pande_dir+'*.fits')
ast_lcs       =glob.glob(ast_dir+'*.fits')
print('# of Pande .ps files:',len(glob.glob(pande_dir+'*.ps')))
print('# of Pande .fits files:',len(glob.glob(pande_dir+'*.fits')))
print('# of Astero. .ps files:',len(glob.glob(ast_dir+'*.ps')))
print('# of Astero. .fits files:',len(glob.glob(ast_dir+'*.fits')))

# Load Berger+ stellar properties catalogues:
gaia     =ascii.read('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/DR2PapTable1.txt',delimiter='&')
gaia     =gaia[gaia['binaryFlag']==0] #remove any binaries

kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])

# Get Kps for all stars:
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
kp_all   =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])


# # Load Asteroseismic Samples:

# Don't want to include any Mathur sample:
mathur_header=['KIC','loggi','e_loggi','r_loggi','n_loggi','logg','E_logg','e_logg','Mass','E_Mass','e_Mass']
mathur_2017 =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/mathur_2017.txt',delimiter=';',skiprows=54,names=mathur_header)
mathur_2017 =mathur_2017[mathur_2017['n_loggi']=='AST'] #include only asteroseismic measurements

yu_header=['KICID','Teff','err','logg','logg_err','Fe/H','err','M_noCorrection','M_nocorr_err','R_noCorrection','err','M_RGB','M_RGB_err','R_RGB','err','M_Clump','M_Clump_err','R_Clump','err','EvoPhase']
yu_2018     =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/rg_yu.txt',delimiter='|',names=yu_header,skiprows=1,index_col=False)#,names=yu_header)
#chaplin_2014=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/Chaplin_2014.txt',skiprows=47,delimiter='\t',names=chaplin_header)
#huber_2013  =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/Huber_2013.txt',delimiter='\t',skiprows=37,names=['KIC','Mass','Mass_err'])

mathur_kics=np.array(mathur_2017['KIC'])
yu_kics=np.array(yu_2018['KICID'])
#chaplin_kics=np.array(chaplin_2014['KIC'])
#huber_kics=np.array(huber_2013['KIC'])
print('# of stars in Yu+2018:',len(yu_kics))
print('# of stars in Mathur+17:',len(mathur_kics))

astero_kics=np.concatenate([mathur_kics,yu_kics])
astero_kics=np.array(list(set(astero_kics)))

print('Total seismic stars:',len(astero_kics))


# # Load Pande sample:

pande   =pd.read_csv('/Users/maryumsayeed/Desktop/pande/pande_granulation.txt')#,skiprows=1,usecols=[0],dtype=int,delimiter=',')
pande_kics=list(pande['#KIC'])
print('# of stars in Pande+2018:',len(pande))


# If star in both sample, treat it as asteroseismic star to increase ast. sample.
# If star only in Pande sample, keep it there. 
# If star only in ast. sample, keep it there. 
pande_stars0=(set(pande_kics) - set(astero_kics))
print('# stars only in Pande+ and not astero',len(pande_stars0))
print('# total astero. stars:',len(astero_kics))
print('# stars in both Pande+ and astero catalogues:',len(list(set(pande_kics) & set(astero_kics))))


# # Get catalogues of non-granulation stars:

not_dir='/Users/maryumsayeed/Desktop/HuberNess/mlearning/ACFcannon-master/not_granulation_star/'
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


# # Remove non-granulation stars:


pande_stars=list(set(pande_stars0)-set(binaries)-set(exoplanets)-set(flares)-set(rotating) -set(superflares)-set(dscutis)-set(dscutis2)-set(gdor)-set(rrlyr))

astero_stars=list(set(astero_kics)-set(binaries)-set(exoplanets)-set(flares)-set(rotating) -set(superflares)-set(dscutis)-set(dscutis2)-set(gdor)-set(rrlyr))

print('# of non-granulation stars removed from astero sample:',len(astero_kics)-len(astero_stars))
print('# of non-granulation stars removed from pande sample:',len(pande_stars0)-len(pande_stars))

# Only get stars in Gaia catalogue (Berger+2018):
print('(before cross-referenced with Gaia) # of Pande stars:',len(pande_stars))
print('(before cross-referenced with Gaia) # of Astero. stars:',len(astero_stars))

pande_stars = list((set(pande_stars) & set(gaia['KIC'])))
astero_stars = list((set(astero_stars) & set(gaia['KIC'])))
print('final # of Pande stars:',len(pande_stars))
print('final # of asteroseismic stars:',len(astero_stars))


# Check if all Pande stars have a light curve downloaded : 
print('\n','=====       PANDE       =====')
pande_kics_downloaded=[]
for file in pande_lcs:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	pande_kics_downloaded.append(kic)

print('These should be the same:')
print('---Stars downloaded:',len(pande_kics_downloaded))
print('---Stars needed:',len(pande_stars))
if len(pande_kics_downloaded) > len(pande_stars):
	print('We have more stars downloaded than we need from Pande+18.')
else:
	print("Don't have all the stars that we need. Download more!")

# Only use Pande stars we have downloaded:
#pande_stars = list(set(set(pande_stars)-set(pande_not_downloaded)))
pande_below_dc=ascii.read(psdir+'LLR_gaia/pande_kics_below_duty_cycle.txt',names=['KICID'])
pande_below_89=ascii.read(psdir+'LLR_gaia/pande_kics_below_89_days.txt',names=['KICID'])
pande_below_dc,pande_below_89=pande_below_dc['KICID'],pande_below_89['KICID']
pande_not_downloaded =[]
pande_stars_downloaded=[]
for kic in pande_stars:
    if kic in pande_kics_downloaded:
        pande_stars_downloaded.append(kic)
    else: 
        pande_not_downloaded.append(kic)

print('Need from Pande+18',len(pande_stars))
print('Downloaded',len(pande_stars_downloaded))
print('Have but removed aka:')
print('---# of Pande stars below 89 days',len(pande_below_89))
print('---# of Pande stars below duty cycle',len(pande_below_dc))
print('Pande not downloaded',len(pande_not_downloaded))
print('Good pande stars',len(pande_stars))

# Check if all astero. stars have a light curve downloaded : 
print('\n','=====       ASTERO.       =====')
ast_kics_downloaded=[]
for file in ast_lcs:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	ast_kics_downloaded.append(kic)

print('These should be the same:')
print('---Stars downloaded:',len(ast_kics_downloaded))
print('---Stars needed:',len(astero_stars))
if len(ast_kics_downloaded) > len(astero_stars):
	print('We have more stars downloaded than we need from astero catalogues.')
else:
	print("Don't have all the stars that we need. Download more!")

astero_below_dc=ascii.read(psdir+'LLR_seismic/astero_kics_below_duty_cycle.txt',names=['KICID'])
astero_below_89=ascii.read(psdir+'LLR_seismic/astero_kics_below_89_days.txt',names=['KICID'])
astero_below_dc,astero_below_89=astero_below_dc['KICID'],astero_below_89['KICID']
astero_not_downloaded =[]
astero_stars_downloaded=[]
for kic in astero_stars:
    if kic in ast_kics_downloaded:
        astero_stars_downloaded.append(kic)
    else: 
        astero_not_downloaded.append(kic)

print('Need from catalogues',len(astero_stars))
print('Downloaded',len(ast_kics_downloaded))
print('Have but removed aka:')
print('---# of astero stars below 89 days',len(astero_below_89))
print('---# of astero stars below duty cycle',len(astero_below_dc))
print('Astero not downloaded',len(astero_not_downloaded))
print('Good astero stars',len(astero_stars))

# In[13]:
# ascii.write([astero_stars],psdir+'astero_stars_we_need.txt',overwrite=True)
# ascii.write([ast_kics_downloaded],psdir+'astero_stars_downloaded.txt',overwrite=True)
# ascii.write([good_astero_stars],psdir+'good_stars_downloaded.txt',overwrite=True)

fn='/Users/maryumsayeed/Downloads/'

# np.savetxt(fn+'pande_not_downloaded.txt',pande_not_downloaded,fmt='%s')
# np.savetxt(fn+'astero_not_downloaded.txt',astero_not_downloaded,fmt='%s')


# # Find logg for Pande:
print('\n','Getting logg for Pande. stars...')
pande_ps=glob.glob(pande_dir+'*.ps')
pande_no_logg=0
pande_final_sample=[]
pande_loggs=[]
check_dups=[]
for file in pande_ps:
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
			pande_no_logg+=1
	else:
		continue

print('Pande w/ no logg:',pande_no_logg)

# Double check all these stars are in Pande: 
kic_not_in_pande=[]
for i in pande_final_sample:
	file=i[0]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic not in pande_kics:
		kic_not_in_pande.append(kic)

print('# stars not in Pande.',len(kic_not_in_pande))
print('# Pande stars to save:',len(pande_final_sample))

diff=2000
# np.savetxt(psdir+'pande_final_sample_full.txt',pande_final_sample,fmt='%s')
# np.savetxt(psdir+'pande_pickle_1.txt',pande_final_sample[0:2000],fmt='%s')
# np.savetxt(psdir+'pande_pickle_2.txt',pande_final_sample[2000:4000],fmt='%s')
# np.savetxt(psdir+'pande_pickle_3.txt',pande_final_sample[4000:],fmt='%s')

# # Find logg for Astero. stars:
print('\n','Getting logg for Astero. stars...')
astero_ps=glob.glob(ast_dir+'*.ps')
files,loggs=[],np.zeros(len(astero_ps))
c1,c2,c3,none=0,0,0,0
for i in range(0,len(astero_ps)):
    file=astero_ps[i]
    kic=re.search('kplr(.*)-', file).group(1)
    kic=int(kic.lstrip('0'))
    if kic in astero_stars:
        if kic in yu_kics:
            row=yu_2018.loc[yu_2018['KICID']==kic]
            logg  =row['logg'].item()
            c1+=1
        elif kic in mathur_kics:
            row =mathur_2017.loc[mathur_2017['KIC']==kic]
            logg  =row['loggi'].item()
            c2+=1
        else:
            none+=1
        loggs[i]=logg
        files.append(file)
#         astero_final_sample.append([file,logg])
#         astero_loggs.append(logg)
    else:
        continue

files,loggs=np.array(files),np.array(loggs).astype(float)
print('Yu+:',c1,'Mathur+',c2,'None',none)

idx=np.where(loggs>0)[0] #aka select valid stars
astero_files,astero_loggs=files[idx],loggs[idx]
astero_final_sample=np.array([astero_files,astero_loggs]).T

# Double check all these stars are in Pande: 
kic_not_in_astero=[]
for i in astero_final_sample:
    file=i[0]
    kic=re.search('kplr(.*)-', file).group(1)
    kic=int(kic.lstrip('0'))
    if kic not in astero_stars:
        kic_not_in_astero.append(kic)

print('# stars not in Astero.',len(kic_not_in_astero))
print('# Astero. stars to save:',len(astero_final_sample))

diff=4000
# np.savetxt(psdir+'astero_final_sample_full.txt',astero_final_sample,fmt='%s')
# np.savetxt(psdir+'astero_final_sample_1.txt',astero_final_sample[0:4000],fmt='%s')
# np.savetxt(psdir+'astero_final_sample_2.txt',astero_final_sample[4000:4000+diff],fmt='%s')
# np.savetxt(psdir+'astero_final_sample_3.txt',astero_final_sample[8000:8000+diff],fmt='%s')
# np.savetxt(psdir+'astero_final_sample_4.txt',astero_final_sample[12000:12000+diff],fmt='%s')

