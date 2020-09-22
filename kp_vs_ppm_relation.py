#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, re
from astropy.io import ascii
from scipy.stats import chisquare
import os

plt.rcParams['lines.markersize']=1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

kpfile   =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv',                     index_col=False,names=['KIC','Kp'],skiprows=1)
wnoise_pande=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',',names=['Kp','wnoise','scatter'])
pande_kp,pande_wn=np.array(wnoise_pande['Kp']),np.array(wnoise_pande['wnoise'])


# ===FIT PANDE RELATION===:

x,y=pande_kp,pande_wn
ylog=np.log10(pande_wn)

degrees=np.arange(1,20,1)

for d in degrees:
    z = np.polyfit(x, ylog, d)
    func_pande = np.poly1d(z)
    c,p=chisquare(f_obs=y,f_exp=10.**func_pande(x),ddof=len(func_pande))
    print(d,c,p)

z = np.polyfit(x, ylog, 14)
func_pande = np.poly1d(z)

plt.figure(figsize=(8,6))
plt.scatter(pande_kp,pande_wn)
plt.plot(x,10.**func_pande(x),c='r')
plt.tight_layout()
plt.yscale('log')
plt.xlabel('Kp')
plt.ylabel('White noise power')
plt.savefig('0.png',bbox_inches='tight')
plt.savefig('/Users/maryumsayeed/LLR_updates/July27/1.png')
plt.show(False)
plt.close()


# ===GET WHITE NOISE VS. TIME DOMAIN NOISE RELATION===:

file=pd.read_csv('wnoise_simulated/time_series_wnoise_simul_10000.txt',delimiter=' ',names=['Factor','Wnoise'],skiprows=1)
TD_noise,PS_noise=np.array(file['Factor']),np.array(file['Wnoise'])

plt.figure(figsize=(8,6))
plt.scatter(TD_noise*1e6,PS_noise)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Time domain noise')
plt.ylabel('White noise power')
plt.tight_layout()
plt.savefig('1.png',bbox_inches='tight')
plt.savefig('/Users/maryumsayeed/LLR_updates/July27/2.png')
plt.show(False)
plt.close()

sim_kp=np.linspace(np.min(pande_kp),np.max(pande_kp),3000)


ppms=[]
wns=[]

for i in range(0,len(sim_kp)):
    k=sim_kp[i]
    
    nearest_kp=find_nearest(pande_kp, k)
    idx=np.where(pande_kp==nearest_kp)[0]
    wn=pande_wn[idx][0]    #white noise value from Pande interpolated between points
    wn=10.**func_pande(k)  #wn from Pande using 14th order fit:

    nearest_wn=find_nearest(PS_noise, wn)
    idx=np.where(PS_noise==nearest_wn)[0]
    
    ppm=TD_noise[idx][0]
    ppms.append(ppm*1e6)
    wns.append(wn)
    #print(i,round(k,3),round(wn,3),round(nearest_wn,3),ppm*1e6)
    

# === FIT PPM VS. KP RELATION === :

x,y  =sim_kp,ppms
ylog =np.log10(ppms)
z    =np.polyfit(x, ylog, 14)
func = np.poly1d(z)

for d in degrees:
    z = np.polyfit(x, ylog, d)
    func_TD = np.poly1d(z)
    c,p=chisquare(f_obs=y,f_exp=10.**func_TD(x),ddof=len(func_TD))
    print(d,c,p)

plt.figure(figsize=(8,6))
plt.scatter(sim_kp,ppms)
plt.plot(x, 10.**func(x),c='r')
plt.yscale('log')
plt.xlabel('Kp')
plt.ylabel('Time domain noise (ppm)')
plt.tight_layout()

print('Fit parameters:')
print('   ',func)
plt.show(False)
plt.savefig('3.png',bbox_inches='tight')
# plt.savefig('3.png')

exit()
ascii.write([sim_kp,ppms],'wnoise_simulated/ppm_vs_kp.txt',names=['Kp','TimeDomainNoise'],overwrite=True)

