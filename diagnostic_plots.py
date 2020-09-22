#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time, re
import pandas as pd
import csv, math
from matplotlib import rc
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol
import matplotlib.gridspec as gridspec
from astropy.stats import mad_std
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


sample='astero'

if sample=='astero':
	start=6135
if sample=='pande':
	start=0

whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
allkps   =list(df['kic_kepmag'])

def sigclip(x,y,subs,sig):
    keep = np.zeros_like(x)
    start=0
    end=subs
    nsubs=int((len(x)/subs)+1)
    for i in range(0,nsubs):        
        me=np.mean(y[start:end])
        sd=np.std(y[start:end])
        good=np.where((y[start:end] > me-sig*sd) & (y[start:end] < me+sig*sd))[0]
        keep[start:end][good]=1
        start=start+subs
        end=end+subs
    return keep

def getclosest(num,collection):
    '''Given a number and a list, get closest number in the list to number given.'''
    return min(collection,key=lambda x:abs(x-num))

def getkp(file):
    kic=re.search('kplr(.*)-', file).group(1)
    kic=int(kic.lstrip('0'))
    kp=allkps[kp_kics.index(kic)]
    if kp in whitenoise[:,0]:
        idx=np.where(whitenoise[:,0]==kp)[0]
        closestkp=whitenoise[idx,0][0]
        wnoise=whitenoise[idx,1][0]
        #print(closestkp,wnoise)
    else:
        closestkp=getclosest(kp,whitenoise[:,0])
        idx=np.where(whitenoise[:,0]==closestkp)[0]
        wnoise=whitenoise[idx,1][0]
        #print(closestkp,wnoise)
    return wnoise

def getps(file,day):

	data=fits.open(file)
	head=data[0].data
	dat=data[1].data
	time=dat['TIME']
	qual=dat['SAP_QUALITY']
	flux=dat['PDCSAP_FLUX']

	good=np.where(qual == 0)[0]
	time=time[good]
	flux=flux[good]

	ndays=(time[-1]-time[0])/1.
	time_1=time
	flux_1=flux
	
	#third=time[0]+ndays
	#idx=np.where(time<third)[0]
	#time=time[idx]
	#flux=flux[idx]
	#time_1=time
	#flux_1=flux

	# Duty cycle:
	total_obs_time=ndays*24.*60  #mins
	cadence       =30.         #mins
	expected_points=total_obs_time/cadence
	observed_points=len(flux)

	# Only analyze stars with light curve duty cycle > 60%:
	#     if observed_points/expected_points<0.5:
	#         continue

	res =sigclip(time,flux,50,3)
	good=np.where(res == 1)[0]
	time=time[good]
	flux=flux[good]
	time_2=time
	flux_2=flux
	        
	width=day
	boxsize=width/(30./60./24.)
	box_kernel = Box1DKernel(boxsize)
	smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
	flux=flux/(smoothed_flux)
	time_3=time
	flux_3=smoothed_flux

	# Remove data points > 3*sigma:
	std=mad_std(flux,ignore_nan=True)
	med=np.median(flux)

	#idx =np.where(abs(flux-med)<3.*std)[0]
	#time=time[idx]
	#flux=flux[idx]

	# now let's calculate the fourier transform. the nyquist frequency is:
	nyq=1./(30./60./24.)
	fres=1./90./0.0864

	fres_cd=0.001
	fres_mhz=fres_cd/0.0864

	freq = np.arange(0.001, 24., 0.001)

	#pdb.set_trace()
	# FT magic
	#freq, amp = LombScargle(time,flux).autopower(method='fast',samples_per_peak=10,maximum_frequency=nyq)
	
	amp = LombScargle(time,flux).power(freq)

	# unit conversions
	freq = 1000.*freq/86.4
	bin = freq[1]-freq[0]
	amp = 2.*amp*np.var(flux*1e6)/(np.sum(amp)*bin)

	# White noise correction:
	wnoise=getkp(file)
	amp1=np.zeros(len(amp))
	for p in range(0,len(amp)):
	    a=amp[p]
	    if a-wnoise < 0.:
	        amp1[p]=amp[p]
	    if a-wnoise > 0.:
	        amp1[p]=a-wnoise

	# smooth by 2 muHz
	n=np.int(2./fres_mhz)
	n_wnoise=np.int(2./fres_mhz)

	gauss_kernel = Gaussian1DKernel(n) 
	gk_wnoise=Gaussian1DKernel(n_wnoise)

	pssm = convolve(amp, gauss_kernel)
	pssm_wnoise = convolve(amp1, gk_wnoise)    
	timeseries=[time_1,flux_1,time_2,flux_2,time_3,flux_3]
	   
	return timeseries,time,flux,freq,amp1,pssm,pssm_wnoise,wnoise

def gettraindata(text_files,pickle_files):
    trainlabels=[]
    alldata=[]
    allpickledata=[]
    allfiles=[]
    star_count=0
    data_lengths=[]
    for i in range(0,len(text_files)):
        print(i,'getting data from:',pickle_files[i])
        labels=np.loadtxt(text_files[i],delimiter=' ',usecols=[1])
        files=np.loadtxt(text_files[i],delimiter=' ',usecols=[0],dtype=str)
        stars= len(labels)
        data = np.memmap(pickle_files[i],dtype=np.float32,mode='r',shape=(21000,stars,3))
        trainlabels.append(labels)
        traindata=data[:,:,1].transpose()
        alldata.append(traindata)
        allfiles.append(files)
        # data_lengths.append(stars)
        star_count+=stars
    
    print('Concatenating data...')
    s1=time.time()
    alldata=list(alldata[0])+list(alldata[1])+list(alldata[2])+list(alldata[3])+list(alldata[4])+list(alldata[5])+list(alldata[6])
    labels=np.concatenate((trainlabels),axis=0)
    allfiles=np.concatenate((allfiles),axis=0)
    print('     ',time.time()-s1)
    
    total_stars=star_count
    return labels,alldata,total_stars,allfiles

def get_idx(files):
	if sample=='astero':
		fname='Astero_Catalogue.txt'
	if sample=='pande':
		fname='Pande_Catalogue.txt'

	df=pd.read_csv(fname,index_col=False,delimiter=';')
	df=df[df['Outlier']==0]
	kics=np.array(df['KICID'])
	true=np.array(df['True_Logg'])
	pred=np.array(df['Inferred_Logg'])
	rad=np.array(df['Radius'])
	teff=np.array(df['Teff'])

	# Get index of good stars using catalogue KICIDs:
	good_index=[]
	for i in range(len(files)):
		file=files[i]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		if kic in kics: # if kic in catalogue:
			good_index.append(i)

	print(len(kics),len(good_index))
	return good_index

dirr='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/jan2020_{}_sample/'.format(sample)
average=np.load(dirr+'average.npy')
end=len(average)
testlabels=np.load(dirr+'testlabels.npy')[start:]
print(len(testlabels))
labels_m1 =np.load(dirr+'labels_m1.npy')
labels_m2 =np.load(dirr+'labels_m2.npy')
spectra_m1=np.load(dirr+'spectra_m1.npy')
chi2_vals =np.load(dirr+'min_chi2.npy')

print('Size of saved data:',len(testlabels),len(average),len(labels_m1),len(labels_m2),len(spectra_m1))
dp='jan2020_pande_sample/'
da='jan2020_astero_sample/'
train_file_names =[dp+'pande_pickle_1',dp+'pande_pickle_2',dp+'pande_pickle_3',da+'astero_final_sample_1',da+'astero_final_sample_2',da+'astero_final_sample_3',da+'astero_final_sample_4']
train_file_pickle=[i+'_memmap.pickle' for i in train_file_names]
train_file_txt   =[i+'.txt' for i in train_file_names]

print('Getting training data...')
all_labels,all_data,total_stars,all_files=gettraindata(train_file_txt,train_file_pickle)
all_labels,all_data,all_files=all_labels[start:start+end],all_data[start:start+end],all_files[start:start+end]

keep=get_idx(all_files)


true,labelm1,models,alldata,allfiles=testlabels,labels_m1,spectra_m1,all_data,all_files

print('Loading in data...')
alldata=np.array(alldata)[keep]
models=models[keep]
savedir='/Users/maryumsayeed/LLR_updates/Aug3/stars_with_high_radii/'
check_kics=pd.read_csv(savedir+'kics_with_high_radii.txt',names=['KIC'],skiprows=1)
check_kics=np.array(check_kics['KIC'])


if sample=='astero':
    fname='Astero_Catalogue.txt'
if sample=='pande':
    fname='Pande_Catalogue.txt'

df=pd.read_csv(fname,index_col=False,delimiter=';')
#df=df[df['Outlier']==0]

raw_kics=[]
raw_kics_idx=[]
for i in range(0,len(keep)):
    file=allfiles[keep][i][0:-3]
    kic=file.split('/')[-1].split('-')[0].split('kplr')[-1]
    kic=int(kic.lstrip('0'))
    if kic in check_kics:
        raw_kics.append(kic)
        raw_kics_idx.append(i)
        
        
print(len(raw_kics))
print(len(raw_kics_idx))
print(len(keep))


all_teffs=np.array(df['Teff'])
all_rads=np.array(df['Radius'])
all_lums=all_rads**2.*(all_teffs/5777.)**4.


wnoise_values=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
wnoise_kics=np.array(wnoise_values[:,0])
wnoise_values

for star in raw_kics_idx[0:200]:
    file=allfiles[keep][star][0:-3]
    timeseries,time,flux,f,amp,pssm,pssm_wnoise,wn=getps(file,1)
    time_1,flux_1,time_2,flux_2,time_3,flux_3=timeseries
    kic=file.split('/')[-1].split('-')[0].split('kplr')[-1]
    kic=int(kic.lstrip('0'))
    print(star,kic)

    idx=np.where(df['KICID']==kic)[0]
    t =np.array(df['Teff'])[idx][0]
    r =np.array(df['Radius'])[idx][0]
    l =r**2.*(t/5777.)**4.
    kp=np.array(df['Kp'])[idx][0]
    snr=np.array(df['SNR'])[idx][0]
    snr='{0:.2f}'.format(snr)
    test =10.**alldata[star,:] #compare log(PSD) values
    model=10.**models[star] #compare log(PSD) values

    plt.figure(figsize=(15,10))
    fs=10
    gs  = gridspec.GridSpec(6, 5)
    ax1 = plt.subplot(gs[0:2, 0:3]) #spans 2 rows U>D, 3 columns L>R
    ax1.plot(time_1,flux_1,linewidth=0.5,c='k')
    ax1.set_title('Raw Lightcurve',fontsize=fs)
    ax1.set_xlabel('Time (Days)',fontsize=fs)
    ax1.set_ylabel('Relative Flux',fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    ax2 = plt.subplot(gs[2:4, 0:3]) #spans 2 rows U>D, 3 columns L>R
    ax2.plot(time,flux,linewidth=0.5,c='k')
    wns='{0:.2f}'.format(wn)
    plt.title('KICID: {} Rad: {} Kp: {}'.format(str(kic),round(r,2),round(kp,2)))
    ax2.set_xlabel('Time (Days)',fontsize=fs)
    ax2.set_ylabel('Relative Flux',fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    ax3 = plt.subplot(gs[4:6, 0:3])
    ax3.loglog(f,amp,c='k',linewidth=0.5,label='Quicklook')
    ax3.loglog(f,pssm_wnoise,c='cyan',label='Quicklook')
    ax3.axhline(wn,c='r')
    ax3.loglog(f[864:864+21000],model,c='r',label='model')
    ax3.set_xlim([4,280])
    ax3.set_ylim([0.01,1e6])
    ax3.set_xlabel('Frequency ($\mu$Hz)',fontsize=fs)
    ax3.set_ylabel(r'PSD (ppm$^2$/$\mu$Hz)',fontsize=fs)
    ax3.set_title('SNR: {} WN: {}'.format(snr,wns))
    a=round(true[keep][star],2)
    b=round(labelm1[keep][star],2)
    
    ax4 = plt.subplot(gs[0:2, 3:5])
    ax4.scatter(all_teffs,all_lums,edgecolor='grey',facecolor='grey',s=10)
    ax4.scatter(all_teffs[star],all_lums[star],c='r',s=50)
    ax4.set_xlim(np.min(all_teffs)-100,np.max(all_teffs)+100)
    plt.gca().invert_xaxis()
    plt.yscale('log')
    ax4.set_xlabel('Effective Temperature [K]')
    ax4.set_ylabel('Luminosity [solar]')
    ax4.set_title('Teff: {} Lum: {}'.format(t,round(l,2)))
    
    ax5 = plt.subplot(gs[2:4, 3:5])
    ax5.plot(true[keep],true[keep],c='k',linewidth=1,zorder=0)
    ax5.scatter(true[keep],labelm1[keep],edgecolor='grey',facecolor='grey',s=10)
    ax5.scatter(true[keep][star],labelm1[keep][star],c='r',s=50)
    ax5.set_title('True: {} Pred.: {}'.format(a,b))#,fontsize=fs)
    ax5.set_xlabel('True Logg [dex]')
    ax5.set_ylabel('Inferred Logg [dex]')

    ax6 = plt.subplot(gs[4:6, 3:5])
    diff=true[keep]-labelm1[keep]
    snrs=np.array(df['SNR'])
    ax6.scatter(snrs,diff,edgecolor='grey',facecolor='grey',s=10)
    ax6.scatter(snrs[idx],diff[idx],c='r',s=50)
    d='{0:.2f}'.format(diff[idx][0])
    s='{0:.2f}'.format(snrs[idx][0])
    ax6.set_title('$\Delta \log g$: {} SNR: {}'.format(d,s))
    ax6.set_xlim(0.25,1.05)
    ax6.set_xlabel('SNR')
    ax6.set_ylabel('True - Inferred')
    
    plt.tight_layout()
    #plt.savefig(savedir+'{}/{}.png'.format(sample,kic),dpi=100,bbox_inches='tight')
    plt.show(True)
    exit()
    #plt.clf()


exit()
from fpdf import FPDF
import glob
pdf = FPDF('L', 'mm', 'A4')
imagelist=glob.glob(savedir+'{}/*.png'.format(sample))[0:2]
# imagelist is the list with all image filenames
for image in imagelist:
    pdf.add_page()
    pdf.image(image)
pdf.output(savedir+"{}/{}_all.pdf".format(sample,sample), "F")





