import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import fnmatch
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol
from astropy.io import fits
import glob, re
import time as TIME
from astropy.io import ascii
import matplotlib.gridspec as gridspec
from astropy.stats import mad_std
from statistics import mode
# subroutine to perform rough sigma clipping
 

# Get Kps for all stars:
# whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/SC_sayeed_relation.txt',skiprows=1,delimiter=' ')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
kps      =list(df['kic_kepmag'])
gaia=ascii.read('DR2PapTable1.txt',delimiter='&')
data=ascii.read('smoothing_relation/width_vs_radius_test1.txt',delimiter= ' ')
fit_radii,fit_width=np.array(data['Radii']),np.array(data['Width'])


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
 
##pltion()
##pltclf()

def getclosest(num,collection):
    '''Given a number and a list, get closest number in the list to number given.'''
    return min(collection,key=lambda x:abs(x-num))

def getkp(file):
    kic=re.search('kplr(.*)-', file).group(1)
    kic=int(kic.lstrip('0'))
    kp=kps[kp_kics.index(kic)]
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

# main program starts here
if __name__ == '__main__':
    # investigate wnoise fraction in power spectra
    d='/Users/maryumsayeed/Desktop/pande/pande_lcs/'
    files=glob.glob(d+'*.fits')[0:1000]
    # files=np.concatenate([files,files])
    start=TIME.time()
    npoints=len(files)
    # npoints=10000
    # factor_of_noise_added=np.logspace(0.001, 4, num=npoints)
    # factor_of_noise_added=factor_of_noise_added/1e6
    wnoise_level=np.zeros(npoints)
    power_above=np.zeros(npoints)
    
    for i in range(0,len(files)):
        data=fits.open(files[i])
        head=data[0].data
        dat=data[1].data
        time=dat['TIME']
        qual=dat['SAP_QUALITY']

        # only keep data with good quality flags
        good=np.where(qual == 0)[0]
        time=time[good]
        
        # Check Duty Cycle:
        ndays=time[-1]-time[0]
        nmins=ndays*24.*60.
        expected_points=nmins/30.
        observed_points=len(time)
        if observed_points < expected_points*0.5:
            # nstars_below_duty_cycle+=1
            # kics_below_duty_cycle.append(kicid)
            # print(kicid,'below')
            continue

        # UNCOMMENT for long-cadence data!
        if time[-1]-time[0] < 89.: # remove photometry below 89 days from the sample
            # stars_less_than_89_days.append(kicid)
            continue


        # ti,tf=809.5780163868912,905.9259315179515
        ti,tf=time[0],time[-1]
        frac=1
        np.random.seed(0) 
        flux    = np.random.randn(len(time))*frac
    
        (values,counts) = np.unique(np.diff(time),return_counts=True)
        cadence=values[np.argmax(counts)]
        
        time_in = np.arange(ti,tf,cadence) # timestamps interpolated at observed cadence
        flux_in = np.interp(time_in, time, flux) # interpolated flux

        # plt.figure(figsize=(10,8))
        # plt.subplot(311)     
        # plt.plot(time,flux,lw=1)
        # plt.scatter(time,flux,label='observed star',s=10,c='k')
        # plt.title('observed star')
        # plt.xlim(844,848)

        # plt.subplot(312)
        # plt.plot(time_in,flux_in,lw=1)
        # plt.scatter(time_in,flux_in,s=10,\
        #     label='gap filled @ {} min cadence (aka cadence of test star)'.format(int(cadence*60*24)))
        
        # time_in = np.arange(ti,tf,30./(60.*24.)) # timestamps interpolated at observed cadence
        # flux_in = np.interp(time_in, time, flux) # interpolated flux
        # plt.plot(time_in,flux_in,lw=1)
        # plt.scatter(time_in,flux_in,s=5,label='gap filled @ 30 min cadence')
        # plt.title('observed star with gap filled')
        # plt.xlim(844,848)
        # plt.legend()
        
        # plt.subplot(313)
        # time_exp = np.arange(ti,tf,cadence) # discrete points at some min cadence
        # flux_exp = np.random.randn(len(time_exp))*frac
        # time_in,flux_in=time_exp,flux_exp

        # plt.plot(time_exp,flux_exp,lw=1)
        # plt.scatter(time_exp,flux_exp,label='discrete time stamps (test star cadence)',s=10)
        
        # time_exp = np.arange(ti,tf,30./(60.*24.)) # discrete points at 30 min cadence
        # flux_exp = np.random.randn(len(time_exp))*frac
        # time_in,flux_in=time_exp,flux_exp        

        # plt.plot(time_exp,flux_exp,lw=1)
        # plt.scatter(time_exp,flux_exp,label='discrete time stamps @ 30 min cadence',s=10)
        # plt.title('simulated star with ')
        # plt.xlim(844,848)
        # plt.tight_layout()
        # plt.legend()
        # plt.show()
        # exit()

        # now let's calculate the fourier transform. the nyquist frequency is:
        nyq=0.5/(30./60./24.)
        fres=1./90./0.0864

        freq  = np.arange(0.01, 24., 0.01) # long-cadence  critically sampled 
        #amp  = LombScargle(new_time,randomflux).power(freq)
        # freq1, amp1 = LombScargle(time,randomflux).autopower(method='fast',samples_per_peak=10,maximum_frequency=nyq)

        amp_in = LombScargle(time_in,flux_in).power(freq)

        # unit conversions
        freq = 1000.*freq/86.4
        bin = freq[1]-freq[0]
        amp_in = 2.*amp_in*np.var(flux_in*1e6)/(np.sum(amp_in)*bin)

        # White noise correction:
        amp=amp_in
        amp_wn=np.zeros(len(amp))
        
        # calculate average white noise between 270-277 uHz:
        idx=np.where((freq>270) & (freq<277))[0]
        wnoise=np.mean(amp[idx])
        power_more_than_wnoise=0
        for p in range(0,len(amp)):
            a=amp[p]
            if a-wnoise > 0.:
                amp_wn[p]=a-wnoise
                power_more_than_wnoise+=1
            else:
                amp_wn[p]=a
        
        fres_cd=0.01
        fres_mhz=fres_cd/0.0864
        
        n=np.int(2./fres_mhz)
        gauss_kernel = Gaussian1DKernel(n)
        wnpssm = convolve(amp_wn, gauss_kernel)

        snr=power_more_than_wnoise/len(amp)
        power_above[i]=snr
        wnoise_level[i]=wnoise

        print(i,frac,wnoise,snr)

    ascii.write([[1]*npoints,wnoise_level,power_above],'/Users/maryumsayeed/LLR_updates/Oct19/wnoise_simul_{}.txt'.format(npoints),names=['Factor','Wnoise','Fraction'],overwrite=True)

    # for loop ends here:
    # if 'pande' in d:
    #     sample='pande'
    # else:
    #     sample='astero'
    print('Time taken for {} files:'.format(npoints),TIME.time()-start)
    # save text file with 
    