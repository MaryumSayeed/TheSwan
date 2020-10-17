import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import kplr
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

# subroutine to perform rough sigma clipping
 

# Get Kps for all stars:
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
kps      =list(df['kic_kepmag'])

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
 
    d='/Users/maryumsayeed/Desktop/pande/pande_lcs/'
    # d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/large_train_sample/'
    # d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/'
    # d='/Users/maryumsayeed/Downloads/addtoast/'
    files=glob.glob(d+'*.fits')
    # files=files[0:10]
    print('# of files',len(files))
    kics=np.zeros(len(files),dtype='int')
    teffs=np.zeros(len(files))
    lums=np.zeros(len(files))
    
    gaia=ascii.read('DR2PapTable1.txt',delimiter='&')
    #pltfigure(figsize=(15,10))

    # investigate wnoise fraction in power spectra
    nfiles=100#len(files)
    more=np.zeros(nfiles)
    less=np.zeros(nfiles)
    wnoise_full_avg=np.zeros(nfiles)
    new_wnoise=np.zeros(nfiles)
    old_wnoise=np.zeros(nfiles)

    start=TIME.time()
    for i in range(0,1):
        f=files[i]
        kicid=int(files[i].split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
        if kicid>0:#==9093289:#4833377:#10675847:
            try:
                print(i,kicid)
                # exit()
                data=fits.open(files[i])
                head=data[0].data
                dat=data[1].data
                time=dat['TIME']
                qual=dat['SAP_QUALITY']
                flux=dat['PDCSAP_FLUX']
                ndays=time[-1]-time[0]
                nmins=ndays*24.*60.
                expected_points=nmins/30.
                observed_points=len(time)
                if observed_points < expected_points*0.5:
                    print(kicid,'below')
                    continue

                good=np.where(qual == 0)[0]
                time=time[good]
                flux=flux[good]
                res=sigclip(time,flux,50,3)
                good=np.where(res == 1)[0]
                time=time[good]
                flux=flux[good]

                width=1.
                boxsize=width/(30./60./24.)
                box_kernel = Box1DKernel(boxsize)
                smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')

                flux=flux/(smoothed_flux)
                
                boxsize    =int(width/(30./60./24.))
                box_kernel = Box1DKernel(boxsize)

                if boxsize % 2 == 0:
                    smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
                else:
                    smoothed_flux = savgol(flux,int(boxsize),1,mode='mirror')
                flux =flux/smoothed_flux
                
                std =mad_std(flux,ignore_nan=True)
                med =np.median(flux)
                idx =np.where(abs(flux-med)<5.*std)[0]
                time=time[idx]
                flux=flux[idx]

                # factor_of_noise_added=np.linspace(0,0.01,100+1)
                npoints=10000
                factor_of_noise_added=np.logspace(0.001, 4, num=npoints)
                factor_of_noise_added=factor_of_noise_added/1e6
                ys=[]
                wnoise_level=[]
                xx=1
                power_above=np.zeros(len(factor_of_noise_added))
                
                ti,tf=809.5780163868912,905.9259315179515
                time_in = np.arange(ti,tf,30./(60.*24.))

                for k in range(len(factor_of_noise_added)):
                    frac=factor_of_noise_added[k]
                    flux_in=np.random.randn(len(time_in))*frac
                    flux_in2=np.random.randn(len(flux))*frac
                    #new_time=np.linspace(time[0],time[-1],len(time))
                    
                    # now let's calculate the fourier transform. the nyquist frequency is:
                    nyq=0.5/(30./60./24.)
                    fres=1./90./0.0864
 
                    freq  = np.arange(0.01, 24., 0.01) # long-cadence  critically sampled 
                    #amp  = LombScargle(new_time,randomflux).power(freq)
                    # freq1, amp1 = LombScargle(time,randomflux).autopower(method='fast',samples_per_peak=10,maximum_frequency=nyq)

                    # UNCOMMENT TO NOT FILL GAPS IN LIGHT CURVE:
                    # time_interp,flux_interp=time,randomflux 

                    # UNCOMMENT TO FILL GAPS IN LIGHT CURVE:
                    # print(time[0],time[-1],len(time))
                    # print(time_in[0],time_in[-1],len(time_in))
                    # exit()
                    amp_in = LombScargle(time_in,flux_in).power(freq)

                    time_in2 = np.arange(time[0],time[-1],30./(60.*24.))
                    flux_in2 = np.interp(time_in2, time, flux_in2)
                    amp_in2 = LombScargle(time_in2,flux_in2).power(freq)

                    # unit conversions
                    freq = 1000.*freq/86.4
                    bin = freq[1]-freq[0]
                    amp_in = 2.*amp_in*np.var(flux_in*1e6)/(np.sum(amp_in)*bin)
                    amp_in2 = 2.*amp_in2*np.var(flux_in2*1e6)/(np.sum(amp_in2)*bin)

                    # plt.figure(figsize=(15,6))
                    # plt.subplot(211)
                    # plt.plot(time_in,flux_in,linewidth=1)
                    # plt.plot(time_in2,flux_in2,linewidth=1)
                    # plt.subplot(212)
                    # plt.loglog(freq,amp_in,linewidth=1,alpha=0.7)
                    # plt.loglog(freq,amp_in2,linewidth=1,alpha=0.7)
                    # plt.xlim([1.,600])
                    # plt.ylim(1e-7,1e7)
                    # plt.tight_layout()
                    # plt.show(False)

                    # White noise correction:
                    # wnoise=getkp(files[i])
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
                    power_above[k]=snr
                    wnoise_level.append(wnoise)

                    print(k,frac,wnoise,snr)

                    '''if power_more_than_wnoise/len(amp) > 65:
                        fig=plt.figure(figsize=(18,8))
                        gs  = gridspec.GridSpec(4, 9)  
                        ax1 = plt.subplot(gs[0:2, 0:6]) #spans 2 rows U>D, 3 columns L>R
                        plt.title(frac*1e6)
                        plt.plot(time,randomflux,linewidth=1)    
                        plt.plot(time_interp,flux_interp,linewidth=1)    
                        
                        ax2 = plt.subplot(gs[2:4, 0:6])
                        plt.loglog(freq1,amp1,linewidth=1)
                        plt.loglog(freq_interp,amp_interp,linewidth=1)
                        plt.axvspan(270,277,color='g',alpha=0.5)
                        #plt.loglog(freq,wnpssm,c='k',linestyle='dashed',label='WN corrected')
                        plt.axhline(wnoise,c='r',linestyle='--')
                        plt.xlabel('Frequency ($\mu$Hz)')
                        plt.ylabel('Power Density')
                        plt.xlim([1.,300])
                        ymin,ymax=1e-7,1e7
                        plt.ylim([ymin,ymax])

                        ax3 = plt.subplot(gs[2:4, 6:9])
                        plt.hist(amp_wn,bins=100)
                        plt.axvline(wnoise,c='r',linestyle='dashed')
                        plt.xlabel('PSD')
                        plt.ylabel('Counts')
                        str1='Fraction='+'{0:.2f}'.format(power_more_than_wnoise/len(amp))
                        str2='Amp$_{max}=$'+'{0:.2f}'.format(np.max(amp_wn))
                        str3='Amp$_{min}=$'+'{0:.2f}'.format(np.min(amp_wn))
                        idx_above=np.where((amp-wnoise)>0)[0]
                        idx_below=np.where((amp-wnoise)<0)[0]
                        str4='Above={}'.format(len(idx_above))
                        str5='Below={}'.format(len(idx_below))
                        str6='Total={}'.format(len(amp))
                        STR=str1+'\n'+str2+'\n'+str3+'\n'+str4+'\n'+str5+'\n'+str6
                        t=ax3.text(0.97,0.9,s=STR,color='k',ha='right',va='top',transform = ax3.transAxes,fontsize=15)
                        t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
                        plt.tight_layout()
                        #plt.show()

                        plt.savefig('/Users/maryumsayeed/LLR_updates/Oct12/{0:.3f}'.format(power_more_than_wnoise/len(amp))+'_{}.png'.format(k),bbox_inches='tight')'''
                        
                ascii.write([factor_of_noise_added,wnoise_level,power_above],'/Users/maryumsayeed/LLR_updates/Oct12/wnoise_simul_{}.txt'.format(npoints),names=['Factor','Wnoise','Fraction'],overwrite=True)
            except Exception as e:
                print(e)
                print('here')
                # exit()
                continue
            #ascii.write([kics[um],teffs[um],lums[um]],d+'labels_0.5d_1muHz.txt',names=['kic','teff','lum'],overwrite=True)
        # 
    # for loop ends here:
    # if 'pande' in d:
    #     sample='pande'
    # else:
    #     sample='astero'
    print('Time taken for {} files:'.format(nfiles),TIME.time()-start)
    # save text file with 
    