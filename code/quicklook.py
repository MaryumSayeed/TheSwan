import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb, math
import kplr
import fnmatch
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol
from astropy.io import fits
import glob, re
from astropy.io import ascii
from astropy.stats import mad_std
# subroutine to perform rough sigma clipping
 

# Get Kps for all stars:
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
kps      =list(df['kic_kepmag'])

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
 
##plt.ion()
##plt.clf()

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
    # d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/cannon_vs_LLR/cannon_no_wnoise/'
    # d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/large_train_sample_cannon/'
    # d='/Users/maryumsayeed/Downloads/pande_remaining/'
    # d='/Users/maryumsayeed/Downloads/pande_sc/'
    files=glob.glob(d+'*.fits')
    # files=files[0:10]
    print('# of files',len(files))
    kics=np.zeros(len(files),dtype='int')
    rads=np.zeros(len(files))
    teffs=np.zeros(len(files))
    lums=np.zeros(len(files))
    
    gaia=ascii.read('DR2PapTable1.txt',delimiter='&')
    kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])
    
    data=ascii.read('smoothing_relation/width_vs_radius_test1.txt',delimiter= ' ')
    fit_radii,fit_width=np.array(data['Radii']),np.array(data['Width'])

    # check=ascii.read('/Users/maryumsayeed/LLR_updates/Aug17/stars_with_high_snr.txt',names=['KIC'])
    # check=np.array(check).astype(int)
    # check=[11767251, 3539778, 10160035, 11507559, 11876278, 3862456, 8960478, 8453152, 7522019]
    # check=[9071948, 4648211, 7465477, 1572317, 3954440, 8417232, 8359236, 11234317, 7622294, 5445912, 4473030, 8645063, 8144907, 9418101, 3842398, 12110908, 6960601, 6525349, 10199289, 10000279, 9697262, 7384966, 2436676, 4168909, 4633774, 6612496]

    more=np.zeros(len(files))
    nstars_below_duty_cycle=0
    kics_below_duty_cycle=[]
    stars_less_than_89_days=[]
    days=[]
    fbins=[] #3535
    for i in range(0,len(files)):
        f=files[i]
        kicid=int(files[i].split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
        if kicid >0:
            data=fits.open(files[i])
            head=data[0].data
            dat=data[1].data
            time=dat['TIME']
            qual=dat['SAP_QUALITY']
            flux=dat['PDCSAP_FLUX']

            um=np.where(gaia['KIC'] == kicid)[0]
            if (len(um) == 0.):
                continue

            kics[i]=kicid
            try:
                row=kepler_catalogue.loc[kepler_catalogue['KIC']==kicid]
                teff  =row['iso_teff'].item()
                rad  =row['iso_rad'].item()
            except:
                idx=np.where(gaia['KIC']==kicid)[0]
                teff=gaia['teff'][idx][0]
                rad=gaia['rad'][idx][0]
            if math.isnan(rad) is True:
                idx=np.where(gaia['KIC']==kicid)[0]
                teff=gaia['teff'][idx][0]
                rad=gaia['rad'][idx][0]
            
            rads[i]=rad
            teffs[i]=teff
            lums[i]=rad**2.*(teff/5777.)**4. 
            
            if (teffs[i] == 0):
                continue

            # if (rads[i] <50.):
            #     continue
                
            # only keep data with good quality flags
            good=np.where(qual == 0)[0]
            time=time[good]
            flux=flux[good]

            # plot the light curve
            # #plt.ion()
            # #plt.clf()

            #plt.figure(figsize=(9,10))
            #plt.subplot(3,1,1)
            #plt.plot(time,flux)
            #plt.xlabel('Time (Days)')
            #plt.ylabel('Flux (counts)')

            # sigma-clip outliers from the light curve and overplot it
            res=sigclip(time,flux,50,3)
            good=np.where(res == 1)[0]
            time=time[good]
            flux=flux[good]

            if len(time)==0:
                continue
            
            # Check Duty Cycle:
            ndays=time[-1]-time[0]
            nmins=ndays*24.*60.
            expected_points=nmins/30.
            observed_points=len(time)
            if observed_points < expected_points*0.5:
                nstars_below_duty_cycle+=1
                kics_below_duty_cycle.append(kicid)
                print(kicid,'below')
                continue

            # UNCOMMENT for long-cadence data!
            if time[-1]-time[0] < 89.: # remove photometry below 89 days from the sample
                stars_less_than_89_days.append(kicid)
                continue
            
            # === IF CHANGING LENGTH OF LIGHT CURVE, LOOK @ NEXT 5 LINES:
            # Use first cut of sample:
            baseline=65.
            baseline_time=time[0]+baseline
            cut=np.where(time<=baseline_time)[0] #take all times below baseline time
            time=time[cut]
            flux=flux[cut]
            # === END

            
            #plt.plot(time,flux)
            
            ## if rads[i] <= 36.8.: #from width_vs_radius_test2.txt
            closestrad=getclosest(rads[i],fit_radii)
            idx       =np.where(fit_radii==closestrad)[0]
            best_fit_width=fit_width[idx][0]
            width      =best_fit_width
            print(i,kicid,width)
            
            boxsize    =int(width/(30./60./24.))
            box_kernel = Box1DKernel(boxsize)
            if boxsize % 2 == 0:
                smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
            else:
                smoothed_flux = savgol(flux,int(boxsize),1,mode='mirror')
            flux =flux/smoothed_flux
    
            # overplot this smoothed version, and then divide the light curve through it
            #plt.plot(time,smoothed_flux)
            #plt.axvspan(time[1000],time[1000+int(boxsize)],color='g',zorder=1,alpha=0.2)
            #flux=flux+np.random.randn(len(flux))*0.01

            # plot the filtered light curve
            #plt.subplot(3,1,2)

            #plt.plot(time,flux)
            #plt.xlabel('Time (Days)')
            #plt.ylabel('Relative flux')

            # Remove data points > 5*sigma:
            std =mad_std(flux,ignore_nan=True)
            med =np.median(flux)
            idx =np.where(abs(flux-med)<5.*std)[0]
            time=time[idx]
            flux=flux[idx]
            #plt.plot(time,flux)

            # now let's calculate the fourier transform. the nyquist frequency is:
            nyq=0.5/(30./60./24.)
            fres=1./90./0.0864
             
            freq  = np.arange(0.01, 24., 0.01) # long-cadence  critically sampled 
            # freq  = np.arange(0.01, 734.1, 0.01) # long-cadence  critically sampled 
            freq0 = freq

            time_interp = np.arange(time[0],time[-1],30./(60.*24.))
            flux_interp = np.interp(time_interp, time, flux)
            time0,flux0 = time,flux #non-interpolated data
            time,flux   = time_interp,flux_interp

            amp  = LombScargle(time,flux).power(freq)
            amp0 = LombScargle(time0,flux0).power(freq)
            

            # FT magic
            # freq, amp = LombScargle(time,flux).autopower(method='fast',samples_per_peak=10,maximum_frequency=nyq)
            # freq_interp, amp_interp = LombScargle(time_interp,flux_interp).autopower(method='fast',samples_per_peak=1,maximum_frequency=nyq)
            # freq0, amp0             = LombScargle(time0,flux0).autopower(method='fast',samples_per_peak=1,maximum_frequency=nyq)
            
            fbins.append(len(freq))
            days.append(ndays)

            # unit conversions
            freq = 1000.*freq/86.4
            bin = freq[1]-freq[0]
            amp = 2.*amp*np.var(flux*1e6)/(np.sum(amp)*bin)

            freq0 = 1000.*freq0/86.4
            bin0 = freq0[1]-freq0[0]
            amp0 = 2.*amp0*np.var(flux0*1e6)/(np.sum(amp0)*bin0)

            # White noise correction:
            wnoise=getkp(files[i])
            #idx=np.where(freq>270)[0]
            #wnoise=np.mean(amp[idx])
            amp_wn=np.zeros(len(amp))
            amp_wn0=np.zeros(len(amp0))

            power_more_than_wnoise=0
            for p in range(0,len(amp)):
                a=amp[p]
                if a-wnoise < 0.:
                    amp_wn[p]=amp[p]
                if a-wnoise > 0.:
                    amp_wn[p]=a-wnoise
                    power_more_than_wnoise+=1

            for p in range(0,len(amp0)):
                a0=amp0[p]
                if a0-wnoise < 0.:
                    amp_wn0[p]=amp0[p]
                if a0-wnoise > 0.:
                    amp_wn0[p]=a0-wnoise
            
            snr=power_more_than_wnoise/len(amp)
            more[i]=snr

            # smooth by 2 muHz
            fres_cd=0.01  #to check if it's smoothed by 2muHz: print(freq[0],freq[0+n]) difference should be 2
            fres_mhz=fres_cd/0.0864

            n=np.int(2./fres_mhz)

            gauss_kernel = Gaussian1DKernel(n)
            pssm = convolve(amp, gauss_kernel)
            wnpssm = convolve(amp_wn, gauss_kernel)
            wnpssm0 = convolve(amp_wn0, gauss_kernel)

            #Save frequency & power spectrum arrays:
            um=np.where(freq > 10.)[0]
            n =files[i].split('/')
            
            # newname=files[i].replace('large_train_sample','large_train_sample_cut')
            newname=files[i].replace('pande_lcs','pande_lcs_cut')
            ascii.write([freq[um],wnpssm[um]],newname+'.ps',names=['freq','power'],overwrite=True)
            # ascii.write([freq[um],wnpssm[um]],files[i]+'.ps',names=['freq','power'],overwrite=True)
            

            #plot the power spectrum
            #plt.subplot(3,1,3)
            #plt.loglog(freq,amp_wn,c='grey')
            #plt.loglog(freq,wnpssm,c='k')
            #plt.axhline(wnoise,c='r',linestyle='dashed')
            ##plt.loglog(psfreq,psflux,c='cyan',linestyle='dashed',label='saved')

            #plt.xlabel('Frequency ($\mu$Hz)')
            #plt.ylabel('Power Density')
            # #plt.xlim([9,300])
            #plt.ylim([1e-4,1e6])
            #plt.title(' Width={0:.1f} '.format(width)+'KICID: '+str(kicid)+' Teff={}'.format(int(teffs[i]))+' Radius={0:.2f}'.format(rads[i])+' Lum={0:.2f}'.format(lums[i])+' SNR={0:.2f}'.format(snr))
            #plt.tight_layout()
            #plt.show(True)

            # #plt.draw()
            savedir='/Users/maryumsayeed/sig_clip/'
            # #plt.savefig(savedir+'{}_no.png'.format(kicid),dpi=100,bbox_inches='tight')
            
            # input(':')
            # pdb.set_trace()
            # um=np.where(teffs > 0.)[0]
            # except Exception as e:
            #     print('----',e)
            #     continue
        # ascii.write([kics[um],teffs[um],lums[um]],d+'labels_0.5d_1muHz.txt',names=['kic','teff','lum'],overwrite=True)
    print('stars_less_than_89_days',len(stars_less_than_89_days))

    # save text file with 
    if 'pande' in d:
        sample='pande'
    else:
        sample='astero'
    exit()
    ascii.write([kics,more,rads],'{}_wnoise_frac.txt'.format(sample),names=['KICID','Fraction','Radius'],overwrite=True)
    ascii.write([kics_below_duty_cycle],'{}_kics_below_duty_cycle.txt'.format(sample),names=['KICID'],overwrite=True)
    ascii.write([stars_less_than_89_days],'{}_kics_below_89_days.txt'.format(sample),names=['KICID'],overwrite=True)