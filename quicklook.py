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
from astropy.io import ascii
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
 
plt.ion()
plt.clf()

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
 
    # d='/Users/maryumsayeed/Desktop/pande/pande_lcs/'
    d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/large_train_sample/'
    # d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/'
    # d='/Users/maryumsayeed/Downloads/addtoast/'
    files=glob.glob(d+'*.fits')
    # files=files[0:10]
    print('# of files',len(files))
    kics=np.zeros(len(files),dtype='int')
    teffs=np.zeros(len(files))
    lums=np.zeros(len(files))
    
    gaia=ascii.read('DR2PapTable1.txt',delimiter='&')
    '''files=glob.glob('data/*fits')
    params=ascii.read('syd_SCLC_teffKIC_lum_rad_mass.dat')
    kps=ascii.read('kicids_kp.txt')
    
    kics=np.zeros(len(files),dtype='int')
    teffs=np.zeros(len(files))
    lums=np.zeros(len(files))
    rads=np.zeros(len(files))
    mass=np.zeros(len(files))
    logg=np.zeros(len(files))
    kpmags=np.zeros(len(files))'''
     
    #test=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/inspect.txt',usecols=0,dtype='str')
    #teststars=[i[0:-3] for i in test]
    #print('# of stars to check:',len(teststars))

    # investigate wnoise fraction in power spectra
    more=np.zeros(len(files))
    less=np.zeros(len(files))


    for i in range(0,len(files)):
        f=files[i]
        kicid=int(files[i].split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
        try:
            # print(i,kicid,f,'h')
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
            
            '''
            um=np.where(gaia['KIC'] == kicid)[0]
            if (len(um) == 0.):
                continue
            kics[i]=kicid
            teffs[i]=gaia['teff'][um[0]]
            rad=gaia['rad'][um[0]]
            lums[i]=rad**2*(teffs[i]/5777.)**4
            '''
             
            um=np.where(gaia['KIC'] == kicid)[0]
            if (len(um) == 0.):
                continue
            kics[i]=kicid
            teffs[i]=gaia['teff'][um[0]]
            rad=gaia['rad'][um[0]]
            lums[i]=rad**2.*(teffs[i]/5777.)**4.
            #rads[i]=params['rad'][um[0]]
            #mass[i]=params['mass'][um[0]]
            #logg[i]=np.log10(10**4.44*mass[i]/rads[i]**2)
             
            #um=np.where(kps['Kepler_ID'] == kicid)[0]
            #kpmags[i]=kps['kepmag'][um]
             
            # print('   ',teffs[i],lums[i])
             
            #if (logg[i] < 4.):
            #    continue

            if (teffs[i] == 0):
                continue
             
            #continue

            # only keep data with good quality flags
            good=np.where(qual == 0)[0]
            time=time[good]
            flux=flux[good]

            # === IF CHANGING LENGTH OF LIGHT CURVE, LOOK @ NEXT 4 LINES:
            # Use first third of sample:
            # third=np.arange(0,int(len(time)*(1./3.)),1)

            # time=time[third]
            # flux=flux[third]
            # === END

            # plot the light curve
            # plt.subplot(3,1,1)
            # plt.plot(time,flux)
            # plt.xlabel('Time (Days)')
            # plt.ylabel('Flux (counts)')

            # sigma-clip outliers from the light curve and overplot it
            res=sigclip(time,flux,50,3)
            good=np.where(res == 1)[0]
            time=time[good]
            flux=flux[good]
            

            # plt.plot(time,flux)

            # next, run a filter through the data to remove long-periodic (low frequency) variations
             
            width=1.
            boxsize=width/(30./60./24.)
            box_kernel = Box1DKernel(boxsize)
            smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
            # overplot this smoothed version, and then divide the light curve through it
            # plt.plot(time,smoothed_flux)

            flux=flux/(smoothed_flux)
             
            #flux=flux+np.random.randn(len(flux))*0.01


            # plot the filtered light curve
            # plt.subplot(3,1,2)
            # plt.plot(time,flux)
            # plt.xlabel('Time (Days)')
            # plt.ylabel('Relative flux')

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
            
            #print(len(time),len(flux),len(freq))
            # unit conversions
            freq = 1000.*freq/86.4
            bin = freq[1]-freq[0]
            amp = 2.*amp*np.var(flux*1e6)/(np.sum(amp)*bin)
            
            # White noise correction:
            wnoise=getkp(files[i])
            amp_wn=np.zeros(len(amp))
            power_less_than_wnoise=0
            power_more_than_wnoise=0
            for p in range(0,len(amp)):
                a=amp[p]
                if a-wnoise < 0.:
                    amp_wn[p]=amp[p]
                    power_less_than_wnoise+=1
                if a-wnoise > 0.:
                    amp_wn[p]=a-wnoise
                    power_more_than_wnoise+=1

            print(i,kicid,'more',power_more_than_wnoise/len(amp),'less',power_less_than_wnoise/len(amp))
            more[i]=power_more_than_wnoise/len(amp)
            less[i]=power_less_than_wnoise/len(amp)

            # smooth by 2 muHz
            n=np.int(2./fres_mhz)
             
            gauss_kernel = Gaussian1DKernel(n)
            pssm = convolve(amp, gauss_kernel)
            wnpssm = convolve(amp_wn, gauss_kernel)

            um=np.where(freq > 10.)[0]
            n=files[i].split('/')
            
            # Save frequency & power spectrum arrays:
            # newname=files[i].replace('pande_lcs','pande_lcs_third')
            # newname=files[i].replace('large_train_sample','large_train_sample_third')
            #ascii.write([freq[um],wnpssm[um]],files[i]+'.ps',names=['freq','power'],overwrite=True)
            # ascii.write([freq[um],wnpssm[um]],newname+'.ps',names=['freq','power'],overwrite=True)
            #exit()
            # try:
            #     b = ascii.read(f+'.ps')
            # except error as e:
            #     continue    
            # psfreq = b['freq'] 
            # psflux = b['power']

            # plot the power spectrum
            total=freq[(freq>10.) & (freq<30.)] 
            satisfies=(pssm[(freq>10.) & (freq<30.) & (pssm>100.)])
            percent=len(satisfies)/len(total)*100.
            # plt.subplot(3,1,3)
            # plt.loglog(freq,amp,c='grey')
            # plt.loglog(freq,amp_wn,c='green')
            # plt.loglog(freq,pssm,c='k')
            # plt.loglog(freq,wnpssm,c='r',linestyle='dashed')
            #plt.loglog(psfreq,psflux,c='cyan',linestyle='dashed',label='saved')

            # plt.xlabel('Frequency ($\mu$Hz)')
            # plt.ylabel('Power Density')
            # plt.xlim([10.,300])
            # plt.ylim([0.01,1e6])
            # plt.title('teff='+str(teffs[i])+' lum='+str(lums[i])+' kicid:'+str(kicid))
            # plt.text(12,10**6.,s=str(round(percent,2)))
            # plt.legend()
            # plt.tight_layout()
            #plt.draw()
            # plt.show(False)
            # exit()
            #input(':')
            #pdb.set_trace()
            um=np.where(teffs > 0.)[0]
        except Exception as e:
            print(e)
            print('here')
            continue
            # exit()
        #ascii.write([kics[um],teffs[um],lums[um]],d+'labels_0.5d_1muHz.txt',names=['kic','teff','lum'],overwrite=True)
        
    # save text file with 
    # ascii.write([kics,more,less],'testing_wnoise_frac.txt',names=['KICID','more','less'],overwrite=True)