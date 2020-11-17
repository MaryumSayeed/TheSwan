import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb, math
# import kplr
import fnmatch
import time as TIME
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol
from astropy.io import fits
import glob, re
from astropy.io import ascii
from astropy.stats import mad_std
# subroutine to perform rough sigma clipping
import argparse
# subroutine to perform rough sigma clipping
 
# Create the parser
# my_parser = argparse.ArgumentParser(description='Prepare power spectra data given a text file with\
#                                                 location of .PS files and associated labels.')

# # Add the arguments
# my_parser.add_argument('-file','--file',
#                        metavar='file',
#                        type=str,required=True,
#                        help='FITS file to add noise to.')


# # Execute the parse_args() method
# args = my_parser.parse_args()

# FITSFILE=args.file

# Get Kps for all stars:
# whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
#whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/new_WNvsKp_relation.txt',delimiter=' ')
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/SC_sayeed_relation.txt',skiprows=1,delimiter=' ')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
kpdf       =pd.read_csv(kpfile,index_col=False,names=['KIC','Kp'],skiprows=1)
kp_kics  =list(kpdf['KIC'])
kps      =list(kpdf['Kp'])


ppmkp_file=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/wnoise_simulated/ppm_vs_kp.txt',skiprows=1,delimiter=' ',names=['Kp','PPM'])
ppmkp_kp,ppmkp_ppm=np.array(ppmkp_file['Kp']),np.array(ppmkp_file['PPM'])


gaia=ascii.read('DR2PapTable1.txt',delimiter='&')
kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])

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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def get_psd(FITSFILE,FACTOR):
    f=FITSFILE
    kicid=int(f.split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
    s0=TIME.time()
    data=fits.open(f)
    head=data[0].data
    dat=data[1].data
    time=dat['TIME']
    qual=dat['SAP_QUALITY']
    flux=dat['PDCSAP_FLUX']

    um=np.where(gaia['KIC'] == kicid)[0]
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

    # only keep data with good quality flags
    good=np.where(qual == 0)[0]
    time=time[good]
    flux=flux[good]

    # sigma-clip outliers from the light curve and overplot it
    res=sigclip(time,flux,50,3)
    good=np.where(res == 1)[0]
    time=time[good]
    flux=flux[good]

    # Check Duty Cycle:
    ndays=time[-1]-time[0]
    nmins=ndays*24.*60.
    expected_points=nmins/30.
    observed_points=len(time)
    
    ## if rads[i] <= 36.8.: #from width_vs_radius_test2.txt
    closestrad=getclosest(rad,fit_radii)
    idx       =np.where(fit_radii==closestrad)[0]
    best_fit_width=fit_width[idx][0]
    width      =best_fit_width
    #print(kicid,width)

    boxsize    =int(width/(30./60./24.))
    box_kernel = Box1DKernel(boxsize)

    if boxsize % 2 == 0:
        smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
    else:
        smoothed_flux = savgol(flux,int(boxsize),1,mode='mirror')
    flux =flux/smoothed_flux

    # Get Kp of star:
    row   =kpdf.loc[kpdf['KIC']==int(kicid)]
    kp    =float(row['Kp'].item())

    # Remove data points > 5*sigma:
    std =mad_std(flux,ignore_nan=True)
    med =np.median(flux)
    idx =np.where(abs(flux-med)<5.*std)[0]
    time=time[idx]
    flux=flux[idx]

    # Get expected time domain noise in photometry:
    idx2,nearest_kp=find_nearest(ppmkp_kp,kp) # find nearest Kp in our PPM vs. Kp file given Kp of our stars
    TD_noise=ppmkp_ppm[idx2]                   # find time domain noise corresponding to that Kp
    TD_noise=TD_noise/1e6
    np.random.seed(0) 
    factor=FACTOR*TD_noise
    flux=flux + np.random.randn(len(flux))*factor
    #print(kicid,TD_noise,FACTOR)

    
    #plt.plot(time,flux)

    # now let's calculate the fourier transform. the nyquist frequency is:
    nyq=0.5/(30./60./24.)
    fres=1./90./0.0864
     
    freq  = np.arange(0.01, 24., 0.01) # long-cadence  critically sampled 
    # freq  = np.arange(0.01, 734.1, 0.01) # short-cadence  critically sampled 
    freq0 = freq

    (values,counts) = np.unique(np.diff(time),return_counts=True)
    cadence=values[np.argmax(counts)]
    # cadence=30./60./24.

    # time_interp = np.arange(time[0],time[-1],30./(60.*24.))
    time_interp = np.arange(time[0],time[-1],cadence)
    flux_interp = np.interp(time_interp, time, flux)
    time0,flux0 = time,flux #non-interpolated data
    time,flux   = time_interp,flux_interp

    amp  = LombScargle(time,flux).power(freq)

    # FT magic
    # freq, amp = LombScargle(time,flux).autopower(method='fast',samples_per_peak=10,maximum_frequency=nyq)

    # unit conversions
    freq = 1000.*freq/86.4
    bin = freq[1]-freq[0]
    amp = 2.*amp*np.var(flux*1e6)/(np.sum(amp)*bin)

    # White noise correction:
    # wnoise=getkp(f)
    idx=np.where(freq>270)[0]
    wnoise=np.mean(amp[idx])
    amp_wn=np.zeros(len(amp))
    # amp_wn0=np.zeros(len(amp0))

    power_more_than_wnoise=0
    for p in range(0,len(amp)):
        a=amp[p]
        if a-wnoise < 0.:
            amp_wn[p]=amp[p]
        if a-wnoise > 0.:
            amp_wn[p]=a-wnoise
            power_more_than_wnoise+=1

    snr=power_more_than_wnoise/len(amp)
    
    # smooth by 2 muHz
    fres_cd=0.01  #to check if it's smoothed by 2muHz: print(freq[0],freq[0+n]) difference should be 2
    fres_mhz=fres_cd/0.0864

    n=np.int(2./fres_mhz)

    gauss_kernel = Gaussian1DKernel(n)
    pssm = convolve(amp, gauss_kernel)
    wnpssm = convolve(amp_wn, gauss_kernel)
    # wnpssm0 = convolve(amp_wn0, gauss_kernel)

    #Save frequency & power spectrum arrays:
    um=np.where(freq > 10.)[0]

    return freq[um],wnpssm[um],kicid,snr
