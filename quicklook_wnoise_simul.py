import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import fnmatch
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol
from astropy.io import fits
import glob, re, math
import time as TIME
from astropy.io import ascii
import argparse
# subroutine to perform rough sigma clipping
 

# Create the parser
my_parser = argparse.ArgumentParser(description='Prepare power spectra data given a text file with\
                                                location of .PS files and associated labels.')

# Add the arguments
my_parser.add_argument('-sample','--sample',
                       metavar='sample',
                       type=str,required=True,
                       help='The sample (Gaia or Seismic) to run LLR on.')


my_parser.add_argument('-factor','--factor',
                       metavar='factor',
                       type=int,required=True,
                       help='Factor of noise added (in multiples of epsilon=TD noise).')

# Execute the parse_args() method
args = my_parser.parse_args()

sample=args.sample
FACTOR=args.factor

# Get Kps for all stars:
# whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/SC_sayeed_relation.txt',skiprows=1,delimiter=' ')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
kpdf       =pd.read_csv(kpfile,index_col=False,names=['KIC','Kp'],skiprows=1)

kp_kics  =list(kpdf['KIC'])
kps      =list(kpdf['Kp'])


ppmkp_file=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/wnoise_simulated/ppm_vs_kp.txt',skiprows=1,delimiter=' ',names=['Kp','PPM'])
ppmkp_kp,ppmkp_ppm=np.array(ppmkp_file['Kp']),np.array(ppmkp_file['PPM'])

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
 
###plt.ion()
###plt.clf()

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

# main program starts here
if __name__ == '__main__':
    if sample.lower()=='gaia':
        d='/Users/maryumsayeed/Desktop/pande/pande_lcs/'
    elif sample.lower()=='seismic':
        d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/large_train_sample/'
    else:
        print('Sample not recognized:',sample)
        sys.exit()
    # d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/'
    # d='/Users/maryumsayeed/Downloads/addtoast/'
    files=glob.glob(d+'*.fits')
    # files=files[0:10]
    print('# of files',len(files))
    kics=np.zeros(len(files),dtype='int')
    teffs=np.zeros(len(files))
    rads=np.zeros(len(files))
    lums=np.zeros(len(files))
    
    gaia=ascii.read('DR2PapTable1.txt',delimiter='&')
    data=ascii.read('smoothing_relation/width_vs_radius_test1.txt',delimiter= ' ')
    fit_radii,fit_width=np.array(data['Radii']),np.array(data['Width'])

    # kepler_catalogue=ascii.read('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])
    # #plt.figure(figsize=(10,10))

    # investigate wnoise fraction in power spectra
    nfiles=len(files)
    more=np.zeros(nfiles)
    less=np.zeros(nfiles)
    wnoise_full_avg=np.zeros(nfiles)
    new_wnoise=np.zeros(nfiles)
    old_wnoise=np.zeros(nfiles)
    stars_stats=[]
    n0,n1,n2,n3,n4=np.zeros(nfiles),np.zeros(nfiles),np.zeros(nfiles),np.zeros(nfiles),np.zeros(nfiles)  #4 time domain noise levels
    f0,f1,f2,f3,f4=np.zeros(nfiles),np.zeros(nfiles),np.zeros(nfiles),np.zeros(nfiles),np.zeros(nfiles)  #4 fraction values

    start=TIME.time()
    for i in range(0,len(files)):
        f=files[i]
        kicid=int(files[i].split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
        #plt.clf()
        if kicid>0:#4833377: #9093289 #10675847

            try:
                s0=TIME.time()
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
                print(1,TIME.time()-s0)
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
                # # Use first cut of sample:
                # baseline=65.
                # baseline_time=time[0]+baseline
                # cut=np.where(time<=baseline_time)[0] #take all times below baseline time
                # time=time[cut]
                # flux=flux[cut]
                # === END

                
                #plt.plot(time,flux)
                
                ## if rads[i] <= 36.8.: #from width_vs_radius_test2.txt
                closestrad=getclosest(rads[i],fit_radii)
                idx       =np.where(fit_radii==closestrad)[0]
                best_fit_width=fit_width[idx][0]
                width      =best_fit_width
                
                boxsize    =int(width/(30./60./24.))
                box_kernel = Box1DKernel(boxsize)

                if boxsize % 2 == 0:
                    smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
                else:
                    smoothed_flux = savgol(flux,int(boxsize),1,mode='mirror')
                flux =flux/smoothed_flux
                # overplot this smoothed version, and then divide the light curve through it
                ##plt.plot(time,smoothed_flux)
                 
                # add noise:
                
                fraction_above=[]
                wnoise_level=[]
                
                row   =kpdf.loc[kpdf['KIC']==int(kicid)]
                kp    =float(row['Kp'].item())
                    
                # find time domain noise using time domain noise vs. Kp relation:
                idx2,nearest_kp=find_nearest(ppmkp_kp,kp) # find nearest Kp in our PPM vs. Kp file given Kp of our stars
                TD_noise=ppmkp_ppm[idx2]                   # find time domain noise corresponding to that Kp
                TD_noise=TD_noise/1e6
                print('KIC: {}, Kp: {}, nearest Kp: {}, TD noise: {}'.format(kicid, kp, nearest_kp,TD_noise*1e6))
                print('---2',FACTOR,FACTOR*TD_noise*1e6)

                # factor_of_noise_added=np.linspace(0,0.01,10)
                # for fi in range(0,1):#len(factor_of_noise_added)):

                # factor=factor_of_noise_added[fi]
                # print('---',factor*1e6)
                np.random.seed(0) 
                factor=FACTOR*TD_noise
                flux=flux+np.random.randn(len(flux))*factor
                # plot the filtered light curve
                #fig=#plt.figure(figsize=(10,6))
                #plt.subplot(2,1,1)
                #plt.plot(time,flux)
                #plt.xlabel('Time (Days)')
                #plt.ylabel('Relative flux')
                #plt.title('Rad: {} Lum: {} Kp: {} KICID: {}'.format(round(rad,1),round(lums[i],1),round(kp,2),kicid),fontsize=9)

                # now let's calculate the fourier transform. the nyquist frequency is:
                nyq=0.5/(30./60./24.)
                fres=1./90./0.0864

                fres_cd=0.01
                fres_mhz=fres_cd/0.0864
                 
                freq  = np.arange(0.01, 24., 0.01) # long-cadence  critically sampled 

                #pdb.set_trace()
                # FT magic
                #freq, amp = LombScargle(time,flux).autopower(method='fast',samples_per_peak=10,maximum_frequency=nyq)
                 
                amp = LombScargle(time,flux).power(freq)

                # unit conversions
                freq = 1000.*freq/86.4
                bin = freq[1]-freq[0]
                amp = 2.*amp*np.var(flux*1e6)/(np.sum(amp)*bin)
                
                # White noise correction:
                wnoise=getkp(files[i])
                amp_wn=np.zeros(len(amp))
                
                #print(len(amp),len(freq))
                # calculate average white noise between 270-277 uHz:
                idx=np.where(freq>270)[0]
                # print(i,kicid,np.mean(amp[idx]),wnoise)
                old_wnoise[i]=wnoise
                wnoise=np.mean(amp[idx])
                new_wnoise[i]=wnoise
                power_less_than_wnoise=0
                power_more_than_wnoise=0

                wnoise_avg=np.mean(amp)
                avg_power_above_wnoise=0
                for p in range(0,len(amp)):
                    a=amp[p]
                    if a-wnoise > 0.:
                        amp_wn[p]=a-wnoise
                        power_more_than_wnoise+=1
                    else:
                        amp_wn[p]=a

                print('   ',i,kicid,'more',power_more_than_wnoise/len(amp),wnoise)
                #more[i]=power_more_than_wnoise/len(amp)
                #less[i]=power_less_than_wnoise/len(amp)
                
                # wnoise_full_avg[i]=avg_power_above_wnoise/len(amp)
                # smooth by 2 muHz
                n=np.int(2./fres_mhz)
                 
                gauss_kernel = Gaussian1DKernel(n)
                pssm = convolve(amp, gauss_kernel)
                wnpssm = convolve(amp_wn, gauss_kernel)
                um=np.where(freq > 10.)[0]
                #n=files[i].split('/')

                #Save frequency & power spectrum arrays:
                # if fi==1: 
                if sample.lower()=='gaia':
                    newname=files[i].replace('pande_lcs','pande_lcs_wnoise')
                elif sample.lower()=='seismic':
                    newname=files[i].replace('large_train_sample','large_train_sample_wnoise')
                else:
                    print('Sample not recognized:',sample)
                    sys.exit()
                
                # ascii.write([freq[um],wnpssm[um]],files[i]+'.ps',names=['freq','power'],overwrite=True)
                ascii.write([freq[um],wnpssm[um]],newname+'.ps',names=['freq','power'],overwrite=True)

                '''try:
                    b = ascii.read(f+'.ps')
                except error as e:
                    continue    
                psfreq = b['freq'] 
                psflux = b['power']'''

                
                #plot the power spectrum
                #plt.subplot(2,1,2)

                ##plt.loglog(freq,amp,c='grey',label='Not WN corrected')
                #plt.loglog(freq,amp_wn,label=factor)
                ##plt.loglog(freq,pssm,c='k',label='Not WN corrected')
                ##plt.loglog(freq,wnpssm,c='r',linestyle='dashed',label='WN corrected')
                #plt.axhline(wnoise,linestyle='--')
                #plt.xlabel('Frequency ($\mu$Hz)')
                #plt.ylabel('Power Density')
                #plt.xlim([10.,300])
                #plt.ylim([0.01,1e8])
                # #plt.legend()
                frac=power_more_than_wnoise/len(amp)
                # #plt.title('teff='+str(teffs[i])+' lum='+str(round(lums[i]))+' kicid:'+str(kicid) + ' : '+str(xx))
                #plt.title(' Fraction: {0:.2f}'.format(frac)+' T.D. Noise: {0:.1f} ppm '.format(TD_noise*1e6)+' Added: {0:2.0f} ppm '.format(factor*1e6),fontsize=9)
                #plt.tight_layout()
                #plt.show(False)
                
                fraction_above.append(power_more_than_wnoise/len(amp))
                wnoise_level.append(wnoise)

                # #plt.draw()
                # savedir='/Users/maryumsayeed/LLR_updates/July20/'
                # #plt.savefig(savedir+'{}/{}_{}.png'.format(kicid,kicid,fi))
                # #plt.savefig(savedir+'{}_{}.png'.format(kicid,fi))
                # #plt.savefig(savedir+'seed/{}_{}_seed.png'.format(kicid,fi))
            
                # d='/Users/maryumsayeed/June29_week/'
                # #plt.savefig(d+'wnoise_added_{}.png'.format(xx))
                # input(':')
                # pdb.set_trace()
                # um=np.where(teffs > 0.)[0]

                # ascii.write([factor_of_noise_added,fraction_above,wnoise_level],'wnoise_simulated_low/{}.txt'.format(kicid),names=['Factor','Fraction','WN_level'],overwrite=True)
                # if 'pande' in d:
                #     n0[i]=factor_of_noise_added[0]
                #     n1[i]=factor_of_noise_added[1]
                #     n2[i]=factor_of_noise_added[2]
                #     n3[i]=factor_of_noise_added[3]
                #     n4[i]=factor_of_noise_added[4]
                #     f0[i]=fraction_above[0]
                #     f1[i]=fraction_above[1]
                #     f2[i]=fraction_above[2]
                #     f3[i]=fraction_above[3]
                #     f4[i]=fraction_above[4]
                # else:
                n1[i]=FACTOR*TD_noise#factor_of_noise_added[0]
                f1[i]=fraction_above[0]
                
            except Exception as e:
                print(e)
                print('here')
                # exit()
                continue
            #ascii.write([kics[um],teffs[um],lums[um]],d+'labels_0.5d_1muHz.txt',names=['kic','teff','lum'],overwrite=True)

    # for loop ends here:
    if 'pande' in d:
        sample='pande'
    else:
        sample='astero'
    print('Time taken for {} files:'.format(nfiles),TIME.time()-start)
    # save text file with 
    #ascii.write([kics,n0,f0,n1,f1,n2,f2,n3,f3,n4,f4],'pande_simulation.txt',overwrite=True)
    # ascii.write([kics,n1,f1],'astero_simulation_1.5e.txt',overwrite=True)
    ascii.write([kics,n1,f1],'{}_simulation_{}e.txt'.format(sample,str(FACTOR)),overwrite=True)



