# analyze LLR results

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


# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
plt.rcParams['axes.linewidth'] = 1.

print('Loading in catalogues...')
gaia     =ascii.read('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/DR2PapTable1.txt',delimiter='&')
kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
allkps   =list(df['kic_kepmag'])

#astero1=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/labels_full.txt',delimiter=' ',names=['KIC','Teff','logg','lum','kp'],skiprows=1)#,usecols=[0,2])
#astero2=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/rg_yu.txt',delimiter='|',skiprows=1,usecols=[0,3,4,7,8],names=['KIC','Logg','Logg_err','Mass','Mass_err'])
#chaplin=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/Chaplin_2014.tsv',skiprows=35,delimiter=';',names=['KIC','Mass','Mass_errp','Mass_errn'])

#astero3=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/mathur_2019.txt',delimiter='\t',skiprows=14,usecols=[0,2,3,4],names=['KIC','Logg','Logg_errp','Logg_errn'])
#astero3_kics=np.array(astero3['KIC']).astype(int)

yu_header=['KICID','Teff','err','logg','logg_err','Fe/H','err','M_noCorrection','M_nocorr_err','R_noCorrection','err','M_RGB','M_RGB_err','R_RGB','err','M_Clump','M_Clump_err','R_Clump','err','EvoPhase']
# chaplin_header=['KIC','Mass','E_Mass','e_Mass','rho','E_rho','e_rho','logg','E_logg','e_logg']
yu_2018 =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/rg_yu.txt',delimiter='|',names=yu_header,skiprows=1,index_col=False)#,names=yu_header)
# chaplin =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/Chaplin_2014.txt',skiprows=47,delimiter='\t',names=chaplin_header)
# huber_2013  =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/Huber_2013.txt',delimiter='\t',skiprows=37,names=['KIC','Mass','Mass_err'])
mathur_header=['KIC','loggi','e_loggi','r_loggi','n_loggi','logg','E_logg','e_logg','Mass','E_Mass','e_Mass']
mathur_2017 =pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/mathur_2017.txt',delimiter=';',skiprows=54,names=mathur_header)
mathur_2017 =mathur_2017[mathur_2017['n_loggi']=='AST'] #include only asteroseismic measurements

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

def getps(file):
	data=fits.open(file)
	head=data[0].data
	dat=data[1].data
	time=dat['TIME']
	qual=dat['SAP_QUALITY']
	flux=dat['PDCSAP_FLUX']

	good=np.where(qual == 0)[0]
	time=time[good]
	flux=flux[good]

	time_1=time
	flux_1=flux

	res =sigclip(time,flux,50,3)
	good=np.where(res == 1)[0]
	time=time[good]
	flux=flux[good]
	time_2=time
	flux_2=flux

	 # Check Duty Cycle:
	ndays=time[-1]-time[0]
	nmins=ndays*24.*60.
	expected_points=nmins/30.
	observed_points=len(time)

	kicid=int(file.split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
	    
	if kicid in np.array(kepler_catalogue['KIC']):
		row  =kepler_catalogue.loc[kepler_catalogue['KIC']==kicid]
		rad  =row['iso_rad'].item()
		teff =row['iso_teff'].item()
	elif kicid in np.array(gaia['KIC']):                
		teff=gaia['teff'][um[0]]
		rad=gaia['rad'][um[0]]

	closestrad=getclosest(rad,fit_radii)
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

	time_3=time
	flux_3=smoothed_flux
	
	# Remove data points > 5*sigma:
	std =mad_std(flux,ignore_nan=True)
	med =np.median(flux)
	idx =np.where(abs(flux-med)<5.*std)[0]
	time=time[idx]
	flux=flux[idx]

	# now let's calculate the fourier transform. the nyquist frequency is:
	nyq=0.5/(30./60./24.)
	fres=1./90./0.0864

	freq = np.arange(0.01, 24., 0.01) # critically sampled

	#pdb.set_trace()
	# FT magic
	#freq, amp = LombScargle(time,flux).autopower(method='fast',samples_per_peak=10,maximum_frequency=nyq)
	time_interp = np.arange(time[0],time[-1],30./(60.*24.))
	flux_interp = np.interp(time_interp, time, flux)
	time,flux   = time_interp,flux_interp
	amp  = LombScargle(time,flux).power(freq)
	
	# unit conversions
	freq = 1000.*freq/86.4
	bin = freq[1]-freq[0]
	amp = 2.*amp*np.var(flux*1e6)/(np.sum(amp)*bin)

	# White noise correction:
	wnoise=getkp(file)
	amp_wn=np.zeros(len(amp))
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
	n_wnoise=np.int(2./fres_mhz)

	gauss_kernel = Gaussian1DKernel(n) 
	gk_wnoise=Gaussian1DKernel(n_wnoise)

	pssm        = convolve(amp, gauss_kernel)
	pssm_wnoise = convolve(amp_wn, gk_wnoise)    
	timeseries=[time_1,flux_1,time_2,flux_2,time_3,flux_3]

	return timeseries,time,flux,freq,amp_wn,pssm,pssm_wnoise,wnoise,width

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
        #data = np.memmap(pickle_files[i],dtype=np.float32,mode='r',shape=(21000,stars,3)) #oversample=10
        data = np.memmap(pickle_files[i],dtype=np.float32,mode='r',shape=(2099,stars,3)) #oversample=1
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

def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms

def init_plots(true,labelm1,labelm2,files):
	plt.rc('font', size=12)                  # controls default text sizes
	plt.rc('axes', titlesize=12)             # fontsize of the axes title
	plt.rc('axes', labelsize=12)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
	plt.rc('axes', linewidth=1)  
	plt.rc('legend', fontsize=12)
	plt.rc('lines',markersize=4)

	fname='LLR_seismic/astero_wnoise_frac.txt'
	wnoise_frac=pd.read_csv(fname,delimiter=' ',names=['KICID','Fraction','Radius'],skiprows=1)
	fracs=[]
	high_frac_kics=[]
	for i in range(0,len(files)): 
		file=files[i]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		row   =wnoise_frac.loc[wnoise_frac['KICID']==kic]
		frac=float(row['Fraction'].item())
		fracs.append(frac)
		if frac > 0.95:
			high_frac_kics.append(kic)

	# print(high_frac_kics)
	print('SNR min/max',np.min(fracs),np.max(fracs))
	b,rms=returnscatter(labelm1-true)
	std=mad_std(labelm1-true)
	fig=plt.figure(figsize=(5,6))
	ax1=plt.subplot(111)
	cmap = plt.get_cmap('viridis', 6)  #6=number of discrete cmap bins
	plt.plot(true,true,c='k',linestyle='dashed') #
	im1=plt.scatter(true,labelm1,c=fracs,vmin=0.,vmax=1.0,cmap=cmap,label='Model 1 ({})'.format(len(true)),s=3)
	plt.xlabel('Seismic Logg [dex]')
	plt.ylabel('Inferred Logg [dex]')
	plt.xlim([0,4.8])
	plt.ylim([-0.5,6.5])
	
	str1=r'$\sigma$ = '+str('{0:.2f}'.format(rms))
	str2=r'$\sigma_{\mathrm{mad}}$ = '+str('{0:.2f}'.format(std))
	str3='Offset = '+str('{0:.2f}'.format(b))
	STR=str1+'\n'+str2+'\n'+str3
	t=ax1.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1  = fig.colorbar(im1, cax=cax1, orientation="horizontal",cmap=cmap)
	cb1.set_label('Fraction of power above white noise')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')


	plt.tight_layout()
	plt.savefig('seismic.png',dpi=100,bbox_inches='tight')
	plt.show(False)
	exit()

	plt.figure(figsize=(10,5))
	plt.subplot(121)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true,labelm1,facecolors='grey', edgecolors='k',label='Model 1 ({})'.format(len(true)),s=10)
	plt.xlabel('Gaia Logg [dex]')
	plt.ylabel('inferred Logg [dex]')
	plt.xlim([0,4.8])
	plt.legend()
	bias,rms=returnscatter(true-labelm1)
	print('Model 1 --','RMS:',rms,'Offset:',bias)
	plt.subplot(122)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true,labelm2,facecolors='grey', edgecolors='k',label='Model 2 ({})'.format(len(true)),s=10)
	plt.xlabel('Gaia Logg [dex]')
	plt.xlim([0,4.8])
	bias,rms=returnscatter(true-labelm2)
	print('Model 2 --','RMS:',rms,'Offset:',bias)
	plt.legend()
	plt.show(True)
    
def get_avg_psd(all_files,all_data):
	'''To remove fast rotators, plot avg psd between 10-12 uHz against radius of star.'''
	
	def get_freq(all_files):
		'''Get frequency values using one data file.'''
		file=all_files[0]
		b = ascii.read(file)
		freq = b['freq'] 
		return freq

	freq=get_freq(all_files)
	start=np.where(freq>10.)[0][0]
	end=np.where(freq<12.)[0][-1]
	
	radii=np.zeros(len(all_files))
	avg_psd=np.zeros(len(all_files))
	KICS=np.array(kepler_catalogue['KIC'])
	kics_to_save=np.zeros(len(all_files))
	for i in range(0,len(all_files)): 
		file=all_files[i]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		if kic in KICS:
			idx=np.where(KICS==kic)
			radius=np.array(kepler_catalogue['iso_rad'])[idx][0]
			if math.isnan(radius) is True:
				idx=np.where(gaia['KIC']==kic)[0]
				radius=gaia['rad'][idx[0]]
		else:
			idx=np.where(gaia['KIC']==kic)[0]
			radius=gaia['rad'][idx[0]]
			if math.isnan(radius) is True:
				print(2,kic)
		#row=kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		#radius  =row['iso_rad'].item()
		
		power=10.**all_data[i][start:end]
		avg  =np.average(power)
		radii[i]=radius
		avg_psd[i]=avg
		kics_to_save[i]=kic
	#ascii.write([kics_to_save,radii,avg_psd],'/Users/maryumsayeed/LLR_updates/Aug3/radii_avg_psd.txt',names=['KIC','Radius','Power'],overwrite=True)
	plt.scatter(radii,avg_psd,s=10)
	plt.xscale('log')
	plt.yscale('log')
	plt.show(False)

	return radii,avg_psd

def fit_power_radius(radii,power):
	plt.rc('font', size=15)                  # controls default text sizes
	plt.rc('axes', titlesize=15)             # fontsize of the axes title
	plt.rc('axes', labelsize=15)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=15)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=15)            # fontsize of the tick labels
	plt.rc('axes', linewidth=1)  
	plt.rc('legend', fontsize=15)

	# print(radii)
	# print(power)
	# ascii.write([power,radii],'LLR_seismic/astero_power_radii.txt',names=['pow','rad'],overwrite=True)
	x,y=radii,power
	xdata,ydata=np.log10(radii),np.log10(power)
	offset=0
	
	def linfit(x,a,b):
	    x=np.array(x)
	    return (10.**b)*(x**a)
	def logfit(x,a,b):
			x=np.array(x)
			return a*x+b
	ms=2
	remove_c  ='#404788FF'
	keep_c    ='#55C667FF'
	
	a,b=3.2,0.70
	diff=1.0
	keep1=np.where((abs(logfit((xdata),a,b)-ydata)<diff))[0]
	keep2=np.where(radii>20.)[0]
	
	keep=list(set(np.concatenate([keep1,keep2])))
	
	plt.figure(figsize=(6,8))
	gs = gridspec.GridSpec(4, 4,hspace=0)

	ax1 = plt.subplot(gs[0:3, 0:4])
	ax1.scatter(x,y,c=remove_c,s=ms,label='Removed')
	ax1.scatter(x[keep],y[keep],c=keep_c,s=ms)
	xfit=xdata[np.where(xdata<np.log10(20.))[0]]
	ax1.plot(10.**(xfit),10.**(logfit(xfit,a,b)),c='k',label='Linear Best fit')
	ax1.set_axisbelow(True)
	plt.grid(b=True, which='major', linestyle='-', alpha=0.2)
	plt.grid(b=True, which='minor', linestyle='-', alpha=0.2)
	
	plt.minorticks_on()
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(2,7e5)
	plt.xlim(0.5,200)
	ax1.set_ylabel('PSD [ppm$^2$/$\mu$Hz]')

	lgnd=plt.legend(loc='upper left')
	lgnd.legendHandles[1]._sizes = [50]
	
	ax2 = plt.subplot(gs[3:4, 0:4])
	yloggdiff=10.**abs(ydata-logfit(xdata,a,b))
	#ax2.scatter(10.**xdata,yloggdiff,c=remove_c,s=ms)
	#ax2.scatter(10.**(xdata[keep]),yloggdiff[keep],c=keep_c,s=ms)

	ax2.axhline(0,c='k',linestyle='dashed')
	yfit=linfit(x,a,b)
	ax2.scatter(x,yfit/y,c=remove_c,s=ms)
	ax2.scatter(x[keep],(yfit/y)[keep],c=keep_c,s=ms)
	plt.xscale('log')
	plt.yscale('log')
	plt.minorticks_on()
	plt.ylim(1e-2,1e2)
	plt.xlim(0.5,200)
	
	ax2.set_xlabel('Radius [R$_{\\odot}$]')
	ax2.set_ylabel(r'$\frac{\mathrm{Model}}{\mathrm{Data}} $')
	
	ax2.set_axisbelow(True)
	plt.grid(b=True, which='major', linestyle='-', alpha=0.2)
	plt.grid(b=True, which='minor', linestyle='-', alpha=0.2)
	plt.tight_layout()
	# plt.savefig('astero_power_radius.png',bbox_inches='tight',dpi=100)
	plt.show(False)	
	print('manual fitting',len(keep))
	print('-- Stars < {} in PSD vs. Rad plot:'.format(diff),len(keep))
	return keep

def remove_high_rss(chi2_vals):
	plt.clf()
	plt.hist(chi2_vals,bins=100)
	plt.xlim([0,500])
	plt.show(False)
	cutoff=100
	keep=np.where(chi2_vals<cutoff)[0]
	print('-- Stars with RSS < {}:'.format(cutoff),len(keep))
	print('-- Stars with RSS > {}:'.format(cutoff),len(np.where(chi2_vals>cutoff)[0]))
	return keep

def final_result(keep1,keep2,true,labelm1,labelm2):
	keep=list(set(keep1) & set(keep2))

	_,rms=returnscatter(labelm1[keep]-true[keep])
	std1=mad_std(labelm1[keep]-true[keep])
	offset=3*rms
	print('RMS:',rms,'Outlier cutoff:',offset)

	check_idx=np.where(abs(true[keep]-labelm1[keep])>offset)[0]
	keep=np.array(keep)
	print(len(check_idx),len(keep),len(labelm1))
	outliers=keep[check_idx]
	newkeep=list(set(keep)-set(outliers))
	# keep=np.array(newkeep)

	std1=mad_std(labelm1[keep]-true[keep])
	bias,rms1=returnscatter(true[keep]-labelm1[keep])
	
	check_idx2=np.where(abs(true[keep]-labelm2[keep])>offset)[0]
	outliers2=keep[check_idx2]

	std2=mad_std(labelm2[keep]-true[keep])
	bias,rms2=returnscatter(true[keep]-labelm2[keep])
	
	print('Model 1 --','RMS:',rms1,'Bias:',bias,'Stdev:',std1)
	print('Model 2 --','RMS:',rms2,'Bias:',bias,'Stdev:',std2)
	
	bfinal,rmsfinal =returnscatter(true[newkeep]-labelm1[newkeep])
	stdfinal        =mad_std(labelm1[newkeep]-true[newkeep])
	bfinal2,rmsfinal2=returnscatter(true[newkeep]-labelm2[newkeep])
	stdfinal2        =mad_std(labelm2[newkeep]-true[newkeep])
	
	
	plt.figure(figsize=(10,5))
	ax1=plt.subplot(121)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true[keep],labelm1[keep],facecolors='grey', edgecolors='k',label='RMS ({})'.format(len(newkeep)),s=10)
	plt.scatter(true[outliers],labelm1[outliers],facecolors='lightgrey',s=10)

	s1='RMS: {0:.3f}'.format(rmsfinal)
	s2='Stdev: {0:.3f}'.format(stdfinal)
	s3='Offset: {0:.3f}'.format(bfinal)
	STR=s1+'\n'+s2+'\n'+s3	
	t=ax1.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	plt.xlabel('Asteroseismic Logg (dex)')
	plt.ylabel('Inferred Logg (dex)')
	plt.xlim([1,4.7])
	plt.ylim([1,4.7])
	plt.legend(loc='lower right')

	ax2=plt.subplot(122)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true[keep],labelm1[keep],facecolors='grey', edgecolors='k',label='mad_std ({})'.format(len(newkeep)),s=10)
	plt.scatter(true[outliers2],labelm1[outliers2],facecolors='lightgrey',s=10)

	s1='RMS: {0:.3f}'.format(rmsfinal2)
	s2='Stdev: {0:.3f}'.format(stdfinal2)
	s3='Offset: {0:.3f}'.format(bfinal2)
	STR=s1+'\n'+s2+'\n'+s3
	t=ax2.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax2.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	plt.xlabel('Asteroseismic Logg (dex)')
	plt.xlim([1,4.7])
	plt.ylim([1,4.7])
	plt.legend(loc='lower right')

	# plt.savefig('check_rms_vs_stdev_astero.png',bbox_inches='tight')
	return keep,check_idx,offset,rms1,rmsfinal,stdfinal,outliers

def investigate_outliers(outliers,keep,testfiles,true,labelm1):
	print('Investigating outliers...')
	good_stars_lum=[]
	good_stars_kp =[]
	good_stars_teff=[]
	good_stars_rad=[]
	good_stars_true_logg=[]
	good_stars_pred_logg=[]

	for star in keep:
		file=testfiles[star][0:-3]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		kp =allkps[kp_kics.index(kic)]
		try:
			row=kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
			t  =row['iso_teff'].item()
			r  =row['iso_rad'].item()
		except:
			print('...',kic,'not in GKSPC_InOut_V4.')
			idx=np.where(gaia['KIC']==kic)[0]
			t=gaia['teff'][idx][0]
			r=gaia['rad'][idx][0]
		if math.isnan(r) is True:
			idx=np.where(gaia['KIC']==kic)[0]
			t     =gaia['teff'][idx][0]
			r     =gaia['rad'][idx][0]
		l=r**2.*(t/5777.)**4.
		tr=true[star]
		pr=labelm1[star]
		good_stars_lum.append(l)
		good_stars_kp.append(kp)
		good_stars_teff.append(t)
		good_stars_rad.append(r)
		good_stars_true_logg.append(tr)
		good_stars_pred_logg.append(pr)


	bad_stars_lum=[]
	bad_stars_kp =[]
	bad_stars_teff=[]
	bad_stars_rad=[]
	bad_stars_true_logg=[]
	bad_stars_pred_logg=[]
	for star in outliers:
		file=testfiles[star][0:-3]
		#print(star,file)
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		kp =allkps[kp_kics.index(kic)]
		# kic_idx=np.where(gaia['KIC']==kic)
		try:
			row=kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
			t  =row['iso_teff'].item()
			r  =row['iso_rad'].item()
		except:
			print('...',kic,'not in GKSPC_InOut_V4.')
			idx=np.where(gaia['KIC']==kic)[0]
			t=gaia['teff'][idx][0]
			r=gaia['rad'][idx][0]
		if math.isnan(r) is True:
			idx=np.where(gaia['KIC']==kic)[0]
			t     =gaia['teff'][idx][0]
			r     =gaia['rad'][idx][0]
		l=r**2.*(t/5777.)**4.
		tr=true[star]
		pr=labelm1[star]
		bad_stars_lum.append(l)
		bad_stars_kp.append(kp)
		bad_stars_teff.append(t)
		bad_stars_rad.append(r)
		bad_stars_true_logg.append(tr)
		bad_stars_pred_logg.append(pr)

	print('Bad stars stats:')
	for i in [bad_stars_lum,bad_stars_kp,bad_stars_teff,bad_stars_rad,bad_stars_true_logg,bad_stars_pred_logg]:
		print(np.min(i),np.max(i))

	print('Good stars stats:')
	for i in [good_stars_lum,good_stars_kp,good_stars_teff,good_stars_rad,good_stars_true_logg,good_stars_pred_logg]:
		print(np.min(i),np.max(i))		

	_,rms=returnscatter(labelm1[keep]-true[keep])
	cutoff=5*rms
	print('Investigate outliers:')
	print('---len(good)',len(keep))
	print('---len(bad)',len(outliers))
	print('cutoff',cutoff)

	fig=plt.figure(figsize=(8,8))
	plt.rc('font', size=15)                  # controls default text sizes
	plt.rc('axes', titlesize=12)             # fontsize of the axes title
	plt.rc('axes', labelsize=12)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
	plt.rc('axes', linewidth=2)  
	plt.rc('lines', linewidth=2)  
	plt.rc('legend', fontsize=12)  

	fig.text(0.03, 0.5, 'Normalized Counts', va='center', rotation='vertical')

	bad_c='#404788FF'
	good_c ='#55C667FF'


	bins=20
	plt.subplot(321)
	binsl=np.linspace(0,100,bins)
	# binsl=bins
	# binsk=bins
	# binsr=bins
	# binst=bins
	plt.hist(good_stars_lum,bins=binsl,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_lum,bins=binsl,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('Luminosity [L$_{\\odot}$]')

	plt.subplot(322)
	binsk=np.linspace(8,16,bins)
	plt.hist(good_stars_kp,bins=binsk,density=True,edgecolor=good_c,facecolor='none',linewidth=2,label='$\Delta \log g < 5\sigma$')
	plt.hist(bad_stars_kp,bins=binsk,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed',label='$\Delta \log g > 5\sigma$')
	plt.xlabel('Kp')
	

	plt.subplot(323)
	binst=np.linspace(3800.,6800.,20)
	plt.hist(good_stars_teff,bins=binst,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_teff,bins=binst,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel(r'T$_\mathrm{eff}$ [K]')
	plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

	plt.subplot(324)
	binsr=np.linspace(0.,40.,bins)
	plt.hist(good_stars_rad,density=True,bins=binsr,edgecolor=good_c,facecolor='none',linewidth=2,label='$\Delta \log g < 5\sigma$')
	plt.hist(bad_stars_rad,density=True,bins=binsr,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed',label='$\Delta \log g > 5\sigma$')
	plt.ticklabel_format(axis='y', style='sci')
	plt.xlabel('Radius [R$_{\\odot}$]')
	plt.legend()

	plt.subplot(325)
	binsl=np.linspace(1,4.5,bins)
	plt.hist(good_stars_true_logg,density=True,bins=binsl,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_true_logg,density=True,bins=binsl,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('True Logg [dex]')

	plt.subplot(326)
	plt.hist(good_stars_pred_logg,density=True,bins=binsl,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_pred_logg,density=True,bins=binsl,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('Predicted Logg [dex]')

	fig.add_subplot(111, frameon=False)
	# hide tick and tick label of the big axis
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.4)
	# plt.savefig('01.pdf')
	plt.show(False)
	return [good_stars_lum,bad_stars_lum,good_stars_kp,bad_stars_kp,good_stars_teff,bad_stars_teff,good_stars_rad,bad_stars_rad,good_stars_true_logg,bad_stars_true_logg,good_stars_pred_logg,bad_stars_pred_logg]

def pplot_outliers_together(keep,outliers,true,labelm1,labelm2,logg_pos_err,logg_neg_err,result_hists):
	good_stars_lum,bad_stars_lum,good_stars_kp,bad_stars_kp,good_stars_teff,bad_stars_teff,good_stars_rad,bad_stars_rad,good_stars_true_logg,bad_stars_true_logg,good_stars_pred_logg,bad_stars_pred_logg=result_hists
	fig=plt.figure(figsize=(14,8))

	plt.rc('font', size=15)                  # controls default text sizes
	plt.rc('axes', titlesize=15)             # fontsize of the axes title
	plt.rc('axes', labelsize=15)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=15)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=15)            # fontsize of the tick labels
	plt.rc('axes', linewidth=1)  
	plt.rc('legend', fontsize=15)

	total=np.concatenate([keep,outliers])
	labelm1=np.array(labelm1)
	ba,rmsa=returnscatter(labelm1[total]-true[total])
	std=mad_std(labelm1[total]-true[total])
	ms=20

	# AFTER PLOT:
	gs = gridspec.GridSpec(8, 14,hspace=0)  #nrows,ncols

	ax1 = plt.subplot(gs[0:6, 0:6])
	ax1.plot(true[keep],true[keep],c='k',linestyle='dashed')
	print(logg_neg_err[0:10],logg_pos_err[0:10])
	# exit()
	ax1.errorbar(true[keep],labelm1[keep],xerr=[logg_neg_err,logg_pos_err],ecolor='lightcoral',markeredgecolor='k',markerfacecolor='grey',ms=1,fmt='o',alpha=0.5)
	ax1.scatter(true[keep],labelm1[keep],edgecolor='k',facecolor='grey',s=4,zorder=10,alpha=0.4)
	ax1.scatter(true[outliers],labelm1[outliers],edgecolor='grey',facecolor='lightgrey',s=4,zorder=1,label='Outliers',alpha=0.7)
	
	locs, labels = plt.yticks()
	newlabels=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
	plt.yticks(newlabels, newlabels)
	ax1.set_xticklabels([]*len(newlabels))
	plt.minorticks_on()
	ax1.set_ylabel('Inferred Logg [dex]')
	ax1.set_xlim([0.1,4.45])
	ax1.set_ylim([0.1,4.45])
	
	lgnd=plt.legend(loc='lower right')
	lgnd.legendHandles[0]._sizes = [60]

	str1=r'$\sigma$ = '+str('{0:.2f}'.format(rmsa))
	str2=r'$\sigma_{\mathrm{mad}}$ = '+str('{0:.2f}'.format(std))
	str3='Offset = '+str('{0:.2f}'.format(ba))
	STR=str1+'\n'+str2+'\n'+str3
	t=ax1.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	# STR='Offset: '+str('{0:.2f}'.format(ba))
	# t=ax1.text(0.03,0.85,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	# t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	ax2 = plt.subplot(gs[6:8, 0:6])
	ax2.scatter(true[keep],true[keep]-labelm1[keep],alpha=0.4,edgecolor='k',facecolor='grey',zorder=10,s=4)
	ax2.scatter(true[outliers],true[outliers]-labelm1[outliers],alpha=0.4,edgecolor='grey',facecolor='lightgrey',zorder=1,s=4)
	ax2.axhline(0,c='k',linestyle='dashed')
	ax2.set_xlabel('Asteroseismic Logg [dex]')
	ax2.set_ylabel('$\Delta$ Logg [dex]')
	plt.minorticks_on()
	# plt.yticks([-1,0,1], [-1.0,0.0,1.0])
	ax2.set_ylim([-0.5,0.5])
	# ax2.tick_params(which='major',axis='y',pad=15)
	ax2.set_xlim([0.1,4.45])
	
	stda=mad_std(labelm1[keep]-true[keep])
	print('Stats after:')
	print(ba,rmsa,stda)
	text_font={'color':'red','weight':'heavy'}
	plt.tight_layout()
	# plt.savefig('astero_final_residual.pdf',dpi=50)
	mjlength=5
	mnlength=3
	ax1.tick_params(which='both', # Options for both major and minor ticks
	                top=False, # turn off top ticks
	                left=True, # turn off left ticks
	                right=False,  # turn off right ticks
	                bottom=True,# turn off bottom ticks)
	                length=mjlength,
	                width=1)
	ax1.tick_params(which='minor',length=mnlength) 
	ax2.tick_params(which='both', top=True, left=True, bottom=True,length=mjlength,width=1) 
	ax2.tick_params(which='minor',axis='y',length=mnlength) 

	mjlength=7
	mnlength=4
	ax2.tick_params(which='major',axis='x',direction='inout',length=mjlength) 
	ax2.tick_params(which='minor',axis='x',direction='inout',length=mnlength)
	
	stda=mad_std(labelm1[keep]-true[keep])
	print('Stats after:')
	print(ba,rmsa,stda)

	bad_c='#404788FF'
	good_c ='#55C667FF'

	plt.rc('axes', titlesize=12)             # fontsize of the axes title
	plt.rc('axes', labelsize=12)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
	# plt.rc('axes', linewidth=2)  
	plt.rc('lines', linewidth=2)  
	plt.rc('legend', fontsize=12)  

	bins=20
	plt.subplot(gs[0:2, 6:10])
	
	# binsl=bins
	# binsk=bins
	# binsr=bins
	# binst=bins
	binsl=np.linspace(0,3.6,bins)
	plt.hist(np.log10(good_stars_lum),bins=binsl,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(np.log10(bad_stars_lum),bins=binsl,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('log$_{10}$(Luminosity [L$_{\\odot}$])')

	plt.subplot(gs[0:2, 10:14])
	binsk=np.linspace(7.3,16,bins)
	plt.hist(good_stars_kp,bins=binsk,density=True,edgecolor=good_c,facecolor='none',linewidth=2,label='$\Delta \log g \leq 3$'+r'$\sigma$')
	plt.hist(bad_stars_kp,bins=binsk,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed',label='$\Delta \log g > 3$'+r'$\times$'+'rms')
	plt.xlabel('Kp')
	

	plt.subplot(gs[3:5, 6:10])
	binst=np.linspace(3000.,6000.,bins)
	plt.hist(good_stars_teff,bins=binst,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_teff,bins=binst,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel(r'T$_\mathrm{eff}$ [K]')
	plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

	plt.subplot(gs[3:5, 10:14])
	binsr=np.linspace(0,2.5,bins)
	plt.hist(np.log10(good_stars_rad),density=True,bins=binsr,edgecolor=good_c,facecolor='none',linewidth=2,label='$\Delta \log g \leq 3$'+r'$\sigma$')
	plt.hist(np.log10(bad_stars_rad),density=True,bins=binsr,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed',label='$\Delta \log g > 3$'+r'$\sigma$')
	plt.ticklabel_format(axis='y', style='sci')
	plt.xlabel('log$_{10}$(Radius [R$_{\\odot}$])')
	plt.legend()

	plt.subplot(gs[6:8, 6:10])
	binsl=np.linspace(0.1,4.4,bins)
	plt.hist(good_stars_true_logg,density=True,bins=binsl,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_true_logg,density=True,bins=binsl,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('True Logg [dex]')

	plt.subplot(gs[6:8, 10:14])
	plt.hist(good_stars_pred_logg,density=True,bins=binsl,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_pred_logg,density=True,bins=binsl,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('Predicted Logg [dex]')

	fig.tight_layout()
	plt.subplots_adjust(wspace=1.3)
	plt.savefig('astero_result_and_outlier.png',dpi=100,bbox_inches='tight')
	# plt.show(True)
	# plt.savefig('/Users/maryumsayeed/Desktop/HuberNess/iPoster/astero_result_and_outlier0.pdf',dpi=400)
	exit()

def paper_plot(keep,outliers,true,labelm1,labelm2,logg_pos_err,logg_neg_err):
	print('Producing paper plot...')
	plt.figure(figsize=(6,8))
	plt.rc('font', size=15)                  # controls default text sizes
	plt.rc('axes', titlesize=15)             # fontsize of the axes title
	plt.rc('axes', labelsize=15)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=15)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=15)            # fontsize of the tick labels
	plt.rc('axes', linewidth=1)  
	plt.rc('legend', fontsize=15)
	labelm1=np.array(labelm1)
	ba,rmsa=returnscatter(labelm1[keep]-true[keep])
	ms=20

	# AFTER PLOT:
	gs = gridspec.GridSpec(4, 4,hspace=0)

	ax1 = plt.subplot(gs[0:3, 0:4])
	ax1.plot(true[keep],true[keep],c='k',linestyle='dashed')
	ax1.errorbar(true[keep],labelm1[keep],xerr=[logg_neg_err,logg_pos_err],ecolor='lightcoral',markeredgecolor='k',markerfacecolor='grey',ms=4,fmt='o')
	ax1.scatter(true[outliers],labelm1[outliers],edgecolor='grey',facecolor='lightgrey',s=15,zorder=1,label='Outliers')
	
	locs, labels = plt.yticks()
	newlabels=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
	plt.yticks(newlabels, newlabels)
	ax1.set_xticklabels([]*len(newlabels))
	plt.minorticks_on()
	ax1.set_ylabel('Inferred Logg [dex]')
	ax1.set_xlim([1.,4.7])
	ax1.set_ylim([1.,4.7])
	
	lgnd=plt.legend(loc='lower right')
	lgnd.legendHandles[0]._sizes = [60]

	STR='RMS: '+str('{0:.2f}'.format(rmsa))
	t=ax1.text(0.03,0.92,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	STR='Offset: '+str('{0:.2f}'.format(ba))
	t=ax1.text(0.03,0.85,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	ax2 = plt.subplot(gs[3:4, 0:4])
	ax2.scatter(true[keep],true[keep]-labelm1[keep],edgecolor='k',facecolor='grey',zorder=10)
	ax2.scatter(true[outliers],true[outliers]-labelm1[outliers],edgecolor='grey',facecolor='lightgrey',zorder=1)
	ax2.axhline(0,c='k',linestyle='dashed')
	ax2.set_xlabel('Asteroseismic Logg [dex]')
	ax2.set_ylabel('True-Inferred Logg [dex]')
	plt.minorticks_on()
	plt.yticks([-1,0,1], [-1.0,0.0,1.0])
	# ax2.tick_params(which='major',axis='y',pad=15)
	ax2.set_xlim([1.,4.7])
	
	stda=mad_std(labelm1[keep]-true[keep])
	print('Stats after:')
	print(ba,rmsa,stda)
	text_font={'color':'red','weight':'heavy'}
	plt.tight_layout()
	# plt.savefig('astero_final_residual.pdf',dpi=50)
	mjlength=5
	mnlength=3
	ax1.tick_params(which='both', # Options for both major and minor ticks
	                top=False, # turn off top ticks
	                left=True, # turn off left ticks
	                right=False,  # turn off right ticks
	                bottom=True,# turn off bottom ticks)
	                length=mjlength,
	                width=1)
	ax1.tick_params(which='minor',length=mnlength) 
	ax2.tick_params(which='both', top=True, left=True, bottom=True,length=mjlength,width=1) 
	ax2.tick_params(which='minor',axis='y',length=mnlength) 

	mjlength=7
	mnlength=4
	ax2.tick_params(which='major',axis='x',direction='inout',length=mjlength) 
	ax2.tick_params(which='minor',axis='x',direction='inout',length=mnlength)
	
	stda=mad_std(labelm1[keep]-true[keep])
	print('Stats after:')
	print(ba,rmsa,stda)
	text_font={'color':'red','weight':'heavy'}
	plt.tight_layout()
	# plt.savefig('/Users/maryumsayeed/Desktop/HuberNess/iPoster/astero_final_residual.pdf',dpi=50)
	plt.savefig('astero_pplot.pdf',dpi=100)
	plt.show(False)
	#exit()

def get_wnoise_frac(outliers,keep,testfiles,true,labelm1):
	frac_good=[]#np.zeros(len(keep))
	wnoise_frac=pd.read_csv('astero_wnoise_fraction.txt',delimiter=' ',names=['KICID','More','Less'])
	allstars=np.concatenate([keep,outliers])

	for star in keep:
		file=testfiles[star][0:-3]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=str(kic.lstrip('0'))
		if kic in np.array(wnoise_frac['KICID']):
			row   =wnoise_frac.loc[wnoise_frac['KICID']==kic]
			frac=float(row['More'].item())
			frac_good.append(frac)
		else:
			continue 

	frac_bad=[]#np.zeros(len(outliers))
	for star in outliers:
		file=testfiles[star][0:-3]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=str(kic.lstrip('0'))
		if kic in np.array(wnoise_frac['KICID']):
			row   =wnoise_frac.loc[wnoise_frac['KICID']==kic]
			frac=float(row['More'].item())
			frac_bad.append(frac)
		else:
			continue 
	frac_good=np.array(frac_good)
	frac_bad=np.array(frac_bad)
	
	allfrac=np.concatenate([frac_bad,frac_good])
	plt.clf()
	fig=plt.figure(figsize=(8,8))
	ax1=plt.subplot(221)
	ms=7
	mso=20
	plt.plot(true[allstars],true[allstars],c='k',linestyle='--')
	plt.scatter(true[keep],labelm1[keep],c=frac_good,s=ms)
	im=plt.scatter(true[outliers],labelm1[outliers],c=frac_bad,s=mso,zorder=10,marker="^",vmin=allfrac.min(),vmax=allfrac.max(),label='Outliers')
	ax1.set_xlim([1.,4.7])
	ax1.set_ylim([1.,4.7])
	
	lgnd=plt.legend(loc='lower right')
	lgnd.legendHandles[0]._sizes = [80]


	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im, cax=cax1, orientation="horizontal")
	cb1.set_label('Fraction of power above white noise')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()
	
	ax2=plt.subplot(222)
	lim=0.4
	idx1=np.where(frac_good>lim)[0]
	plt.plot(true[allstars],true[allstars],c='k',linestyle='--')
	plt.scatter(true[keep][idx1],labelm1[keep][idx1],c=frac_good[idx1],s=ms)
	idx2=np.where(frac_bad>lim)[0]
	im2=plt.scatter(true[outliers][idx2],labelm1[outliers][idx2],c=frac_bad[idx2],s=mso,zorder=10,marker="^",vmin=allfrac.min(),vmax=allfrac.max(),label='Outliers')
	ax2.set_xlim([1.,4.7])
	ax2.set_ylim([1.,4.7])
	
	STR='power > {}'.format(lim) + '\n' + str('{0:2.0f}'.format((len(idx1)+len(idx2))*100/(len(keep)+len(outliers))))+'%'
	t=ax2.text(0.03,0.92,s=STR,color='k',ha='left',va='center',transform = ax2.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
	a=((len(idx1)+len(idx2))/(len(keep)+len(outliers)))*100
	
	ax1_divider = make_axes_locatable(ax2)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im2, cax=cax1, orientation="horizontal")
	# cb1.set_label('Fraction of power above white noise')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()
	
	ax3=plt.subplot(223)
	lim=0.5
	idx1=np.where(frac_good>lim)[0]
	plt.plot(true[allstars],true[allstars],c='k',linestyle='--')
	plt.scatter(true[keep][idx1],labelm1[keep][idx1],c=frac_good[idx1],s=ms)
	idx2=np.where(frac_bad>lim)[0]
	im3=plt.scatter(true[outliers][idx2],labelm1[outliers][idx2],c=frac_bad[idx2],s=mso,zorder=10,marker="^",vmin=allfrac.min(),vmax=allfrac.max(),label='Outliers')
	ax3.set_xlim([1.,4.7])
	ax3.set_ylim([1.,4.7])
	
	STR='power > {}'.format(lim) + '\n' + str('{0:2.0f}'.format((len(idx1)+len(idx2))*100/(len(keep)+len(outliers))))+'%'
	t=ax3.text(0.03,0.92,s=STR,color='k',ha='left',va='center',transform = ax3.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))


	ax1_divider = make_axes_locatable(ax3)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im3, cax=cax1, orientation="horizontal")
	# cb1.set_label('Fraction of power above white noise')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()
	
	ax4=plt.subplot(224)
	lim=0.6
	idx1=np.where(frac_good>lim)[0]
	plt.plot(true[allstars],true[allstars],c='k',linestyle='--')
	plt.scatter(true[keep][idx1],labelm1[keep][idx1],c=frac_good[idx1],s=ms)
	idx2=np.where(frac_bad>lim)[0]
	im4=plt.scatter(true[outliers][idx2],labelm1[outliers][idx2],c=frac_bad[idx2],s=mso,zorder=10,marker="^",vmin=allfrac.min(),vmax=allfrac.max(),label='Outliers')
	ax4.set_xlim([1.,4.7])
	ax4.set_ylim([1.,4.7])
	
	STR='power > {}'.format(lim) + '\n' + str('{0:2.0f}'.format((len(idx1)+len(idx2))*100/(len(keep)+len(outliers))))+'%'
	t=ax4.text(0.03,0.92,s=STR,color='k',ha='left',va='center',transform = ax4.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	ax1_divider = make_axes_locatable(ax4)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im4, cax=cax1, orientation="horizontal")
	# cb1.set_label('Fraction of power above white noise')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	plt.tight_layout()
	plt.show(False)
	plt.savefig('0astero_wnoise_frac.png')
	plt.tight_layout()
	# plt.savefig('astero_wnoise_frac.png')

def step_by_step_plot(keep,true,labelm1,labelm2,models,alldata,allfiles):
	good_idx=np.where((abs(true[keep]-labelm1[keep])<0.02) & (true[keep]>2) & (true[keep]<2.5))[0]
	print(len(good_idx))
	good_idx=np.where(true[keep]>0)[0]

	alldata=np.array(alldata)[keep]
	print('---done loading array.')
	models=models[keep]
	# exit()
	for star in good_idx[194:195]:
		file=allfiles[keep][star][0:-3]
		# print(star,file)
		timeseries,time,flux,f,amp,pssm,pssm_wnoise,wnoise,width=getps(file)
		time_1,flux_1,time_2,flux_2,time_3,flux_3=timeseries  #flux1=original, flux2=sigma=clipped, flux3=smoothed
		kic=file.split('/')[-1].split('-')[0].split('kplr')[-1]
		kic=int(kic.lstrip('0'))
		test =10.**alldata[star,:] #compare log(PSD) values
		model=10.**models[star] #compare log(PSD) values
		print(kic,width)
		fig=plt.figure(figsize=(6,8))

		fs = 10
		ms = 10
		av = 0.5
		plt.rc('font', size=12)                  # controls default text sizes
		plt.rc('axes', titlesize=15)             # fontsize of the axes title
		plt.rc('axes', labelsize=14)             # fontsize of the x and y labels
		plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
		plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
		plt.rc('axes', linewidth=1)  
		# plt.rc('lines', linewidth=1)  
		plt.rc('legend', fontsize=15)

		
		c1  ='lightcoral'
		c2  ='k'#
		c1  ='#404788FF'
		c1  ='#55C667FF'
		ax1 = plt.subplot(311)
		# ax1.plot(time_1,flux_1,label='Original',c=c1)
		ax1.plot(time_2,flux_2,label='Outliers removed',c=c2,linewidth=1)
		ax1.plot(time_3,flux_3,label='Long-periodic variations removed',c=c1)
		ax1.set_xlabel('Time [Days]')
		ax1.set_ylabel('Flux [Counts]')
		ax1.ticklabel_format(axis='y',style='sci',scilimits=(1,4))

		ax2 = plt.subplot(312,sharex=ax1)
		ax2.plot(time,flux,c=c2,linewidth=1)
		ax2.set_xlabel('Time [Days]')
		ax2.set_ylabel('Normalized Flux')

		y_fmt = ticker.FormatStrFormatter('%0.3f')
		ax2.get_yaxis().get_major_formatter().set_useOffset(True)
		ax2.yaxis.set_major_formatter(y_fmt)
		# ax2.ticklabel_format(axis='y',style='sci',useOffset=True)#scilimits=(-1,1))

		ax3 = plt.subplot(313)
		nbins=2099
		ax3.loglog(f,amp,c=c2,linewidth=1)
		ax3.loglog(f,pssm_wnoise,c=c1,label='Smoothed')
		# ax3.axvspan(f[86],f[86+nbins], facecolor='white', alpha=0.2,zorder=10)
		newfreq=f[86:86+nbins]
		ax3.set_xlim([8,300])
		ax3.set_ylim([0.01,17e4])
		ax3.set_xlabel('Frequency [$\mu$Hz]')
		ax3.set_ylabel(r'PSD [ppm$^2$/$\mu$Hz]')

		fig.align_ylabels([ax1,ax2,ax3])
		# plt.subplots_adjust(wspace=0,hspace=0.3)

		ax1.get_shared_x_axes().join(ax1, ax2)
		fig.tight_layout(pad=0.50)
		# plt.xticks(fontsize=xticksize)
		# plt.yticks(fontsize=yticksize)
		#ax3.set_title('KICID: {} Teff: {} Rad: {} Lum: {} Kp: {}'.format(str(kic),t,round(r,2),round(l,2),round(kp,2)),fontsize=20)
		#tr=round(testlabels[star],2)
		#pr=round(labelm1[star],2)

		#if kic in [7800907,8812919,12006368,2716376]:
		if kic in [8812919]:
			plt.savefig('{}.png'.format(kic),dpi=100,bbox_inches='tight')
			# np.save('time_1.npy',time_1)
			# np.save('time_2.npy',time_2)
			# np.save('time_3.npy',time_3)
			# np.save('flux_1.npy',flux_1)
			# np.save('flux_2.npy',flux_2)
			# np.save('flux_3.npy',flux_3)
			# np.save('time.npy',time)
			# np.save('flux.npy',flux)
			# np.save('f.npy',f)
			# np.save('amp.npy',amp)
			# np.save('pssm_wnoise.npy',pssm_wnoise)
		plt.show(False)
		# exit()
	exit()

def check_models(keep,badidx,true,labelm1,labelm2,models,alldata,allfiles):
	plt.plot(true,true)
	plt.scatter(true[keep],labelm2[keep])
	plt.scatter(true[keep][badidx],labelm2[keep][badidx],c='r')
	plt.show()
	alldata=np.array(alldata)[keep]
	models=models[keep]
	for star in badidx:
		file=allfiles[keep][star][0:-3]
		print(file)
		timeseries,time,flux,f,amp,pssm,pssm_wnoise=getps(file,1)
		time_1,flux_1,time_2,flux_2,time_3,flux_3=timeseries
		kic=file.split('/')[-1].split('-')[0].split('kplr')[-1]
		kic=int(kic.lstrip('0'))
		idx=np.where(gaia['KIC']==kic)[0]
		t=gaia['teff'][idx][0]
		r=gaia['rad'][idx][0]
		l=r**2.*(t/5777.)**4.
		kp =allkps[kp_kics.index(kic)]
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
		plt.title('KICID: {} Teff: {} Rad: {} Lum: {} Kp: {}'.format(str(kic),t,round(r,2),round(l,2),round(kp,2)))
		ax2.set_xlabel('Time (Days)',fontsize=fs)
		ax2.set_ylabel('Relative Flux',fontsize=fs)
		plt.xticks(fontsize=fs)
		plt.yticks(fontsize=fs)

		ax3 = plt.subplot(gs[4:6, 0:3])
		ax3.loglog(f,amp,c='k',linewidth=0.5,label='Quicklook')
		ax3.loglog(f,pssm_wnoise,c='cyan',label='Quicklook')
		#         ax3.loglog(f[864:21000+864],test,c='k',label='smoothed')
		ax3.loglog(f[864:864+21000],model,c='r',label='model')
		ax3.set_xlim([4,280])
		ax3.set_ylim([0.01,1e6])
		ax3.set_xlabel('Frequency ($\mu$Hz)',fontsize=fs)
		ax3.set_ylabel(r'PSD (ppm$^2$/$\mu$Hz)',fontsize=fs)
		a=round(true[keep][star],2)
		b=round(labelm2[keep][star],2)
		#plt.text(10,10**4.,s='True: {} Pred.: {}'.format(a,b),fontsize=20,ha='left')
		ax3.set_title('True: {} Pred.: {}'.format(a,b),fontsize=fs)


		ax4 = plt.subplot(gs[2:5, 3:5])
		plt.plot(true,true,c='k',linewidth=1)
		plt.scatter(true[keep],labelm2[keep],edgecolor='k',facecolor='grey',s=10)
		plt.scatter(true[keep][star],labelm2[keep][star],c='r',s=50)
		plt.tight_layout()
		#plt.savefig('jan2020_pande_LLR4/{}.pdf'.format(kic),dpi=50)
		#plt.show()
		#plt.clf()
		
	#plt.show(False)

def get_logg_error(keep,allfiles):
	print('Getting logg error...')
	logg_pos_err=np.zeros(len(keep))
	logg_neg_err=np.zeros(len(keep))
	counting=np.arange(0,14000,2000)
	for i in range(0,len(keep)):
		star=keep[i]
		file=allfiles[i]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		if kic in astero1:
			idx=np.where(np.asarray(astero3_kics)==kic)[0]
			tlogg    =astero3[idx,1]
			logg_errp=astero3[idx,2]
			logg_errn=astero3[idx,3]
		elif kic in astero2:
			idx=np.where(np.asarray(astero2[:,0])==kic)[0]
			tlogg=astero2[idx,1][0]
			logg_errp=astero2[idx,2][0]
			logg_errn=astero2[idx,2][0]
		logg_pos_err[i]=logg_errp
		logg_neg_err[i]=logg_errn
		if i in counting: print('Logg error bars found for {} stars.'.format(i))
	
	return logg_pos_err,logg_neg_err

def get_mass(keep,true,labelm1,labelm2,allfiles):
	allkics=np.zeros(len(keep))
	allteffs=np.zeros(len(keep))
	radii=np.zeros(len(keep))
	mass =np.zeros(len(keep))
	true_logg=np.zeros(len(keep))
	infer_logg=np.zeros(len(keep))
	rad_pos_err=np.zeros(len(keep))
	rad_neg_err=np.zeros(len(keep))
	logg_pos_err=np.zeros(len(keep))
	logg_neg_err=np.zeros(len(keep))
	grav_const  =6.67e-8   #cm^3*g^-1*s^-2
	solar_radius=6.956e10  #cm
	solar_mass  =1.99e33   #g
	print('Getting stellar params (KICID, radius, mass & err)...')
	c1,c2=0,0
	for i in range(0,len(keep)):
		star=keep[i]
		file=allfiles[keep][i]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		try:
			row=kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
			t  =row['iso_teff'].item()
			r  =row['iso_rad'].item()
			r_errp=row['iso_rad_err1'].item()
			r_errn=row['iso_rad_err2'].item()
		except:
			idx=np.where(gaia['KIC']==kic)[0]
			t     =gaia['teff'][idx][0]
			r     =gaia['rad'][idx][0]
			r_errp=gaia['radep'][idx][0]
			r_errn=gaia['radem'][idx][0]
		if math.isnan(r) is True:
			idx=np.where(gaia['KIC']==kic)[0]
			t     =gaia['teff'][idx][0]
			r     =gaia['rad'][idx][0]
			r_errp=gaia['radep'][idx][0]
			r_errn=gaia['radem'][idx][0]
		true_logg[i]=true[i]
		ilogg=labelm1[keep][i]
		infer_logg[i]=ilogg
		g=10.**ilogg              #cm/s^2
		m=g*((r*solar_radius)**2.)/grav_const
		mass[i]  =m/solar_mass
		radii[i] =r
		rad_pos_err[i]=r_errp
		rad_neg_err[i]=r_errn
		# logg_pos_err[i]=logg_errp
		# logg_neg_err[i]=logg_errn
		allkics[i]=kic
		allteffs[i]=t

	# data=np.array([allkics,true_logg,infer_logg,mass,radii,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err])
	# np.savetxt('ast_data.txt',data.T,fmt='%s')
	return [allkics,allteffs],radii,mass,true_logg,infer_logg,rad_pos_err,rad_neg_err#,logg_pos_err,logg_neg_err

def get_true_mass(kics):
	print('\n')
	print('Finding true mass...')
	mathur_kics=np.array(mathur_2017['KIC'])
	d1,d2,d3=0,0,0
	masses=np.zeros(len(kics))
	mass_perrs=np.zeros(len(kics))
	mass_nerrs=np.zeros(len(kics))

	for i in range(0,len(kics)):
		kic=kics[i]
		if kic in np.array(yu_2018['KICID']):
			row   =yu_2018.loc[yu_2018['KICID']==kic]
			phase=int(row['EvoPhase'])
			if phase==0:
				mass=row['M_noCorrection'].item()
				mass_err=row['M_nocorr_err'].item()
			elif phase==1:
				mass=row['M_RGB'].item()
				mass_err=row['M_RGB_err'].item()
			elif phase==2:
				mass=row['M_Clump'].item()
				mass_err=row['M_Clump_err'].item()
			masses[i]=mass
			mass_perrs[i]=mass_err
			mass_nerrs[i]=mass_err
			d1+=1
		elif kic in mathur_kics:
			idx=np.where(mathur_kics==kic)[0]
			mass=np.array(mathur_2017['Mass'])[idx][0]
			mass_errp=np.array(mathur_2017['E_Mass'])[idx][0]
			mass_errn=np.array(mathur_2017['e_Mass'])[idx][0]
			masses[i]=mass
			mass_perrs[i]=mass_errp
			mass_nerrs[i]=mass_errn
			d2+=1
		else:
			d3+=1

	print('--- Found mass for {} stars in Yu+2018'.format(d1))
	print('--- Found mass for {} stars in Mathur+2017'.format(d2))
	print('--- No mass found for {} stars.'.format(d3))
	return masses, mass_perrs, mass_nerrs

def get_true_logg_err(allfiles,keep,kics):
	mathur_kics=np.array(mathur_2017['KIC'])
	yu_kics=np.array(yu_2018['KICID'])
	d1,d2=0,0
	logg_pos_err=np.zeros(len(keep))
	logg_neg_err=np.zeros(len(keep))
	print('\n')
	print('Getting true logg errors...')
	for i in range(0,len(keep)):
		star=keep[i]
		file=allfiles[keep][i]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		if kic in yu_kics:
			row   =yu_2018.loc[yu_2018['KICID']==kic]
			logg_errp=row['logg_err'].item()
			logg_errn=logg_errp
			d1+=1
		elif kic in mathur_kics:
			idx=np.where(mathur_kics==kic)[0]
			logg_errp=np.array(mathur_2017['e_loggi'])[idx][0]
			logg_errn=logg_errp
			d2+=1
		logg_pos_err[i]=logg_errp
		logg_neg_err[i]=logg_errn
	print('Yu+2018',d1,'Mathur+2017',d2)
	return logg_pos_err,logg_neg_err
	    
def get_inferred_logg_error(fracs,radii):
	data =ascii.read('inferred_logg_error.txt',names=['Lower','Upper','Error'])
	errors=np.zeros(len(fracs))
	def powerlaw(x, p):
		return p[0]*np.power(x,p[1])+p[2]
	popt=[ 2.15178882e-04, -3.86638614e+00,  9.94698735e-03]

	for fi in range(0,len(fracs)):
		f=fracs[fi]
		for i in range(0,len(data)):
			if f > 0.8 or f < 0.2:
				errors[fi]=0.2
			else:
				errors[fi]=powerlaw(f,popt)
	logg_pos_err,logg_neg_err=errors,errors
	return logg_pos_err,logg_neg_err

def get_mass_error(radii,mass,infer_logg,rad_pos_err,rad_neg_err,fracs):
	logg=infer_logg
	logg_pos_err,logg_neg_err=get_inferred_logg_error(fracs,radii)
	# rules are different for logg error calculation with base 10. See here: https://sites.science.oregonstate.edu/~gablek/CH361/Propagation.htm
	rel_pos_err=(((np.log(10)*logg_pos_err)**2.+(2.*rad_pos_err/radii)**2. )**0.5)
	rel_neg_err=(((np.log(10)*logg_neg_err)**2.+(2.*rad_neg_err/radii)**2. )**0.5)
	abs_pos_err=rel_pos_err*mass
	abs_neg_err=rel_neg_err*mass
	return abs_pos_err,abs_neg_err

def get_mass_radius_plot(kics,rms,radii,mass,logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err):
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	plt.rc('font', size=20)                  # controls default text sizes
	plt.rc('axes', titlesize=15)             # fontsize of the axes title
	plt.rc('axes', labelsize=20)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=15)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=15)            # fontsize of the tick labels
	plt.rc('figure', titlesize=15)           # fontsize of the figure title
	plt.rc('axes', linewidth=2)    
	
	print('Making Mass-Radius plot:')
	print('---Logg error to use:',rms)
	
	mass_errp,mass_errn=get_mass_error(radii,mass,logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)
	# ax=plt.subplot(121)
	# bins=np.linspace(0,1,20)
	# plt.hist(mass_errp,bins=bins)
	# plt.xlabel('Mass Positive Error [dex]')
	# plt.ylabel('Count')
	# STR='Median: {}'.format(round(np.median(mass_errp),3))
	# t=plt.text(0.65,0.9,s=STR,color='k',ha='left',va='center',transform = ax.transAxes)
	# t.set_bbox(dict(facecolor='white',edgecolor='white'))
	
	# plt.subplot(122)
	# plt.hist(mass_errn,bins=bins)
	# plt.xlabel('Mass Negative Error [dex]')
	# plt.tight_layout()
	# plt.savefig('if_err_is_error.png')
	# plt.show(False)

	# definitions for the axes
	left, width = 0.14, 0.60
	bottom, height = 0.1, 0.6
	spacing   = 0.0#15
	histwidth = 0.18
	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom + height + spacing, width, histwidth]
	rect_histy = [left + width + spacing, bottom, histwidth, height]

	# start with a rectangular Figure
	fig=plt.figure(figsize=(14, 12))
	ax_scatter = plt.axes(rect_scatter)
	ax_scatter.tick_params(direction='out', top=True, right=True,length=6)
	ax_histx = plt.axes(rect_histx)
	ax_histx.tick_params(direction='out', labelbottom=False,labeltop=True,length=4)
	ax_histy = plt.axes(rect_histy)
	ax_histy.tick_params(direction='out', labelleft=False,labelright=True,labeltop=True,labelbottom=False,length=4)
	cbaxes = fig.add_axes([0.08, 0.1, 0.02, 0.6])  # This is the position for the colorbar
	xy = np.vstack([mass,radii])
	z = gaussian_kde(xy)(xy)
	
	
	#plt.figure(figsize=(13,10))
	#gs  = gridspec.GridSpec(6, 4)
	#ax1 = plt.subplot(gs[2:6, 0:4]) #spans 4 rows and 4 columns

	# Scatter plot
	ax_scatter.errorbar(mass,radii,ms=10,fmt='o',xerr=[mass_errn,mass_errp],yerr=[rad_neg_err,rad_pos_err],markerfacecolor='none',markeredgecolor='none',ecolor='lightgrey')
	im1=ax_scatter.scatter(mass,radii,s=20,c=z,zorder=10)
	ax_scatter.set_xlim([0,16])
	ax_scatter.set_ylim([0.5,5])
	ax_scatter.minorticks_on()
	rdot=' R$_{\\odot}$'
	mdot='M$_{\\odot}$'
	ax_scatter.set_xlabel('Mass [{}]'.format(mdot))
	#ax_scatter.set_ylabel('Radius [{}]'.format(rdot))
	plt.gca().invert_xaxis()

	cb = plt.colorbar(im1,cax=cbaxes)#,location='left')
	cb_label='$\\log_{10}$(Count)'
	cb.set_label(label=cb_label)#,fontsize=30)
	cb.ax.tick_params(labelsize=20)
	cbaxes.yaxis.set_ticks_position('left')
	cbaxes.yaxis.set_label_position('left')


	# cbar.ax.tick_params(labelsize='large')
	

	# definitions for the axes
	binwidth = 0.25
	ax_scatter.set_xlim((5,0))
	ax_scatter.set_ylim((0.5,5))

	mass_bins  =np.arange(0,5,0.1)
	radius_bins=np.arange(0.5,5,0.1)
	lw=2

	xm,ym,_=ax_histx.hist(mass, bins=mass_bins,edgecolor="k",facecolor='lightgrey',linewidth=lw,histtype=u'step')
	xr,yr,_=ax_histy.hist(radii, bins=radius_bins, orientation='horizontal',edgecolor="k",facecolor='lightgrey',histtype=u'step',linewidth=lw)

	# Find peak of histograms:
	mass_peak   =ym[np.argmax(xm)]
	radius_peak =yr[np.argmax(xr)]


	ax_histx.axvline(mass_peak,c='#404788FF',linewidth=2,linestyle='dashed')
	ax_histy.axhline(radius_peak,c='#404788FF',linewidth=2,linestyle='dashed')

	mass_string  = '    {} {}'.format(str(round(mass_peak,1)),mdot)
	radius_string= '    {} {}'.format(str(round(radius_peak,1)),rdot)
	ax_histx.text(mass_peak,np.max(xm)-50,s=mass_string,ha='left',fontsize=20)
	ax_histy.text(5,radius_peak+0.05,s=radius_string,ha='left',fontsize=20)


	ax_histx.set_xlim(ax_scatter.get_xlim())
	ax_histy.set_ylim(ax_scatter.get_ylim())
	ax_histy.set_ylabel('Radius [{}]'.format(rdot),rotation=270,labelpad=20)
	ax_histy.yaxis.set_label_position('right')

	mjlength=8
	mnlength=4
	# Mass-Radius Plot:
	ax_scatter.minorticks_on()
	ax_scatter.yaxis.set_major_locator(ticker.MultipleLocator(1.))
	ax_scatter.grid(which='both',linestyle=':', linewidth='0.5', color='grey',alpha=0.2)
	ax_scatter.tick_params(which='minor', # Options for both major and minor ticks
	                top=False, # turn off top ticks
	                left=True, # turn off left ticks
	                right=False,  # turn off right ticks
	                bottom=True, # turn off bottom ticks
	                length=mnlength)
	ax_scatter.tick_params(which='major',labelsize=20,length=mjlength)


	# Mass Histogram:
	ax_histx.minorticks_on()
	ax_histx.grid(which='both',linestyle=':', linewidth='0.5', color='grey',alpha=0.2)
	ax_histx.yaxis.set_major_locator(ticker.MultipleLocator(100))
	ax_histx.tick_params(which='major', # Options for both major and minor ticks
	                top=True, # turn off top ticks
	                left=True, # turn off left ticks
	                right=False,  # turn off right ticks
	                bottom=False, # turn off bottom ticks
	                labelsize=20,length=mjlength)
	ax_histx.tick_params(which='minor',bottom='off',left=True,top=True,length=mnlength)
	ax_histx.set_yticks([100,200,300,400])#, [100,200])

	# Radius Histogram:
	ax_histy.minorticks_on()
	ax_histy.yaxis.set_major_locator(ticker.MultipleLocator(1.))
	ax_histy.xaxis.set_major_locator(ticker.MultipleLocator(100))
	ax_histy.grid(which='both',linestyle=':', linewidth='0.5', color='grey',alpha=0.2)

	ax_histy.tick_params(which='major', # Options for both major and minor ticks
	                top=True, # turn off top ticks
	                left=False, # turn off left ticks
	                right=True,  # turn off right ticks
	                bottom=False, # turn off bottom ticks
	                labelsize=20,length=mjlength)
	ax_histy.tick_params(which='minor',bottom=False,left=False,top=True,right=True,length=mnlength)
	ax_histy.set_xticks([100,200])#, [100,200])
	
	plt.tight_layout()
	plt.savefig('ast_mr.pdf',dpi=100)
	plt.show(False)

	unc_below1=len(np.where((mass_errp/mass)<0.1)[0])/len(mass)
	unc_below2=len(np.where((mass_errp/mass)<0.15)[0])/len(mass)
	frac_error=np.median((mass_errp/mass))
	print('Mass stats:')
	print('---Max mass:',np.max(mass),'Min mass:',np.min(mass))
	print('---Max radius:',np.max(radii),'Min radius:',np.min(radii))
	print('---Frac. of stars with frac uncertainty below 0.1:',unc_below1)
	print('---Frac. of stars with frac uncertainty below 0.15:',unc_below2)
	print('---Median of frac. uncertainty:',frac_error)
	print('---Max frac. error',np.max(mass_errp/mass))
	print('---Min frac. error',np.min(mass_errp/mass))

	mass_outside=np.where(np.logical_or(mass>16, mass<0))[0]
	radius_outside=np.where(np.logical_or(radii>5, radii<0.5))[0]
	total_outside=len(mass_outside)+len(radius_outside)
	print('---Total # of stars outside mass-radius plot',total_outside)
	# exit()
    # Error Analysis:\n",
    # plt.scatter(mass,radius,s=15,c=radius_err_pos,cmap='Paired',zorder=10)\n",

def get_table(keep,allfiles,testlabels,labels_m1,mass,rad_pos_err,rad_neg_err,rms,tlogg_pos_err,tlogg_neg_err,outliers,tmass,tmass_errp,tmass_errn):
	kics=np.zeros(len(keep))
	teffs=np.zeros(len(keep))
	true_logg=np.zeros(len(keep))
	infer_logg=np.zeros(len(keep))
	radii=np.zeros(len(keep))
	
	# rad_neg_err=np.zeros(len(keep))
	kps=np.zeros(len(keep))
	for i in range(0,len(keep)):
		star=keep[i]
		file=allfiles[star]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		idx=np.where(gaia['KIC']==kic)[0]
		t     =gaia['teff'][idx][0]
		r     =gaia['rad'][idx][0]
		# r_errp=gaia['radep'][idx][0]
		# r_errn=gaia['radem'][idx][0]
		kics[i]  =kic
		teffs[i] =t
		radii[i] =r
		#rad_pos_err[i]=r_errp
		#rad_neg_err[i]=r_errn
		true_logg[i]=testlabels[star]
		infer_logg[i]=labels_m1[star]
		kps[i]=allkps[kp_kics.index(kic)]

	# Flag outliers:
	print('--- setting outlier flag.')
	outliers_flag=np.zeros(len(kics))
	outliers_flag[outliers]=1
	outliers_flag=outliers_flag.astype(int)
	kics,teffs=kics.astype(int),teffs.astype(int) 
	rad_pos_err,rad_neg_err=rad_pos_err,rad_neg_err
	tlogg_pos_err=['{0:.3f}'.format(i) for i in tlogg_pos_err]
	tlogg_neg_err=['{0:.3f}'.format(i) for i in tlogg_neg_err]
	kps=['{0:.3f}'.format(i) for i in kps]

	

	print('--- getting SNR...')
	wnoise_frac=pd.read_csv('LLR_seismic/astero_wnoise_frac.txt',delimiter=' ',names=['KICID','More','Less'])
	snr=[]
	for kic in kics:
		kic=str(kic)
		if kic in np.array(wnoise_frac['KICID']):
			row   =wnoise_frac.loc[wnoise_frac['KICID']==kic]
			frac=float(row['More'].item())
			snr.append(frac)

	mass_errp,mass_errn=get_mass_error(radii,mass,infer_logg,rad_pos_err,rad_neg_err,snr)
	ilogg_pos_err,ilogg_neg_err=get_inferred_logg_error(snr,radii)
	
	# Set -999 for inferred mass of outliers:
	mass[outliers]=-999
	
	from itertools import zip_longest
	header = ['KICID','Kp', 'Teff', 'Radius','Radp','Radn','True_Logg','TLoggp','TLoggn','Inferred_Logg','ILoggp','ILoggn','True_Mass','TMassp','TMassn','Inferred_Mass','IMassp','IMassn','SNR','Outlier'] 
	text_filename='LLR_seismic/Seismic_Sample_v2.csv'
	
	data=kics,kps,teffs,radii,rad_pos_err,rad_neg_err,true_logg,tlogg_pos_err,tlogg_neg_err,infer_logg,ilogg_pos_err,ilogg_neg_err,tmass,tmass_errp,tmass_errn,mass,mass_errp,mass_errn,snr,outliers_flag
	export_data = zip_longest(*data, fillvalue = '')

	with open(text_filename, 'w',newline='') as f:
		w = csv.writer(f)
		w.writerow(header)
		w.writerows(export_data)

	df = pd.read_csv(text_filename,names=header,skiprows=1,index_col=False)
	df.sort_values(by=['KICID'], inplace=True)
	df.to_csv(text_filename,index=False)

	print('...catalogue done!')
	exit()

	df = pd.read_csv(text_filename,index_col=False,delimiter=';')
	df.sort_values(by=['KICID'], inplace=True)
	#print(df)
	HEADER=';'.join(header)
	FMT=['%1.2f' for i in range(0,len(header))]
	FMT[0]='%i'
	FMT[1]='%i'
	# np.savetxt('output2.txt', df.values,delimiter=';',fmt=FMT,header=HEADER)
	df_short=df.head(10)
	FMT=['%s' for i in range(0,len(header)-1)]
	FMT.insert(0,'%i')
	print(FMT)
	#np.savetxt('Astero_short.txt', df_short.values,delimiter=' & ',fmt=FMT,newline=' \\\\\n',header=HEADER)
	exit()

def main(start):
	dirr='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/LLR_seismic/'
	average=np.load(dirr+'average.npy')
	end=len(average)
	testlabels=np.load(dirr+'testlabels.npy')[start:]
	print(len(testlabels))
	labels_m1 =np.load(dirr+'labels_m1.npy')
	labels_m2 =np.load(dirr+'labels_m2.npy')
	spectra_m1=np.load(dirr+'spectra_m1.npy')
	chi2_vals =np.load(dirr+'min_chi2.npy')
	
	print('Size of saved data:',len(testlabels),len(average),len(labels_m1),len(labels_m2),len(spectra_m1))
	dp='LLR_gaia/'
	da='LLR_seismic/'
	train_file_names =[dp+'pande_pickle_1',dp+'pande_pickle_2',dp+'pande_pickle_3',da+'astero_final_sample_1',da+'astero_final_sample_2',da+'astero_final_sample_3',da+'astero_final_sample_4']
	train_file_pickle=[i+'_memmap.pickle' for i in train_file_names]
	train_file_txt   =[i+'.txt' for i in train_file_names]

	print('Getting training data...')
	all_labels,all_data,total_stars,all_files=gettraindata(train_file_txt,train_file_pickle)
	all_labels,all_data,all_files=all_labels[start:start+end],all_data[start:start+end],all_files[start:start+end]
	
	# init_plots(testlabels,labels_m1,labels_m2,all_files)

	radii,avg_psd    =get_avg_psd(all_files,all_data)
	keep_idx_1       =fit_power_radius(radii,avg_psd)
	keep_idx_2       =remove_high_rss(chi2_vals)
	keep,badidx,diff,rms,rmsfinal,std,outliers  =final_result(keep_idx_1,keep_idx_2,testlabels,labels_m1,labels_m2)
	print(len(keep),len(outliers))
	# get_wnoise_frac(outliers,keep,all_files,testlabels,labels_m1)
	# exit()
	# check_models(keep,badidx,testlabels,labels_m1,labels_m2,spectra_m1,all_data,all_files)
	rfit=keep_idx_1
	cfit=keep_idx_2
	a=np.arange(0,len(radii),1)
	r=list(set(a)-set(rfit))
	c=list(set(a)-set(cfit))
	b=list(set(r) & set(c))
	print('\n')
	print('=== rms:',rmsfinal)
	print('=== mad_std:',std)
	print('=== original:',len(radii))
	print('=== final # of stars:',len(keep))
	print('=== stars removed by fitting power vs. radii:',len(r))
	print('=== stars removed by analzying RSS:',len(c))
	print('=== stars common in both removal steps:',len(b))
	print('=== total stars removed:',len(r)+len(c)-len(b))
	print('=== outliers (diff>{}):'.format(diff),len(badidx))
	print('=== fraction of outliers:',len(badidx)/len(keep))
	print('=== logg range:',np.min(testlabels[keep]),np.max(testlabels[keep]))

	oparams,radii,mass,true_logg,infer_logg,rad_pos_err,rad_neg_err=get_mass(keep,testlabels,labels_m1,labels_m2,all_files)
	tlogg_pos_err,tlogg_neg_err=get_true_logg_err(all_files,keep,oparams[0])
	# tlogg_pos_err,tlogg_neg_err=[0]*len(radii),[0]*len(radii) ONLY USE FOR TESTING

	# result_hists=investigate_outliers(outliers,keep,all_files,testlabels,labels_m1)
	# pplot_outliers_together(keep,outliers,testlabels,labels_m1,labels_m2,tlogg_pos_err,tlogg_neg_err,result_hists)

	true_mass,tmass_errp,tmass_errn=get_true_mass(oparams[0])
	get_table(keep,all_files,testlabels,labels_m1,mass,rad_pos_err,rad_neg_err,std,tlogg_pos_err,tlogg_neg_err,badidx,true_mass,tmass_errp,tmass_errn)
	exit()
	# get_mass_radius_plot(kics,rms,radii,mass,infer_logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)
	# np.save('index_of_good_stars.npy',keep)
	
	# step_by_step_plot(keep,testlabels,labels_m1,labels_m2,spectra_m1,all_data,all_files)
	
	#logg_pos_err,logg_neg_err=ge#t_logg_error(keep,all_files)
	# logg_pos_err,logg_neg_err=[0]*len(keep),[0]*len(keep)
	
	# paper_plot(keep,outliers,testlabels,labels_m1,labels_m2,logg_pos_err,logg_neg_err)
	exit()
	
	
main(5964)

