# analyze LLR results

import numpy as np
import time, re
import csv
import pandas as pd
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

gaia     =ascii.read('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/DR2PapTable1.txt',delimiter='&')
kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])
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
	   
	return timeseries,time,flux,freq,amp,pssm,pssm_wnoise

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

def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms

def init_plots(true,labelm1,labelm2):
	plt.figure(figsize=(10,5))
	plt.subplot(121)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true,labelm1,facecolors='grey', edgecolors='k',label='Model 1 ({})'.format(len(true)),s=10)
	plt.xlabel('Gaia Logg (dex)')
	plt.ylabel('inferred Logg (dex)')
	plt.xlim([2,4.8])
	plt.legend()
	bias,rms=returnscatter(true-labelm1)
	print('Model 1 --','RMS:',rms,'Bias:',bias)
	plt.subplot(122)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true,labelm2,facecolors='grey', edgecolors='k',label='Model 2 ({})'.format(len(true)),s=10)
	plt.xlabel('Gaia Logg (dex)')
	plt.xlim([2,4.8])
	bias,rms=returnscatter(true-labelm2)
	print('Model 2 --','RMS:',rms,'Bias:',bias)
	plt.legend()
	plt.show(False)
	print('nothing')
    
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
	for i in range(0,len(all_files)): 
		file=all_files[i]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		idx=np.where(gaia['KIC']==kic)[0]
		radius=gaia['rad'][idx[0]]
		power=10.**all_data[i][start:end]
		avg  =np.average(power)
		radii[i]=radius
		avg_psd[i]=avg
	return radii,avg_psd

def fit_power_radius(radii,power):
	#plt.figure(figsize=(14,4))
	plt.subplot(131)
	plt.loglog(radii,power,'k+')
	plt.xlabel('Radii (solar)')
	plt.ylabel('Power (ppm^2/uHz)')

	xdata,ydata=np.log10(radii),np.log10(power)
	offset=0
	def fitted_model(x, a, b):
	    x=np.array(x)
	    return a*x+b

	def func(p, x, y):
	    return  p[0]*x+p[1]-y

	
	plt.subplot(132)
	x0 = np.array([2.3, 1.2])
	res_lsq = least_squares(func, x0, args=(xdata, ydata),max_nfev=2000)
	y_lsq = fitted_model(sorted(xdata), *res_lsq.x)
	plt.plot(sorted(xdata),y_lsq)
	print('Function constants are:',res_lsq.x)
	diff=0.7
	keep=np.where(abs(y_lsq-ydata)<diff)[0]
	plt.scatter(xdata[keep],ydata[keep],c='g',s=5)
	plt.grid(True)
	print('auto. fitting:',len(keep))

	plt.subplot(133)
	def lin(x,a,b):
		return a*np.array(x)+b
	plt.scatter(xdata, ydata, c='k',label="original data",s=5)

	m,b=2.3,1.2
	plt.plot(xdata,lin(xdata,m,b),c='r')
	diff=0.6
	keep=np.where(abs(lin((xdata),m,b)-ydata)<diff)[0]
	print('manual fitting',len(keep))
	plt.scatter(xdata[keep],ydata[keep],c='g',s=5)
	plt.grid(True)
	plt.show(False)
	# plt.clf()
	print('-- Stars < {} in PSD vs. Rad plot:'.format(diff),len(keep))
	return keep

def remove_high_chi2(chi2_vals):
	plt.hist(chi2_vals,bins=100)
	plt.xlim([0,500])
	plt.show(False)
	plt.clf()
	cutoff=500
	keep=np.where(chi2_vals<cutoff)[0]
	print('-- Stars with chi2 < {}:'.format(cutoff),len(keep))
	print('-- Stars with chi2 > {}:'.format(cutoff),len(np.where(chi2_vals>cutoff)[0]))
	return keep,chi2_vals[keep]

def final_result(keep1,keep2,true,labelm1,labelm2):
	keep=list(set(keep1) & set(keep2))
	print('-- Final stars:',len(keep))
	_,rms=returnscatter(labelm1[keep]-true[keep])

	offset=3*rms
	print('RMS:',rms,'Outlier cutoff:',offset)
	# exit()
	check_idx=np.where(abs(true[keep]-labelm1[keep])>offset)[0]
	keep=np.array(keep)
	print(len(check_idx),len(keep),len(labelm1))
	outliers=keep[check_idx]
	newkeep=list(set(keep)-set(outliers))
	keep=newkeep
	# keep=keep

	plt.figure(figsize=(10,5))
	plt.subplot(121)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true[keep],labelm1[keep],facecolors='grey', edgecolors='k',label='Model 1 ({})'.format(len(true)),s=10)
	plt.scatter(true[outliers],labelm1[outliers],facecolors='lightgrey',s=10)
	plt.xlabel('Gaia Logg (dex)')
	plt.ylabel('Inferred Logg (dex)')
	plt.xlim([2,4.8])
	plt.legend()
	bias,rms1=returnscatter(true[keep]-labelm1[keep])
	
	print('Model 1 --','RMS:',rms1,'Bias:',bias)
	plt.subplot(122)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true[keep],labelm2[keep],facecolors='grey', edgecolors='k',label='Model 1 ({})'.format(len(true)),s=10)
	plt.scatter(true[outliers],labelm2[outliers],facecolors='r',s=10)
	plt.xlabel('Gaia Logg (dex)')
	plt.xlim([2,4.8])
	plt.legend()
	bias,rms2=returnscatter(true[keep]-labelm2[keep])
	print('Model 2 --','RMS:',rms2,'Bias:',bias)
	plt.show(False)
	
	bfinal,rmsfinal=returnscatter(true[newkeep]-labelm1[newkeep])
	return keep,check_idx,offset,rms1,rmsfinal,outliers

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
		# kic_idx=np.where(gaia['KIC']==kic)
		row=kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		t  =row['iso_teff'].item()
		r  =row['iso_rad'].item()
		# t=gaia['teff'][kic_idx][0]
		# r=gaia['rad'][kic_idx][0]
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
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		kp =allkps[kp_kics.index(kic)]
		# kic_idx=np.where(gaia['KIC']==kic)
		row=kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		t  =row['iso_teff'].item()
		r  =row['iso_rad'].item()
		l=r**2.*(t/5777.)**4.
		tr=true[star]
		pr=labelm1[star]
		bad_stars_lum.append(l)
		bad_stars_kp.append(kp)
		bad_stars_teff.append(t)
		bad_stars_rad.append(r)
		bad_stars_true_logg.append(tr)
		bad_stars_pred_logg.append(pr)
	    #print(kic)

	#np.save('badstars.npy',[bad_stars_lum,bad_stars_kp,bad_stars_teff,bad_stars_rad,bad_stars_true_logg,bad_stars_pred_logg])
	#np.save('goodstars.npy',[good_stars_lum,good_stars_kp,good_stars_teff,good_stars_rad,good_stars_true_logg,good_stars_pred_logg])
	# print(good_stars_rad)
	# print(bad_stars_rad)
	# plt.subplot(121)
	# plt.hist(good_stars_rad)
	# plt.subplot(122)
	# plt.hist(bad_stars_rad)
	# plt.show(True)

	_,rms=returnscatter(labelm1[keep]-true[keep])
	cutoff=3*rms
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
	binsl=np.linspace(0,24,bins)
	plt.hist(good_stars_lum,bins=binsl,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_lum,bins=binsl,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('Luminosity [L$_{\\odot}$]')

	plt.subplot(322)
	binsk=np.linspace(9,14,bins)
	plt.hist(good_stars_kp,bins=binsk,density=True,edgecolor=good_c,facecolor='none',linewidth=2,label='$\Delta \log g < 3\sigma$')
	plt.hist(bad_stars_kp,bins=binsk,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed',label='$\Delta \log g > 3\sigma$')
	plt.xlabel('Kp')
	plt.legend(loc='upper left')

	plt.subplot(323)
	binst=np.linspace(4800.,7300.,20)
	plt.hist(good_stars_teff,bins=binst,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_teff,bins=binst,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel(r'T$_\mathrm{eff}$ [K]')
	plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

	plt.subplot(324)
	binsr=np.linspace(0.,10.,bins)
	plt.hist(good_stars_rad,density=True,bins=binsr,edgecolor=good_c,facecolor='none',linewidth=2,label='$\Delta \log g < 3\sigma$')
	plt.hist(bad_stars_rad,density=True,bins=binsr,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed',label='$\Delta \log g > 3\sigma$')
	plt.ticklabel_format(axis='y', style='sci')
	plt.xlabel('Radius [R$_{\\odot}$]')
	

	plt.subplot(325)
	binsl=np.linspace(2.6,4.8,bins)
	plt.hist(good_stars_true_logg,density=True,bins=binsl,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_true_logg,density=True,bins=binsl,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('True Logg [dex]')

	plt.subplot(326)
	binsl=np.linspace(2.6,4.8,bins)
	plt.hist(good_stars_pred_logg,density=True,bins=binsl,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_pred_logg,density=True,bins=binsl,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('Predicted Logg [dex]')

	fig.add_subplot(111, frameon=False)
	# hide tick and tick label of the big axis
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.4)
	plt.savefig('00.pdf')
	plt.show(False)
	return [good_stars_lum,bad_stars_lum,good_stars_kp,bad_stars_kp,good_stars_teff,bad_stars_teff,good_stars_rad,bad_stars_rad,good_stars_true_logg,bad_stars_true_logg,good_stars_pred_logg,bad_stars_pred_logg]
	# exit()

def paper_plot(keep,outliers,true,labelm1,labelm2,logg_pos_err,logg_neg_err):
	plt.figure(figsize=(6,8))
	plt.rc('font', size=20)                  # controls default text sizes
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
	ax1.errorbar(true[keep],labelm1[keep],xerr=[logg_neg_err*-1,logg_pos_err],ecolor='lightcoral',markeredgecolor='k',markerfacecolor='grey',ms=4,fmt='o',zorder=10)
	ax1.scatter(true[outliers],labelm1[outliers],edgecolor='grey',facecolor='lightgrey',s=20,zorder=1,label='Outliers')
	locs, labels = plt.yticks()
	newlabels=[2.5,3.0,3.5,4.0,4.5,5.0]
	plt.yticks(newlabels, newlabels)
	# ax1.set_xticks([])
	ax1.set_xticklabels([]*len(newlabels))
	plt.minorticks_on()
	ax1.set_ylabel('Inferred Logg [dex]')#,labelpad=15)
	ax1.set_xlim([2.,4.8])
	ax1.set_ylim([2.,4.8])
	lgnd=plt.legend(loc='lower right')
	lgnd.legendHandles[0]._sizes = [60]

	STR='RMS: '+str('{0:.2f}'.format(rmsa))
	t=ax1.text(0.03,0.92,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	STR='Bias: '+str('{0:.2f}'.format(ba))
	t=ax1.text(0.03,0.85,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	ax2 = plt.subplot(gs[3:4, 0:4])
	ax2.scatter(true[keep],true[keep]-labelm1[keep],edgecolor='k',facecolor='grey',zorder=10)
	ax2.scatter(true[outliers],true[outliers]-labelm1[outliers],edgecolor='grey',facecolor='lightgrey',s=20,zorder=1)
	ax2.axhline(0,c='k',linestyle='dashed')
	ax2.set_xlabel('Gaia Logg [dex]')
	ax2.set_ylabel('True - Inferred Logg [dex]')
	plt.minorticks_on()
	plt.yticks([-1,0,1], [-1.0,0.0,1.0])
	#ax2.tick_params(which='major',axis='y')#,pad=15)
	ax2.set_xlim([2.,4.8])
	
	mjlength=5
	mnlength=3
	ax1.tick_params(which='both', # Options for both major and minor ticks
	                top='False', # turn off top ticks
	                left='True', # turn off left ticks
	                right='False',  # turn off right ticks
	                bottom='True',# turn off bottom ticks)
	                length=mjlength,
	                width=1)
	ax1.tick_params(which='minor',length=mnlength) 
	ax2.tick_params(which='both', top='True', left='True', bottom='True',length=mjlength,width=1) 
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
	# plt.savefig('/Users/maryumsayeed/Desktop/HuberNess/iPoster/00.pdf',dpi=50)
	plt.show(False)
	# exit()
	return plt

def pplot_outliers_together(keep,outliers,true,labelm1,labelm2,logg_pos_err,logg_neg_err,result_hists):
	good_stars_lum,bad_stars_lum,good_stars_kp,bad_stars_kp,good_stars_teff,bad_stars_teff,good_stars_rad,bad_stars_rad,good_stars_true_logg,bad_stars_true_logg,good_stars_pred_logg,bad_stars_pred_logg=result_hists
	fig=plt.figure(figsize=(14,8))
	plt.rc('font', size=20)                  # controls default text sizes
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
	gs = gridspec.GridSpec(8, 14,hspace=0)  #nrows,ncols

	ax1 = plt.subplot(gs[0:6, 0:6])
	ax1.plot(true[keep],true[keep],c='k',linestyle='dashed')
	# print(len(keep),len(logg_neg_err),len(logg_pos_err),logg_pos_err,logg_neg_err)
	ax1.errorbar(true[keep],labelm1[keep],xerr=[logg_neg_err*-1,logg_pos_err],ecolor='lightcoral',markeredgecolor='k',markerfacecolor='grey',ms=4,fmt='o',zorder=10)
	# ax1.scatter(true[keep],labelm1[keep],s=4)
	ax1.scatter(true[outliers],labelm1[outliers],edgecolor='grey',facecolor='lightgrey',s=20,zorder=1,label='Outliers')
	locs, labels = plt.yticks()
	newlabels=[2.5,3.0,3.5,4.0,4.5,5.0]
	plt.yticks(newlabels, newlabels)
	# ax1.set_xticks([])
	ax1.set_xticklabels([]*len(newlabels))
	plt.minorticks_on()
	ax1.set_ylabel('Inferred Logg [dex]')#,labelpad=15)
	ax1.set_xlim([2.,4.8])
	ax1.set_ylim([2.,4.8])
	lgnd=plt.legend(loc='lower right')
	lgnd.legendHandles[0]._sizes = [60]

	STR='RMS: '+str('{0:.2f}'.format(rmsa))
	t=ax1.text(0.03,0.92,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	STR='Bias: '+str('{0:.2f}'.format(ba))
	t=ax1.text(0.03,0.85,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	ax2 = plt.subplot(gs[6:8, 0:6])
	ax2.scatter(true[keep],true[keep]-labelm1[keep],edgecolor='k',facecolor='grey',zorder=10)
	ax2.scatter(true[outliers],true[outliers]-labelm1[outliers],edgecolor='grey',facecolor='lightgrey',s=20,zorder=1)
	ax2.axhline(0,c='k',linestyle='dashed')
	ax2.set_xlabel('Gaia Logg [dex]')
	ax2.set_ylabel('True - Inferred Logg [dex]')
	plt.minorticks_on()
	plt.yticks([-1,0,1], [-1.0,0.0,1.0])
	#ax2.tick_params(which='major',axis='y')#,pad=15)
	ax2.set_xlim([2.,4.8])
	
	mjlength=5
	mnlength=3
	ax1.tick_params(which='both', # Options for both major and minor ticks
	                top='False', # turn off top ticks
	                left='True', # turn off left ticks
	                right='False',  # turn off right ticks
	                bottom='True',# turn off bottom ticks)
	                length=mjlength,
	                width=1)
	ax1.tick_params(which='minor',length=mnlength) 
	ax2.tick_params(which='both', top='True', left='True', bottom='True',length=mjlength,width=1) 
	ax2.tick_params(which='minor',axis='y',length=mnlength) 

	mjlength=7
	mnlength=4
	ax2.tick_params(which='major',axis='x',direction='inout',length=mjlength) 
	ax2.tick_params(which='minor',axis='x',direction='inout',length=mnlength)
	
	stda=mad_std(labelm1[keep]-true[keep])
	print('Stats after:')
	print(ba,rmsa,stda)
	# text_font={'color':'red','weight':'heavy'}

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
	binsl=np.linspace(0,24,bins)
	plt.hist(good_stars_lum,bins=binsl,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_lum,bins=binsl,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('Luminosity [L$_{\\odot}$]')

	plt.subplot(gs[0:2, 10:14])
	binsk=np.linspace(9,14,bins)
	plt.hist(good_stars_kp,bins=binsk,density=True,edgecolor=good_c,facecolor='none',linewidth=2,label='$\Delta \log g < 3\sigma$')
	plt.hist(bad_stars_kp,bins=binsk,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed',label='$\Delta \log g > 3\sigma$')
	plt.xlabel('Kp')
	plt.legend(loc='upper left')

	# plt.subplot(323)
	plt.subplot(gs[3:5, 6:10])
	binst=np.linspace(4800.,7300.,20)
	plt.hist(good_stars_teff,bins=binst,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_teff,bins=binst,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel(r'T$_\mathrm{eff}$ [K]')
	plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

	# plt.subplot(324)
	plt.subplot(gs[3:5, 10:14])
	binsr=np.linspace(0.,10.,bins)
	plt.hist(good_stars_rad,density=True,bins=binsr,edgecolor=good_c,facecolor='none',linewidth=2,label='$\Delta \log g < 3\sigma$')
	plt.hist(bad_stars_rad,density=True,bins=binsr,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed',label='$\Delta \log g > 3\sigma$')
	plt.ticklabel_format(axis='y', style='sci')
	plt.xlabel('Radius [R$_{\\odot}$]')
	

	# plt.subplot(325)
	plt.subplot(gs[6:8, 6:10])
	binsl=np.linspace(2.6,4.8,bins)
	plt.hist(good_stars_true_logg,density=True,bins=binsl,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_true_logg,density=True,bins=binsl,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('True Logg [dex]')

	# plt.subplot(326)
	plt.subplot(gs[6:8, 10:14])
	binsl=np.linspace(2.6,4.8,bins)
	plt.hist(good_stars_pred_logg,density=True,bins=binsl,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_pred_logg,density=True,bins=binsl,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('Predicted Logg [dex]')
	# fig.text(0.98, 0.5, 'Normalized Counts', va='center', rotation='vertical',fontsize=15)
	# fig.add_subplot(111, frameon=False)
	# hide tick and tick label of the big axis
	# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	fig.tight_layout()
	# plt.subplots_adjust(wspace=0.5)
	plt.savefig('101.png')
	plt.savefig('/Users/maryumsayeed/Desktop/HuberNess/iPoster/pande_result_and_outlier.pdf')
	plt.show(False)
	
	# exit()

def check_models(keep,badidx,true,labelm1,labelm2,models,alldata,allfiles):
	plt.clf()
	plt.ion()
	plt.plot(true,true)
	plt.scatter(true[keep],labelm2[keep])
	plt.scatter(true[keep][badidx],labelm2[keep][badidx],c='r')
	plt.show(False)
	alldata=np.array(alldata)[keep]
	models=models[keep]
	print(len(keep),len(badidx),len(allfiles))
	allkics=[]
	for star in badidx:
		file=allfiles[keep][star][0:-3]
		#print(file)
		timeseries,time,flux,f,amp,pssm,pssm_wnoise=getps(file,1)
		time_1,flux_1,time_2,flux_2,time_3,flux_3=timeseries
		kic=file.split('/')[-1].split('-')[0].split('kplr')[-1]
		kic=int(kic.lstrip('0'))
		idx=np.where(gaia['KIC']==kic)[0]
		t=gaia['teff'][idx][0]
		r=gaia['rad'][idx][0]
		l=r**2.*(t/5777.)**4.
		kp =allkps[kp_kics.index(kic)]
		allkics.append(kic)
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
		plt.show(False)
		plt.clf()
	
	print('done')
	exit()
	#plt.show(False)

def get_mass_error(radii,mass,infer_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err):
	logg=infer_logg
	abs_pos_err=(((logg_pos_err/logg)**2.+((rad_pos_err*2)/(radii**2.))**2.)**0.5)
	abs_neg_err=(((logg_neg_err/logg)**2.+((rad_neg_err*2)/(radii**2.))**2.)**0.5)
	rel_pos_err=abs_pos_err*mass
	rel_neg_err=abs_neg_err*mass
	return rel_pos_err,rel_neg_err

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
	solar_mass  =1.99e33     #g
	print('Calculating mass...')

	for i in range(0,len(keep)):
		star=keep[i]
		file=allfiles[keep][i]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		# idx=np.where(gaia['KIC']==kic)[0]
		row=kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		t  =row['iso_teff'].item()
		r  =row['iso_rad'].item()
		
		#t     =gaia['teff'][idx][0]
		#r     =gaia['rad'][idx][0]
		#r_errp=gaia['radep'][idx][0]
		#r_errn=gaia['radem'][idx][0]
		r_errp=row['iso_rad_err1'].item()
		r_errn=row['iso_rad_err2'].item()
		row   =kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		tlogg  =row['iso_logg'].item()
		logg_errp=row['iso_logg_err1']
		logg_errn=row['iso_logg_err2']	
		true_logg[i]=tlogg
		
		ilogg=labelm1[keep][i]
		infer_logg[i]=ilogg
		g=10.**ilogg              #cm/s^2
		m=g*((r*solar_radius)**2.)/grav_const
		mass[i]  =m/solar_mass
		radii[i] =r
		rad_pos_err[i]=r_errp
		rad_neg_err[i]=r_errn
		logg_pos_err[i]=logg_errp
		logg_neg_err[i]=logg_errn
		allkics[i]=kic
		allteffs[i]=t
	
	data=np.array([allkics,true_logg,infer_logg,mass,radii])
	#np.savetxt('data.txt',data.T,fmt='%s')
	return [allkics,allteffs],radii,mass,true_logg,infer_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err

def get_true_mass(kics):
	tmass=np.zeros(len(kics))
	mass_errp=np.zeros(len(kics))
	mass_errn=np.zeros(len(kics))

	for i in range(0,len(kics)):
		kic=int(kics[i])
		row   =kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		tmass[i]       =row['iso_mass'].item()
		mass_errp[i]       =row['iso_mass_err1'].item()
		mass_errn[i]       =row['iso_mass_err2'].item()
	return tmass,mass_errp,mass_errn

def get_table(oparams,radii,mass,true,infer_logg,rad_pos_err,rad_neg_err,rms,logg_pos_err,logg_neg_err,outliers,tmass,tmass_errp,tmass_errn):
	print('Making table...')
	logg=infer_logg
	kics,teffs=oparams
	kics,teffs=kics.astype(int),teffs.astype(int) 
	ilogg_pos_err,ilogg_neg_err=[rms]*len(kics),[rms]*len(kics)
	mass_errp,mass_errn=get_mass_error(radii,mass,infer_logg,rad_pos_err,rad_neg_err,ilogg_pos_err,ilogg_neg_err)

	header = ['KICID','Kp', 'Teff', 'Radius','Radp','Radn','True_Logg','Loggp','Loggn','Inferred_Logg','True_Mass','TMassp','TMassn','Inferred_Mass','IMassp','IMassn','Outlier'] 
	
	# Flag outliers:
	print('--- setting outlier flag.')
	outliers_flag=np.zeros(len(kics))
	outliers_flag[outliers]=1
	outliers_flag=outliers_flag.astype(int)

	# Find Kp values:
	print('--- finding Kp values.')
	kps=[allkps[kp_kics.index(kic)] for kic in kics]

	text_filename='testing_pande_catalogue.txt'
	with open(text_filename, 'w') as f:
		w = csv.writer(f, delimiter=';')
		w.writerow(header)
		for row in zip(kics,kps,teffs,radii,rad_pos_err,rad_neg_err,true,logg_pos_err,logg_neg_err,logg,tmass,tmass_errp,tmass_errn,mass,mass_errp,mass_errn,outliers_flag):
			w.writerow(row)
	print('...catalogue done!')
	exit()
	df = pd.read_csv(text_filename,index_col=False,delimiter=';')
	df.sort_values(by=['KICID'], inplace=True)
	#print(df)
	HEADER=';'.join(header)
	FMT=['%1.2f' for i in range(0,len(header))]
	FMT[0]='%i'
	FMT[1]='%i'
	# np.savetxt('output2_may27.txt', df.values,delimiter=';',fmt=FMT,header=HEADER)
	df_short=df.head(10)
	FMT=['%s' for i in range(0,len(header)-1)]
	FMT.insert(0,'%i')
	print(FMT)
	# np.savetxt('Pande_short_may27.txt', df_short.values,delimiter=' & ',fmt=FMT,newline=' \\\\\n',header=HEADER)
	# exit()

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
	ax_scatter.errorbar(mass,radii,ms=10,fmt='o',xerr=[mass_errn,mass_errp],yerr=[rad_neg_err*-1,rad_pos_err],markerfacecolor='none',markeredgecolor='none',ecolor='lightgrey')
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
	                top='False', # turn off top ticks
	                left='True', # turn off left ticks
	                right='False',  # turn off right ticks
	                bottom='True', # turn off bottom ticks
	                length=mnlength)
	ax_scatter.tick_params(which='major',labelsize=20,length=mjlength)


	# Mass Histogram:
	ax_histx.minorticks_on()
	ax_histx.grid(which='both',linestyle=':', linewidth='0.5', color='grey',alpha=0.2)
	ax_histx.yaxis.set_major_locator(ticker.MultipleLocator(100))
	ax_histx.tick_params(which='major', # Options for both major and minor ticks
	                top='True', # turn off top ticks
	                left='True', # turn off left ticks
	                right='False',  # turn off right ticks
	                bottom='False', # turn off bottom ticks
	                labelsize=20,length=mjlength)
	ax_histx.tick_params(which='minor',bottom='off',left='True',top='True',length=mnlength)
	ax_histx.set_yticks([100,200,300,400])#, [100,200])

	# Radius Histogram:
	ax_histy.minorticks_on()
	ax_histy.yaxis.set_major_locator(ticker.MultipleLocator(1.))
	ax_histy.xaxis.set_major_locator(ticker.MultipleLocator(100))
	ax_histy.grid(which='both',linestyle=':', linewidth='0.5', color='grey',alpha=0.2)

	ax_histy.tick_params(which='major', # Options for both major and minor ticks
	                top='True', # turn off top ticks
	                left='False', # turn off left ticks
	                right='True',  # turn off right ticks
	                bottom='False', # turn off bottom ticks
	                labelsize=20,length=mjlength)
	ax_histy.tick_params(which='minor',bottom='False',left='False',top='True',right='True',length=mnlength)
	ax_histy.set_xticks([100,200])#, [100,200])
	
	plt.tight_layout()
	plt.savefig('mr0.pdf',dpi=100)
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

	mass_outside=np.where(np.logical_or(mass>5, mass<0))[0]
	radius_outside=np.where(np.logical_or(radii>5, radii<0.5))[0]
	total_outside=len(mass_outside)+len(radius_outside)
	print('---Total # of stars outside mass-radius plot',total_outside)
	# exit()
    # Error Analysis:\n",
    # plt.scatter(mass,radius,s=15,c=radius_err_pos,cmap='Paired',zorder=10)\n",

def get_mass_outliers(kics,radii,mass,logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err):
	plt.rc('font', size=15)                  # controls default text sizes
	plt.rc('axes', titlesize=15)             # fontsize of the axes title
	plt.rc('axes', labelsize=20)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=15)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=15)            # fontsize of the tick labels
	plt.rc('figure', titlesize=15)           # fontsize of the figure title
	plt.rc('axes', linewidth=2)    

	mass_errp,mass_errn=get_mass_error(radii,mass,logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)

	fig=plt.figure(figsize=(14, 14))
	xy = np.vstack([mass,radii])
	z = gaussian_kde(xy)(xy)	
	
	a=[]
	for i in [kics,radii,mass,logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err]:
		a.append(len(i))
	print('len',a)
	logg_neg_err=logg_neg_err*-1
	# Scatter plot
	# good=np.where((mass<0.5) & (radii>1.8))[0]
	masslim=20
	# good=np.where(((mass>=masslim)) )[0]
	# good2=np.where(((mass<5) & (radii>5)))[0]
	# good =np.concatenate([good1,good2])
	# print('# stars with mass < 5:',len(good))
	ax1=plt.subplot(221)
	ax1.plot(true_logg,true_logg,c='k',linestyle='dashed')
	ax1.errorbar(true_logg,logg,xerr=[logg_neg_err,logg_pos_err],ecolor='lightcoral',markeredgecolor='grey',markerfacecolor='lightgrey',ms=4,fmt='o')
	cmap = plt.get_cmap('viridis', 5)  #6=number of discrete cmap bins
	im1=ax1.scatter(true_logg,logg,c=mass,cmap=cmap,s=20,zorder=10)
	# ax1.scatter(true_logg[good],logg[good],facecolor='none',edgecolor='k',s=20,zorder=10)
	ax1.minorticks_off()
	locs, labels = plt.yticks()
	plt.yticks([2.5,3.0,3.5,4.0,4.5,5.0], [2.5,3.0,3.5,4.0,4.5,5.0], fontsize=20)
	ax1.set_ylabel('Inferred Logg [dex]')#,fontsize=20)
	ax1.set_xlabel('Gaia Logg [dex]')#,fontsize=20)
	ax1.set_xlim([2.,4.8])

	# Colorbar Formatting:
	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
	cb1.set_label('Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')

	ax2=plt.subplot(222)
	ax2.plot(true_logg,true_logg,c='k',linestyle='dashed')
	ax2.errorbar(true_logg,logg,xerr=[logg_neg_err,logg_pos_err],ecolor='lightcoral',markeredgecolor='grey',markerfacecolor='lightgrey',ms=4,fmt='o')
	cmap = plt.get_cmap('viridis', 5)  #6=number of discrete cmap bins
	im2=ax2.scatter(true_logg,logg,c=radii,cmap=cmap,s=20,zorder=10)
	# ax1.scatter(true_logg[good],logg[good],facecolor='none',edgecolor='k',s=20,zorder=10)
	ax2.minorticks_off()
	locs, labels = plt.yticks()
	plt.yticks([2.5,3.0,3.5,4.0,4.5,5.0], [2.5,3.0,3.5,4.0,4.5,5.0], fontsize=20)
	ax2.set_ylabel('Inferred Logg [dex]')#,fontsize=20)
	ax2.set_xlabel('Gaia Logg [dex]')#,fontsize=20)
	ax2.set_xlim([2.,4.8])

	# Colorbar Formatting:
	ax2_divider = make_axes_locatable(ax2)
	cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
	cb2  = fig.colorbar(im2, cax=cax2, orientation="horizontal")
	cb2.set_label('Radius [R$_{\\odot}$]')
	cax2.xaxis.set_ticks_position('top')
	cax2.xaxis.set_label_position('top')

	# cb1.ax.tick_params(labelsize=15)

	ax3=plt.subplot(223)
	plt.errorbar(mass,radii,ms=10,fmt='o',xerr=[mass_errn,mass_errp],yerr=[rad_neg_err,rad_pos_err],markerfacecolor='none',markeredgecolor='none',ecolor='lightgrey')
	im3=plt.scatter(mass,radii,s=20,c=z,zorder=10)
	# plt.scatter(mass[good],radii[good],s=20,c='r',zorder=10)
	plt.xlim([0,5])
	# plt.xlim([masslim,np.max(mass)+10])
	plt.ylim([0.5,5])
	rdot=' R$_{\\odot}$'
	mdot='M$_{\\odot}$'
	plt.xlabel('Mass [{}]'.format(mdot))
	plt.ylabel('Radius [{}]'.format(rdot))
	plt.gca().invert_xaxis()
	ax3.xaxis.set_ticks_position('top')
	ax3.xaxis.set_label_position('top')


	# Colorbar Formatting:
	ax3_divider = make_axes_locatable(ax3)
	cax3 = ax3_divider.append_axes("bottom", size="7%", pad="2%")
	cb3 = fig.colorbar(im3, cax=cax3, orientation="horizontal")
	cb_label='$\\log_{10}$(Count)'
	cb3.set_label(label=cb_label)#,fontsize=30)
	cax3.xaxis.set_ticks_position('bottom')
	cax3.xaxis.set_label_position('bottom')

	plt.tight_layout()
	plt.savefig('11.png',dpi=100)
	plt.show(False)
	# exit()

def compare_travis_mass(kics,radii,mass,logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err):
	plt.rc('font', size=12)                  # controls default text sizes
	plt.rc('axes', titlesize=15)             # fontsize of the axes title
	plt.rc('axes', labelsize=12)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
	plt.rc('figure', titlesize=15)           # fontsize of the figure title
	plt.rc('axes', linewidth=2)    
	plt.rc('lines',markersize=4)

	
	a=[]
	for i in [kics,radii,mass,logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err]:
		a.append(len(i))
	print('len',a)

	tmass=np.zeros(len(mass))
	
	for i in range(0,len(mass)):
		kic=int(kics[i])
		row            =kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		tmass[i]       =row['iso_mass'].item()

	plt.clf()
	diff=tmass-mass
	bins=np.arange(-4.5,2.,0.5)
	plt.hist(diff,bins=80)
	plt.xlabel('Berger - Our Mass [M$_{\\odot}$]')
	plt.savefig('hist_of_mass_diff.png')
	plt.show(False)

	logg_neg_err=logg_neg_err*-1
	# tosave=np.array([kics,mass])
	# np.savetxt('kics_our_mass.txt',tosave.T,fmt='%s')
	# tosave=np.array([kics,tmass])
	# np.savetxt('kics_travis_mass.txt',tosave.T,fmt='%s')
	# # exit()
	print(len(mass),len(tmass))
	# exit()
	mass_errp,mass_errn=get_mass_error(radii,mass,logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)

	mdot='M$_{\\odot}$'

	fig=plt.figure(figsize=(8,8))

	ax1=plt.subplot(221)
	ax1.plot(true_logg,true_logg,c='k',linestyle='dashed')
	ax1.errorbar(true_logg,logg,xerr=[logg_neg_err,logg_pos_err],ecolor='lightcoral',markeredgecolor='grey',markerfacecolor='lightgrey',ms=4,fmt='o')
	cmap = plt.get_cmap('viridis', 7)  #6=number of discrete cmap bins
	im1=ax1.scatter(true_logg,logg,c=mass,cmap=cmap,vmin=np.min(tmass), vmax=np.max(tmass),s=20,zorder=10)
	# ax1.scatter(true_logg[good],logg[good],facecolor='none',edgecolor='k',s=20,zorder=10)
	ax1.minorticks_off()
	locs, labels = plt.yticks()
	plt.yticks([2.5,3.0,3.5,4.0,4.5,5.0], [2.5,3.0,3.5,4.0,4.5,5.0])
	ax1.set_ylabel('Inferred Logg [dex]')#,fontsize=20)
	ax1.set_xlabel('Gaia Logg [dex]')#,fontsize=20)
	ax1.set_xlim([2.,4.8])

	# Colorbar Formatting:
	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
	cb1.set_label('Our Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')

	ax2=plt.subplot(222)
	ax2.plot(true_logg,true_logg,c='k',linestyle='dashed')
	ax2.errorbar(true_logg,logg,xerr=[logg_neg_err,logg_pos_err],ecolor='lightcoral',markeredgecolor='grey',markerfacecolor='lightgrey',ms=4,fmt='o')
	cmap = plt.get_cmap('viridis', 7)  #6=number of discrete cmap bins
	im2=ax2.scatter(true_logg,logg,c=tmass,cmap=cmap,s=20,zorder=10)
	# ax1.scatter(true_logg[good],logg[good],facecolor='none',edgecolor='k',s=20,zorder=10)
	ax2.minorticks_off()
	locs, labels = plt.yticks()
	plt.yticks([2.5,3.0,3.5,4.0,4.5,5.0], [2.5,3.0,3.5,4.0,4.5,5.0])
	ax2.set_ylabel('Inferred Logg [dex]')#,fontsize=20)
	ax2.set_xlabel('Gaia Logg [dex]')#,fontsize=20)
	ax2.set_xlim([2.,4.8])

	# Colorbar Formatting:
	ax2_divider = make_axes_locatable(ax2)
	cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
	cb2  = fig.colorbar(im2, cax=cax2, orientation="horizontal")
	cb2.set_label("Travis' Mass [M$_{\\odot}$]")
	cax2.xaxis.set_ticks_position('top')
	cax2.xaxis.set_label_position('top')

	ax3=plt.subplot(223)
	ax3.plot(true_logg,true_logg,c='k',linestyle='dashed')
	ax3.errorbar(true_logg,logg,xerr=[logg_neg_err,logg_pos_err],ecolor='lightcoral',markeredgecolor='grey',markerfacecolor='lightgrey',ms=4,fmt='o')
	cmap = plt.get_cmap('bwr')  #6=number of discrete cmap bins
	im3=ax3.scatter(true_logg,logg,c=(tmass-mass),cmap=cmap,s=20,zorder=10)
	# ax1.scatter(true_logg[good],logg[good],facecolor='none',edgecolor='k',s=20,zorder=10)
	ax3.minorticks_off()
	locs, labels = plt.yticks()
	plt.yticks([2.5,3.0,3.5,4.0,4.5,5.0], [2.5,3.0,3.5,4.0,4.5,5.0])
	ax3.set_ylabel('Inferred Logg [dex]')#,fontsize=20)
	ax3.set_xlabel('Gaia Logg [dex]')#,fontsize=20)
	ax3.set_xlim([2.,4.8])

	# Colorbar Formatting:
	ax3_divider = make_axes_locatable(ax3)
	cax3 = ax3_divider.append_axes("top", size="7%", pad="2%")
	cb3  = fig.colorbar(im3, cax=cax3, orientation="horizontal")
	cb3.set_label("Travis-Our Mass [M$_{\\odot}$]")
	cax3.xaxis.set_ticks_position('top')
	cax3.xaxis.set_label_position('top')

	ax4=plt.subplot(224)
	idx=np.where(true_logg<4)[0]
	
	xy = np.vstack([tmass,mass])
	z = gaussian_kde(xy)(xy)	
	
	ax4.plot(tmass,tmass,c='k',linestyle='dashed')
	cmap = plt.get_cmap('viridis', 5)  #6=number of discrete cmap bins
	im4=ax4.scatter(tmass,mass,c=(tmass-mass),s=5)#,facecolor='grey',edgecolor='k')
	ax4.minorticks_off()
	ax4.set_xlabel('Berger+2020 Mass [{}]'.format(mdot))
	ax4.set_ylabel('Inferred Mass [{}]'.format(mdot))


	# Colorbar Formatting:
	ax4_divider = make_axes_locatable(ax4)
	cax4 = ax4_divider.append_axes("top", size="7%", pad="2%")
	cb4 = fig.colorbar(im4, cax=cax4, orientation="horizontal")
	#cb_label='$\\log_{10}$(Count)'
	cb_label='Berger - Our Mass [{}]'.format(mdot)
	cb4.set_label(label=cb_label)#,fontsize=30)
	cax4.xaxis.set_ticks_position('top')
	cax4.xaxis.set_label_position('top')
	
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.4)
	plt.savefig('Travis_vs_Our_Mass_diff_in_mass.png',dpi=100)
	plt.show(False)
	# exit()

def get_hr_plot(kics,radii,mass,infer_logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err):
	plt.rc('font', size=12)                  # controls default text sizes
	plt.rc('axes', titlesize=12)             # fontsize of the axes title
	plt.rc('axes', labelsize=12)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
	plt.rc('figure', titlesize=12)           # fontsize of the figure title
	plt.rc('axes', linewidth=2)    
	plt.rc('lines',markersize= 4)

	good=np.where((mass>0.9) & (mass<1.2) )[0]
	print('# of stars with mass between 0.9-1.2 M:',len(good),round(len(good)/len(mass)*100))
	teffs=[]#np.zeros(len(good))
	rads =[]#np.zeros(len(good))
	lums =[]#np.zeros(len(good))
	teff_errs   =[]#np.zeros(len(good))
	rad_pos_errs=[]#np.zeros(len(good))
	rad_neg_errs=[]#np.zeros(len(good))
	tmass=[]#np.zeros(len(good))
	tmass_errp=[]#np.zeros(len(good))
	tmass_errn=[]#np.zeros(len(good))
	acc_idx=[]
	for star in range(0,len(good)):
		kic   =kics[good][star]	
		idx   =np.where(gaia['KIC']==kic)[0]
		t     =gaia['teff'][idx][0]
		r     =gaia['rad'][idx][0]
		l     =r**2.*(t/5777.)**4.

		row               =kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		tm=row['iso_mass'].item()
		if tm < 1.5:
			tmass.append(tm)
			tmass_errp.append(row['iso_mass_err1'])
			tmass_errn.append(row['iso_mass_err2'])

			lums.append(l)
			rads.append(r)
			teffs.append(t)
			rad_pos_errs.append(gaia['radep'][idx][0])
			rad_neg_errs.append(gaia['radem'][idx][0])
			teff_errs.append(gaia['teffe'][idx][0])
			acc_idx.append(star)

	
	teff_errs,rad_pos_errs,rad_neg_errs=np.array(teff_errs),np.array(rad_pos_errs),np.array(rad_neg_errs)
	# rel_pos_err,rel_neg_err=np.array(rel_pos_err),np.array(rel_neg_err)
	teffs,lums,rads,tmass=np.array(teffs),np.array(lums),np.array(rads),np.array(tmass)
	# tmass_errp,tmass_errn=np.array(tmass_errp),np.array(tmass_errn)

	# Luminosity Uncertainty:
	abs_pos_err=(((teff_errs/teffs)**2.+((rad_pos_errs*2)/(rads**2.))**2.)**0.5)
	abs_neg_err=(((teff_errs/teffs)**2.+((rad_neg_errs*2)/(rads**2.))**2.)**0.5)
	rel_pos_err=(abs_pos_err*lums)
	rel_neg_err=(abs_neg_err*lums)
	
	for i in [teff_errs,abs_pos_err,abs_neg_err,rel_pos_err,rel_neg_err,teffs,lums,tmass,acc_idx,mass]:
		print(len(i))

	filename='KITP/APOKASC_cat_v6.6.1.fits'
	kep_cat='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv'
	kep_cat=pd.read_csv(kep_cat)

	infile= fits.open(filename)#,ignore_missing_end=True)
	header=infile[1].header
	data=infile[1].data

	kitp_kics=data['KEPLER_ID']
	kitp_feh=data['FE_H_ADOP_COR']


	kitp_kics=np.array([int(i) for i in kitp_kics])
	kics=np.array([int(i) for i in kics])

	fehs=[]
	good_idx=[]
	yes=0
	no=0

	for i in range(0,len(acc_idx)):
		kic  =int(kics[acc_idx][i])
		row  =kep_cat.loc[kep_cat['KIC']==kic]
		feh  =row['feh'].item()
		if feh > -90:
			yes=yes+1
			fehs.append(feh)
			good_idx.append(i)
		elif kic in kitp_kics:
			idx=np.where(kitp_kics==kic)[0]
			feh=kitp_feh[idx][0]
			if feh > -90:
				fehs.append(feh)
				good_idx.append(i)
				yes=yes+1
			else:
				no=no+1	
				fehs.append(-99)
		else:
			no=no+1
			fehs.append(-99)

	fehs    =np.array(fehs)
	print(fehs.min(),fehs.max())
	good_idx=np.array(good_idx)
	good    =np.array(acc_idx)

	plt.clf()
	plt.hist(tmass-mass[good],bins=15)
	plt.xlabel('Berger - Our Mass [M$_{\\odot}$]')
	plt.show(False)
	# plt.savefig('hist_of_mass_diff.png')


	fig=plt.figure(figsize=(8,8))
	
	ax1=plt.subplot(221)
	cmap = plt.get_cmap('viridis', 8)  #6=number of discrete cmap bins
	ax   = plt.errorbar(teffs,lums,xerr=[teff_errs,teff_errs],yerr=[rel_neg_err,rel_pos_err],fmt='o',mfc='none',mec='k',ecolor='lightgrey')
	cax  = plt.scatter(teffs,lums,c=mass[good], cmap=cmap, vmin=np.min(tmass), vmax=np.max(tmass) ,zorder=10)
	plt.yscale('log')
	plt.ylim([0.7,100])
	plt.xlim([4300,7300])
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Luminosity [Solar]')
	plt.gca().invert_xaxis()
	
	# Colorbar Formatting:
	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(cax, cax=cax1, orientation="horizontal")
	cb1.set_label('Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()


	ax2=plt.subplot(222)
	cmap = plt.get_cmap('viridis', 8)  #6=number of discrete cmap bins
	plt.errorbar(teffs,lums,xerr=[teff_errs,teff_errs],yerr=[rel_neg_err,rel_pos_err],fmt='o',mfc='grey',mec='k',ecolor='lightgrey')
	cax  = plt.scatter(teffs,lums,c=tmass,cmap=cmap, vmin=np.min(tmass), vmax=np.max(tmass),zorder=10)
	plt.yscale('log')
	plt.ylim([0.7,100])
	plt.xlim([4300,7300])
	plt.xlabel('Effective Temperature [K]')
	plt.gca().invert_xaxis()

	# Colorbar Formatting:	
	ax2_divider = make_axes_locatable(ax2)
	cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
	cb2 = fig.colorbar(cax, cax=cax2, orientation="horizontal")
	cb2.set_label('Berger Mass [M$_{\\odot}$]')
	cax2.xaxis.set_ticks_position("top")
	cax2.xaxis.set_label_position('top')
	cb2.ax.tick_params()
	plt.tight_layout()
	plt.subplots_adjust(wspace=None)


	ax2=plt.subplot(223)
	cmap = plt.get_cmap('viridis', 5)  #6=number of discrete cmap bins
	plt.errorbar(teffs,lums,xerr=[teff_errs,teff_errs],yerr=[rel_neg_err,rel_pos_err],fmt='o',mfc='grey',mec='k',ecolor='lightgrey')
	cax  = plt.scatter(teffs,lums,c=(tmass-mass[good]),cmap=cmap, vmin=np.min((tmass-mass[good])), vmax=np.max((tmass-mass[good])),zorder=10)
	plt.yscale('log')
	plt.ylim([0.7,100])
	plt.xlim([4300,7300])
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Luminosity [Solar]')
	plt.gca().invert_xaxis()

	# Colorbar Formatting:	
	ax2_divider = make_axes_locatable(ax2)
	cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
	cb2 = fig.colorbar(cax, cax=cax2, orientation="horizontal")
	cb2.set_label('Berger Mass - Our Mass [M$_{\\odot}$]')
	cax2.xaxis.set_ticks_position("top")
	cax2.xaxis.set_label_position('top')
	cb2.ax.tick_params()
	plt.tight_layout()
	plt.subplots_adjust(wspace=None) 

	ax2=plt.subplot(224)
	cmap = plt.get_cmap('viridis', 8)  #6=number of discrete cmap bins
	plt.errorbar(teffs[good_idx],lums[good_idx],xerr=[teff_errs[good_idx],teff_errs[good_idx]],yerr=[rel_neg_err[good_idx],rel_pos_err[good_idx]],fmt='o',mfc='grey',mec='k',ecolor='lightgrey')
	cax  = plt.scatter(teffs[good_idx],lums[good_idx],c=fehs[good_idx],cmap=cmap, vmin=np.min(fehs[good_idx]), vmax=np.max(fehs[good_idx]),zorder=10)
	plt.yscale('log')
	plt.ylim([0.7,100])
	plt.xlim([4300,7300])
	plt.xlabel('Effective Temperature [K]')
	plt.gca().invert_xaxis()

	# Colorbar Formatting:	
	ax2_divider = make_axes_locatable(ax2)
	cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
	cb2 = fig.colorbar(cax, cax=cax2, orientation="horizontal")
	cb2.set_label('[Fe/H]')
	cax2.xaxis.set_ticks_position("top")
	cax2.xaxis.set_label_position('top')
	cb2.ax.tick_params()
	plt.tight_layout()
	plt.subplots_adjust(wspace=0.2) 

	
	plt.savefig('HR_Diagram_from_Mass_better.png',dpi=100)
	plt.show(False)
	# exit()

	print(len(mass),len(kics),len(radii))
	return 0

def get_mass_lum_plot(kics,radii,mass,logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err):
	plt.rc('font', size=12)                  # controls default text sizes
	plt.rc('axes', titlesize=15)             # fontsize of the axes title
	plt.rc('axes', labelsize=12)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
	plt.rc('figure', titlesize=15)           # fontsize of the figure title
	plt.rc('axes', linewidth=2)    

	lums=[]
	rads=[]
	tmass=[]
	for kic in kics:
		idx=np.where(gaia['KIC']==kic)[0]
		t    	=gaia['teff'][idx][0]
		r       =gaia['rad'][idx][0]
		l      	=r**2.*(t/5777.)**4.
		row     =kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		tmass.append(row['iso_mass'].item())
		lums.append(l)
		rads.append(r)
	
	
	# start with a rectangular Figure
	fig=plt.figure(figsize=(8,8))
	ldot='L$_{\\odot}$'
	rdot='R$_{\\odot}$'
	mdot='M$_{\\odot}$'
	
	ax1=plt.subplot(221)

	xy = np.vstack([tmass,np.log10(lums)])
	z = gaussian_kde(xy)(xy)

	# Scatter plot
	im1=ax1.scatter(tmass,lums,s=10,c=z,zorder=10)
	ax1.set_xlabel('Berger Mass [{}]'.format(mdot))
	ax1.set_ylabel('Stellar Luminosity [{}]'.format(ldot))
	plt.xlim(0.2,3)
	plt.gca().invert_xaxis()
	ax1.set_yscale('log')
	ax1.minorticks_on()

	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1  = fig.colorbar(im1, cax=cax1, orientation="horizontal")
	cb1.set_label('log$_{10}$(Count)')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	    

	ax2=plt.subplot(222)
	xy = np.vstack([mass,np.log10(lums)])
	z = gaussian_kde(xy)(xy)

	# Scatter plot
	im2=ax2.scatter(mass,lums,s=10,c=z,zorder=10)
	rdot='L$_{\\odot}$'
	mdot='M$_{\\odot}$'
	ax2.set_xlabel('Inferred Mass [{}]'.format(mdot))
	ax2.set_ylabel('Stellar Luminosity [{}]'.format(ldot))
	plt.xlim(0.2,3)
	plt.gca().invert_xaxis()
	ax2.set_yscale('log')
	ax2.minorticks_on()
	locs,labels=plt.yticks()

	ax2_divider = make_axes_locatable(ax2)
	cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
	cb2  = fig.colorbar(im2, cax=cax2, orientation="horizontal")
	cb2.set_label('log$_{10}$(Count)')
	cax2.xaxis.set_ticks_position('top')
	cax2.xaxis.set_label_position('top')

	ax2=plt.subplot(223)

	# Scatter plot
	z=np.array(tmass)-np.array(mass)
	im2=ax2.scatter(tmass,lums,s=10,c=z,zorder=10)
	ax2.set_xlabel('Berger Mass [{}]'.format(mdot))
	ax2.set_ylabel('Stellar Luminosity [{}]'.format(ldot))
	ax2.set_yscale('log')
	plt.xlim(0.2,3)
	plt.gca().invert_xaxis()
	ax2.minorticks_on()

	ax2_divider = make_axes_locatable(ax2)
	cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
	cb2  = fig.colorbar(im2, cax=cax2, orientation="horizontal")
	cb2.set_label('Berger - Our Mass')
	cax2.xaxis.set_ticks_position('top')
	cax2.xaxis.set_label_position('top')


	ax2=plt.subplot(224)
	xy = np.vstack([mass,rads])
	z = gaussian_kde(xy)(xy)

	# Scatter plot
	im2=ax2.scatter(mass,rads,s=10,c=z,zorder=10)
	ax2.set_xlabel('Inferred Mass [{}]'.format(mdot))
	ax2.set_ylabel('Stellar Radius [{}]'.format(rdot))
	plt.xlim(0.2,5)
	plt.ylim(0.5,5)
	plt.gca().invert_xaxis()
	ax2.minorticks_on()

	ax2_divider = make_axes_locatable(ax2)
	cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
	cb2  = fig.colorbar(im2, cax=cax2, orientation="horizontal")
	cb2.set_label('log$_{10}$(Count)')
	cax2.xaxis.set_ticks_position('top')
	cax2.xaxis.set_label_position('top')

	plt.tight_layout()
	plt.subplots_adjust(hspace=0.4)

	plt.savefig('mass_lum.png',dpi=100)
	plt.show(False)
	# exit()

def get_wnoise_frac(outliers,keep,testfiles,true,labelm1):
	frac_good=[]#np.zeros(len(keep))
	wnoise_frac=pd.read_csv('pande_wnoise_fraction.txt',delimiter=' ',names=['KICID','More','Less'])
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
	# plt.rc('font', size=20)                  # controls default text sizes

	ms=12
	plt.plot(true[allstars],true[allstars],c='k',linestyle='--')
	plt.scatter(true[keep],labelm1[keep],c=frac_good,s=ms)
	im1=plt.scatter(true[outliers],labelm1[outliers],c=frac_bad,s=ms,marker="^",vmin=allfrac.min(),vmax=allfrac.max(),label='Outliers')
	lgnd=plt.legend(loc='lower right')
	lgnd.legendHandles[0]._sizes = [80]
	
	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
	# cb1.set_label('Fraction of power above white noise')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()
	
	ax2=plt.subplot(222)
	lim=0.4
	idx1=np.where(frac_good>lim)[0]
	plt.plot(true[allstars],true[allstars],c='k',linestyle='--')
	plt.scatter(true[keep][idx1],labelm1[keep][idx1],c=frac_good[idx1],s=ms)
	idx2=np.where(frac_bad>lim)[0]
	im2=plt.scatter(true[outliers][idx2],labelm1[outliers][idx2],c=frac_bad[idx2],s=ms,marker="^",vmin=lim,vmax=allfrac.max(),label='Outliers')
	lgnd.legendHandles[0]._sizes = [80]
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
	im3=plt.scatter(true[outliers][idx2],labelm1[outliers][idx2],c=frac_bad[idx2],s=ms,marker="^",vmin=lim,vmax=allfrac.max(),label='Outliers')
	lgnd.legendHandles[0]._sizes = [80]
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
	im4=plt.scatter(true[outliers][idx2],labelm1[outliers][idx2],c=frac_bad[idx2],s=ms,marker="^",vmin=lim,vmax=allfrac.max(),label='Outliers')
	lgnd.legendHandles[0]._sizes = [80]
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
	plt.savefig('0pande_wnoise_frac.png')

def main(start):
	dirr='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/jan2020_pande_sample/'
	average=np.load(dirr+'average.npy')
	end=len(average)
	testlabels=np.load(dirr+'testlabels.npy')[start:end]
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
	
	#init_plots(testlabels,labels_m1,labels_m2)
	radii,avg_psd       =get_avg_psd(all_files,all_data)
	keep_idx_1          =fit_power_radius(radii,avg_psd)
	keep_idx_2,chi2     =remove_high_chi2(chi2_vals)
	keep,badidx,diff,rms,rmsfinal,outliers  =final_result(keep_idx_1,keep_idx_2,testlabels,labels_m1,labels_m2)
	# result_hists=investigate_outliers(outliers,keep,all_files,testlabels,labels_m1)
	#
	# get_wnoise_frac(outliers,keep,all_files,testlabels,labels_m1)
	# exit()
	
	print('Max/Min',np.max(testlabels[keep]),np.min(testlabels[keep]))
	# check_models(keep,badidx,testlabels,labels_m1,labels_m2,spectra_m1,all_data,all_files)

	print('=== rms:',rmsfinal)
	print('=== original:',len(radii))
	print('=== after cleaning:',len(keep))
	print('=== outliers (diff>{}):'.format(diff),len(badidx))
	print('=== fraction of outliers:',len(badidx)/len(keep))
	
	exit()
	oparams,radii,mass,true_logg,infer_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err=get_mass(keep,testlabels,labels_m1,labels_m2,all_files)
	logg_pos_err,logg_neg_err=[rms]*len(mass),[-rms]*len(mass)

	# print(len(radii),len(outliers),len(keep),len(oparams[0]))
	# Save chi2 values for appendix plots:
	# ascii.write([(oparams[0]).astype(int),chi2_vals[keep]],'pande_chi2.txt',names=['KICID','Chi2'],overwrite=True)
	# np.savetxt('pande_chi2.txt',np.array([(,]).T,fmt='%s')

	# true_mass,tmass_errp,tmass_errn=get_true_mass(oparams[0])

	# exit()
	# get_table(oparams,radii,mass,true_logg,infer_logg,rad_pos_err,rad_neg_err,rmsfinal,logg_pos_err,logg_neg_err,badidx,true_mass,tmass_errp,tmass_errn)
	
	# paper_plot(keep,outliers,testlabels,labels_m1,labels_m2,logg_pos_err,logg_neg_err)
	# logg_pos_err,logg_neg_err=[0.15]*len(keep),[-0.15]*len(keep)
	# pplot_outliers_together(keep,outliers,testlabels,labels_m1,labels_m2,logg_pos_err,logg_neg_err,result_hists)

	# exit()
	print(rms)
	print(rad_pos_err,rad_neg_err)
	get_mass_radius_plot(oparams[0],rms,radii,mass,infer_logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)
	exit()
	# get_mass_lum_plot(oparams[0],radii,mass,infer_logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)
	# get_mass_outliers(oparams[0],radii,mass,infer_logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)
	# get_hr_plot(oparams[0],radii,mass,infer_logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)
	# compare_travis_mass(oparams[0],radii,mass,infer_logg,true_logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)
	exit()



main(0)

