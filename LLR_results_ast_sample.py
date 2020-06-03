# analyze LLR results

import numpy as np
import time, re
import pandas as pd
import csv
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

astero1=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/labels_full.txt',delimiter=' ',skiprows=1,usecols=[0,2])
astero2=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/rg_yu.txt',delimiter='|',skiprows=1,usecols=[0,3,4])
astero3=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/mathur_2019.txt',skiprows=14,usecols=[0,2,3,4])
astero3_kics=[int(i[0]) for i in astero3]

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
	plt.xlabel('Gaia Logg [dex]')
	plt.ylabel('inferred Logg [dex]')
	plt.xlim([0,4.8])
	plt.legend()
	bias,rms=returnscatter(true-labelm1)
	print('Model 1 --','RMS:',rms,'Bias:',bias)
	plt.subplot(122)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true,labelm2,facecolors='grey', edgecolors='k',label='Model 2 ({})'.format(len(true)),s=10)
	plt.xlabel('Gaia Logg [dex]')
	plt.xlim([0,4.8])
	bias,rms=returnscatter(true-labelm2)
	print('Model 2 --','RMS:',rms,'Bias:',bias)
	plt.legend()
	plt.show()
    

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
	# plt.subplot(121)
	# plt.loglog(radii,power,'k+')
	# plt.xlabel('Radii (solar)')
	#plt.ylabel('Power (ppm^2/uHz)')

	xdata,ydata=np.log10(radii),np.log10(power)
	offset=0
	'''def fitted_model(x, a, b, c,d):
		x=np.array(x)
		return a*np.sin(b*x+c)+d

	def func(p, x, y):
		x=np.array(x)
		return p[0]*np.sin(p[1]*x+p[2])+p[3]-y'''

	def fitted_model(x, a, b):
	    x=np.array(x)
	    return a*x+b

	def func(p, x, y):
	    return  p[0]*x+p[1]-y
	plt.subplot(121)
	x0 = np.array([2.8,1.])
	res_lsq = least_squares(func, x0, args=(xdata, ydata),max_nfev=2000)
	y_lsq = fitted_model(sorted(xdata), *res_lsq.x)
	plt.scatter(xdata, ydata, c='k',s=5)
	plt.plot(sorted(xdata),y_lsq,c='r')
	print('Function constants are:',res_lsq.x)
	diff=0.8
	keep=np.where(abs(y_lsq-ydata)<diff)[0]
	plt.scatter(xdata[keep],ydata[keep],c='g',s=5)
	plt.grid(True)
	print('auto. fitting:',len(keep))

	plt.subplot(122)
	def man(x,a,b,c,d):
		x=np.array(x)
		return a*np.sin(b*x+c)+d
	a,b,c,d=1.6,2.6,-1.95,3.
	plt.scatter(xdata, ydata, c='k',label="original data",s=5)
	
	def man(x,a,b):
		x=np.array(x)
		return a*x+b
	a,b=2.8,1.
	
	plt.plot(sorted(xdata),man(sorted(xdata),a,b),c='r')
	diff=1.
	keep=np.where((abs(man((xdata),a,b)-ydata)<diff))[0]
	print('manual fitting',len(keep))
	plt.scatter(xdata[keep],ydata[keep],c='g',s=5)
	plt.grid(True)
	plt.show(False)
	
	#plt.clf()
	print('-- Stars < {} in PSD vs. Rad plot:'.format(diff),len(keep))

	return keep

def remove_high_chi2(chi2_vals):
	plt.hist(chi2_vals,bins=100)
	plt.xlim([0,500])
	plt.show(False)
	plt.clf()
	cutoff=400
	keep=np.where(chi2_vals<cutoff)[0]
	print('-- Stars with chi2 < {}:'.format(cutoff),len(keep))
	print('-- Stars with chi2 > {}:'.format(cutoff),len(np.where(chi2_vals>cutoff)[0]))
	return keep


def final_result(keep1,keep2,true,labelm1,labelm2):
	keep=list(set(keep1) & set(keep2))
	_,rms=returnscatter(labelm1[keep]-true[keep])
	print('-- Final stars:',len(keep))
	offset=3*rms
	print('RMS:',rms,'Outlier cutoff:',offset)
	
	check_idx=np.where(abs(true[keep]-labelm1[keep])>offset)[0]
	#plt.figure(figsize=(10,5))
	plt.subplot(121)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true[keep],labelm1[keep],facecolors='grey', edgecolors='k',label='Model 1 ({})'.format(len(keep)),s=10)
	plt.scatter(true[keep][check_idx],labelm1[keep][check_idx],facecolors='r',s=10)
	plt.xlabel('Asteroseismic Logg [dex]')
	plt.ylabel('Inferred Logg [dex]')
	plt.xlim([0.,4.8])
	plt.legend()
	bias,rms=returnscatter(true[keep]-labelm1[keep])
	
	print('Model 1 --','RMS:',rms,'Bias:',bias)
	plt.subplot(122)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true[keep],labelm2[keep],facecolors='grey', edgecolors='k',label='Model 2 ({})'.format(len(keep)),s=10)
	plt.scatter(true[keep][check_idx],labelm2[keep][check_idx],facecolors='r',s=10)

	plt.xlabel('Asteroseismic Logg [dex]')
	plt.xlim([0.,4.8])
	plt.legend()
	bias,rms=returnscatter(true[keep]-labelm2[keep])
	print('Model 2 --','RMS:',rms,'Bias:',bias)
	
	plt.show(False)
	
	keep=np.array(keep)
	outliers=keep[check_idx]
	print(outliers)
	print(true[outliers])
	return keep,check_idx,offset,outliers

def investigate_outliers(outliers,keep,testfiles,true,labelm1):
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
		kic_idx=np.where(gaia['KIC']==kic)
		t=gaia['teff'][kic_idx][0]
		r=gaia['rad'][kic_idx][0]
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
	    kic_idx=np.where(gaia['KIC']==kic)
	    t=gaia['teff'][kic_idx][0]
	    r=gaia['rad'][kic_idx][0]
	    l=r**2.*(t/5777.)**4.
	    tr=true[star]
	    pr=labelm1[star]
	    bad_stars_lum.append(l)
	    bad_stars_kp.append(kp)
	    bad_stars_teff.append(t)
	    bad_stars_rad.append(r)
	    bad_stars_true_logg.append(tr)
	    bad_stars_pred_logg.append(pr)

	_,rms=returnscatter(labelm1[keep]-true[keep])
	cutoff=3*rms
	print('cutoff',cutoff)
	fig=plt.figure(figsize=(8,8))
	plt.rc('font', size=15)                  # controls default text sizes
	plt.rc('axes', titlesize=12)             # fontsize of the axes title
	plt.rc('axes', labelsize=14)             # fontsize of the x and y labels
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
	plt.hist(good_stars_kp,bins=binsk,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_kp,bins=binsk,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel('Kp')

	plt.subplot(323)
	binst=np.linspace(3800.,6800.,20)
	plt.hist(good_stars_teff,bins=binst,density=True,edgecolor=good_c,facecolor='none',linewidth=2)
	plt.hist(bad_stars_teff,bins=binst,density=True,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed')
	plt.xlabel(r'T$_\mathrm{eff}$ [K]')
	plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

	plt.subplot(324)
	binsr=np.linspace(0.,40.,bins)
	plt.hist(good_stars_rad,density=True,bins=binsr,edgecolor=good_c,facecolor='none',linewidth=2,label='$\Delta \log g < 3\sigma$')
	plt.hist(bad_stars_rad,density=True,bins=binsr,edgecolor=bad_c,facecolor='none',linewidth=2,linestyle='dashed',label='$\Delta \log g > 3\sigma$')
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
	# plt.savefig('astero_outliers.pdf')
	plt.show(False)
	exit()

def paper_plot(keep,true,labelm1,labelm2,logg_pos_err,logg_neg_err):
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
	locs, labels = plt.yticks()
	newlabels=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
	plt.yticks(newlabels, newlabels)
	ax1.set_xticklabels([]*len(newlabels))
	plt.minorticks_on()
	ax1.set_ylabel('Inferred Logg [dex]')
	ax1.set_xlim([1.,4.7])
	ax1.set_ylim([1.,4.7])
	ax1.text(1.1,4.5,'RMS: '+str(round(rmsa,2)),ha='left',va='center')
	ax1.text(1.1,4.2,'Bias: '+str('{0:.2f}'.format(ba)),ha='left',va='center')

	ax2 = plt.subplot(gs[3:4, 0:4])
	ax2.scatter(true[keep],true[keep]-labelm1[keep],edgecolor='k',facecolor='grey',s=20)
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
	# plt.savefig('/Users/maryumsayeed/Desktop/HuberNess/iPoster/astero_final_residual.pdf',dpi=50)
	plt.show(False)
	#exit()

def step_by_step_plot(keep,true,labelm1,labelm2,models,alldata,allfiles):
	good_idx=np.where((abs(true[keep]-labelm1[keep])<0.02) & (true[keep]>2) & (true[keep]<2.5))[0]
	print(len(good_idx))
	
	alldata=np.array(alldata)[keep]
	print('---done loading array.')
	models=models[keep]
	# exit()
	for star in good_idx[60:70]:
		file=allfiles[keep][star][0:-3]
		print(star,file)
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

		fig=plt.figure(figsize=(6,8))

		fs=10
		ms  = 10
		av  = 0.5
		plt.rc('font', size=12)                  # controls default text sizes
		plt.rc('axes', titlesize=15)             # fontsize of the axes title
		plt.rc('axes', labelsize=14)             # fontsize of the x and y labels
		plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
		plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
		plt.rc('axes', linewidth=1)  
		plt.rc('lines', linewidth=1)  
		plt.rc('legend', fontsize=15)

		c2  ='#404788FF'
		c1  ='lightcoral'
		c3  ='#55C667FF'
		ax1 = plt.subplot(311)
		ax1.plot(time_1,flux_1,label='Original',c=c1)
		ax1.plot(time_2,flux_2,label='Outliers removed',c=c2)
		ax1.plot(time_3,flux_3,label='Long-periodic variations removed',c=c3)
		# ax1.set_title('KICID: {}'.format(str(kic)))
		ax1.set_xlabel('Time [Days]')
		ax1.set_ylabel('Flux [Counts]')

		# y_fmt = ticker.FormatStrFormatter('%1.1e')
		# ax1.yaxis.set_major_formatter(y_fmt)
		ax1.ticklabel_format(axis='y',style='sci',scilimits=(1,4))

		# y_formatter = ticker.ScalarFormatter(useOffset=True)
		# ax1.yaxis.set_major_formatter(y_formatter)

		# plt.yticks(fontsize=yticksize)

		ax2 = plt.subplot(312,sharex=ax1)
		ax2.plot(time,flux,c='k')
		ax2.set_xlabel('Time [Days]')
		ax2.set_ylabel('Normalized Flux')

		y_fmt = ticker.FormatStrFormatter('%0.3f')
		ax2.get_yaxis().get_major_formatter().set_useOffset(True)
		ax2.yaxis.set_major_formatter(y_fmt)
		# ax2.ticklabel_format(axis='y',style='sci',useOffset=True)#scilimits=(-1,1))

		ax3 = plt.subplot(313)
		nbins=21000
		ax3.loglog(f,amp,c='k')
		ax3.loglog(f,pssm_wnoise,c=c3,label='Smoothed',linewidth=2)
		ax3.axvspan(f[864],f[864+nbins], facecolor='white', alpha=0.2,zorder=10)
		newfreq=f[864:864+nbins]
		ax3.set_xlim([10,255])
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
			plt.savefig('{}_new.pdf'.format(kic),dpi=50)
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
			logg_errp=astero3[idx,2]
			logg_errn=astero3[idx,3]
		elif kic in astero2:
			idx=np.where(np.asarray(astero2[:,0])==kic)[0]
			logg=astero2[idx,1][0]
			logg_errp=astero2[idx,2][0]
			logg_errn=astero2[idx,2][0]
		logg_pos_err[i]=logg_errp
		logg_neg_err[i]=logg_errn
		if i in counting: print('Logg error bars found for {} stars.'.format(i))
	
	return logg_pos_err,logg_neg_err

def get_table(keep,allfiles,testlabels,labels_m1,logg_pos_err,logg_neg_err):
	kics=np.zeros(len(keep))
	teffs=np.zeros(len(keep))
	true_logg=np.zeros(len(keep))
	infer_logg=np.zeros(len(keep))
	radii=np.zeros(len(keep))
	rad_pos_err=np.zeros(len(keep))
	rad_neg_err=np.zeros(len(keep))
	for i in range(0,len(keep)):
		star=keep[i]
		file=allfiles[star]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		idx=np.where(gaia['KIC']==kic)[0]
		t     =gaia['teff'][idx][0]
		r     =gaia['rad'][idx][0]
		r_errp=gaia['radep'][idx][0]
		r_errn=gaia['radem'][idx][0]
		kics[i]  =kic
		teffs[i] =t
		radii[i] =r
		rad_pos_err[i]=r_errp
		rad_neg_err[i]=r_errn
		true_logg[i]=testlabels[star]
		infer_logg[i]=labels_m1[star]

	kics,teffs=[int(i) for i in kics],[int(i) for i in teffs]
	header = ['KICID', 'Teff', 'Radius','Radp','Radn','True_Logg','Loggp','Loggn','Inferred_Logg'] 
	with open('Astero_Final_Catalogue.txt', 'w') as f:
		w = csv.writer(f, delimiter=';')
		w.writerow(header)
		for row in zip(kics,teffs,radii,rad_pos_err,rad_neg_err,true_logg,logg_pos_err,logg_neg_err,infer_logg):
			w.writerow(row)
	
	df = pd.read_csv('Astero_Final_Catalogue.txt',index_col=False,delimiter=';')
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
	dirr='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/jan2020_astero_sample/'
	average=np.load(dirr+'average.npy')
	end=len(average)
	testlabels=np.load(dirr+'testlabels.npy')[start:]
	print(len(testlabels))
	labels_m1 =np.load(dirr+'labels_m1.npy')
	labels_m2 =np.load(dirr+'labels_m2.npy')
	spectra_m1=np.load(dirr+'spectra_m1.npy')
	chi2_vals =np.load(dirr+'min_chi2.npy')
	
	print('Size of saved data:',len(testlabels),len(average),len(labels_m1),len(labels_m2),len(spectra_m1))
	train_file_names =['pande_pickle_1','pande_pickle_2','pande_pickle_3','astero_final_sample_1','astero_final_sample_2','astero_final_sample_3','astero_final_sample_4']
	train_file_pickle=[i+'_memmap.pickle' for i in train_file_names]
	train_file_txt   =[i+'.txt' for i in train_file_names]

	print('Getting training data...')
	all_labels,all_data,total_stars,all_files=gettraindata(train_file_txt,train_file_pickle)
	all_labels,all_data,all_files=all_labels[start:start+end],all_data[start:start+end],all_files[start:start+end]
	
	#init_plots(testlabels,labels_m1,labels_m2)

	radii,avg_psd    =get_avg_psd(all_files,all_data)
	keep_idx_1       =fit_power_radius(radii,avg_psd)
	keep_idx_2       =remove_high_chi2(chi2_vals)
	keep,badidx,diff,outliers =final_result(keep_idx_1,keep_idx_2,testlabels,labels_m1,labels_m2)
	investigate_outliers(outliers,keep,all_files,testlabels,labels_m1)
	exit()
	#check_models(keep,badidx,testlabels,labels_m1,labels_m2,spectra_m1,all_data,all_files)

	print('=== original:',len(radii))
	print('=== after cleaning:',len(keep))
	print('=== outliers (diff>{}):'.format(diff),len(badidx))
	print('=== fraction of outliers:',len(badidx)/len(keep))

	# np.save('index_of_good_stars.npy',keep)
	# exit()
	step_by_step_plot(keep,testlabels,labels_m1,labels_m2,spectra_m1,all_data,all_files)
	exit()
	logg_pos_err,logg_neg_err=get_logg_error(keep,all_files)
	# logg_pos_err,logg_neg_err=[0]*len(keep),[0]*len(keep)
	#get_table(keep,all_files,testlabels,labels_m1,logg_pos_err,logg_neg_err)
	# paper_plot(keep,testlabels,labels_m1,labels_m2,logg_pos_err,logg_neg_err)
	exit()
	
	



main(6135)

