# analyze LLR results

import numpy as np
import time, re
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


plt.rc('font', family='serif')
plt.rc('text', usetex=True)
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
	plt.clf()
	print('-- Stars < {} in PSD vs. Rad plot:'.format(diff),len(keep))
	return keep

def remove_high_chi2(chi2_vals):
	plt.hist(chi2_vals,bins=100)
	plt.xlim([0,500])
	plt.show(False)
	plt.clf()
	cutoff=300
	keep=np.where(chi2_vals<cutoff)[0]
	print('-- Stars with chi2 < {}:'.format(cutoff),len(keep))
	return keep


def final_result(keep1,keep2,true,labelm1,labelm2):
	keep=list(set(keep1) & set(keep2))
	print('-- Final stars:',len(keep))
	check_idx=np.where(abs(true[keep]-labelm1[keep])>0.6)[0]
	#plt.figure(figsize=(10,5))
	plt.subplot(121)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true[keep],labelm1[keep],facecolors='grey', edgecolors='k',label='Model 1 ({})'.format(len(true)),s=10)
	plt.scatter(true[keep][check_idx],labelm1[keep][check_idx],facecolors='r',s=10)
	plt.xlabel('Gaia Logg (dex)')
	plt.ylabel('Inferred Logg (dex)')
	plt.xlim([2,4.8])
	plt.legend()
	bias,rms=returnscatter(true[keep]-labelm1[keep])
	
	print('Model 1 --','RMS:',rms,'Bias:',bias)
	plt.subplot(122)
	plt.plot(true,true,c='k',linestyle='dashed')
	plt.scatter(true[keep],labelm2[keep],facecolors='grey', edgecolors='k',label='Model 2 ({})'.format(len(true)),s=10)
	plt.scatter(true[keep][check_idx],labelm2[keep][check_idx],facecolors='r',s=10)
	plt.xlabel('Gaia Logg (dex)')
	plt.xlim([2,4.8])
	plt.legend()
	bias,rms=returnscatter(true[keep]-labelm2[keep])
	print('Model 2 --','RMS:',rms,'Bias:',bias)
	plt.show(False)
	plt.clf()
	keep=np.array(keep)
	investigate=keep[check_idx]
	return keep,check_idx


def paper_plot(keep,true,labelm1,labelm2,logg_pos_err,logg_neg_err):
	plt.figure(figsize=(6,8))
	labelm1=np.array(labelm1)
	ba,rmsa=returnscatter(labelm1[keep]-true[keep])
	ms=20

	# AFTER PLOT:
	gs = gridspec.GridSpec(4, 4,hspace=0)

	ax1 = plt.subplot(gs[0:3, 0:4])
	ax1.plot(true[keep],true[keep],c='k',linestyle='dashed')
	ax1.errorbar(true[keep],labelm1[keep],xerr=[logg_neg_err*-1,logg_pos_err],ecolor='lightcoral',markeredgecolor='k',markerfacecolor='grey',ms=4,fmt='o')
	ax1.minorticks_off()
	locs, labels = plt.yticks()
	plt.yticks([2.5,3.0,3.5,4.0,4.5,5.0], [2.5,3.0,3.5,4.0,4.5,5.0], fontsize=20)
	plt.xticks([])
	ax1.set_ylabel('Inferred Logg (dex)',fontsize=20)
	ax1.set_xlim([2.,4.8])
	ax1.set_ylim([2.,4.8])
	ax1.text(2.1,4.6,'RMS: '+str(round(rmsa,2)),fontsize=30,ha='left',va='center')
	ax1.text(2.1,4.4,'Bias: '+str(round(ba,4)),fontsize=30,ha='left',va='center')


	ax2 = plt.subplot(gs[3:4, 0:4])
	ax2.scatter(true[keep],true[keep]-labelm1[keep],edgecolor='k',facecolor='grey')
	ax2.axhline(0,c='k',linestyle='dashed')
	ax2.set_xlabel('Gaia Logg (dex)',fontsize=20)
	ax2.set_ylabel('True-Inferred Logg (dex)',fontsize=15)
	ax2.minorticks_off()
	plt.xticks(fontsize=20)
	plt.yticks([-1,0,1], [-1,0,1], fontsize=20)
	ax2.tick_params(which='major',axis='y',pad=1)
	ax2.set_xlim([2.,4.8])
	
	stda=mad_std(labelm1[keep]-true[keep])
	print('Stats after:')
	print(ba,rmsa,stda)
	text_font={'color':'red','weight':'heavy'}
	plt.tight_layout()
	plt.savefig('pande_final_residual.pdf',dpi=50)
	plt.show(False)
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

def get_mass_error(radii,mass,logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err):
	abs_pos_err=(((logg_pos_err/logg)**2.+((rad_pos_err*2)/(radii**2.))**2.)**0.5)
	abs_neg_err=(((logg_neg_err/logg)**2.+((rad_neg_err*2)/(radii**2.))**2.)**0.5)
	rel_pos_err=abs_pos_err*mass
	rel_neg_err=abs_neg_err*mass
	return rel_pos_err,rel_neg_err

def get_mass(keep,true,labelm1,labelm2,allfiles):
	radii=np.zeros(len(keep))
	mass =np.zeros(len(keep))
	rad_pos_err=np.zeros(len(keep))
	rad_neg_err=np.zeros(len(keep))
	logg_pos_err=np.zeros(len(keep))
	logg_neg_err=np.zeros(len(keep))
	grav_const  =6.67*10.**(-8.)   #cm^3*g^-1*s^-2
	solar_radius=6.956*10.**(10.)  #cm
	solar_mass  =1.99*10.**33.     #g
	for i in range(0,len(keep)):
		star=keep[i]
		file=allfiles[i]
		kic=re.search('kplr(.*)-', file).group(1)
		kic=int(kic.lstrip('0'))
		idx=np.where(gaia['KIC']==kic)[0]
		r     =gaia['rad'][idx][0]
		r_errp=gaia['radep'][idx][0]
		r_errn=gaia['radem'][idx][0]
		row   =kepler_catalogue.loc[kepler_catalogue['KIC']==kic]
		logg  =row['iso_logg'].item()
		logg_errp=row['iso_logg_err1']
		logg_errn=row['iso_logg_err2']	

		logg=labelm2[star]
		g=10.**logg              #cm/s^2
		m=g*((r*solar_radius)**2.)/grav_const
		mass[i]  =m/solar_mass
		radii[i] =r
		rad_pos_err[i]=r_errp
		rad_neg_err[i]=r_errn
		logg_pos_err[i]=logg_errp
		logg_neg_err[i]=logg_errn
	
	logg=true[keep]
	return radii,mass,logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err

def get_mass_radius_plot(radii,mass,logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err):
	mass_errp,mass_errn=get_mass_error(radii,mass,logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)
	
	# definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.6
	spacing = 0.015

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom + height + spacing, width, 0.2]
	rect_histy = [left + width + spacing, bottom, 0.2, height]

	# start with a rectangular Figure
	plt.figure(figsize=(12, 12))
	ax_scatter = plt.axes(rect_scatter)
	ax_scatter.tick_params(direction='in', top=True, right=True)
	ax_histx = plt.axes(rect_histx)
	ax_histx.tick_params(direction='in', labelbottom=False,labeltop=True)
	ax_histy = plt.axes(rect_histy)
	ax_histy.tick_params(direction='in', labelleft=False,labelright=True)
	xy = np.vstack([mass,radii])
	z = gaussian_kde(xy)(xy)
	
	
	#plt.figure(figsize=(13,10))
	#gs  = gridspec.GridSpec(6, 4)
	#ax1 = plt.subplot(gs[2:6, 0:4]) #spans 4 rows and 4 columns

	# Scatter plot
	ax_scatter.errorbar(mass,radii,ms=10,fmt='o',xerr=[mass_errn,mass_errp],yerr=[rad_neg_err,rad_pos_err],markerfacecolor='none',markeredgecolor='none',ecolor='lightgrey')
	ax_scatter.scatter(mass,radii,s=20,c=z,zorder=10)
	ax_scatter.set_xlim([0,7])
	ax_scatter.set_ylim([0.5,5])
	rdot=' R$_{\\odot}$'
	mdot='M$_{\\odot}$'
	ax_scatter.set_xlabel('Mass ({})'.format(mdot),fontsize=20)
	ax_scatter.set_ylabel('Radius ({})'.format(rdot),fontsize=20)
	plt.gca().invert_xaxis()
	#plt.xticks(fontsize=20)
	#plt.yticks(fontsize=20)

	#bar=plt.colorbar(im1, ax=ax1)
	#cb_label='$\\log_{10}$(Count)'
	#cbar.set_label(label=cb_label,fontsize=30)
	#cbar.ax.tick_params(labelsize='large')
	

	# definitions for the axes
	binwidth = 0.25
	ax_scatter.set_xlim((4,0))
	ax_scatter.set_ylim((0.5,5))

	mass_bins  =np.arange(0,4,0.1)
	radius_bins=np.arange(0.5,10,0.1)
	lw=2

	ax_histx.hist(mass, bins=mass_bins,edgecolor="k",facecolor='lightgrey',linewidth=lw,histtype=u'step')
	ax_histy.hist(radii, bins=radius_bins, orientation='horizontal',edgecolor="k",facecolor='lightgrey',histtype=u'step',linewidth=lw)

	ax_histx.set_xlim(ax_scatter.get_xlim())
	ax_histy.set_ylim(ax_scatter.get_ylim())

	# Mass-Radius Plot:
	ax_scatter.minorticks_on()
	ax_scatter.yaxis.set_major_locator(ticker.MultipleLocator(1.))
	ax_scatter.grid(which='both',linestyle=':', linewidth='0.5', color='grey',alpha=0.2)
	ax_scatter.tick_params(which='minor', # Options for both major and minor ticks
	                top='False', # turn off top ticks
	                left='False', # turn off left ticks
	                right='False',  # turn off right ticks
	                bottom='False') # turn off bottom ticks
	ax_scatter.tick_params(which='major',labelsize=20)


	# Mass Hist:
	ax_histx.minorticks_on()
	ax_histx.grid(which='both',linestyle=':', linewidth='0.5', color='grey',alpha=0.2)
	ax_histx.yaxis.set_major_locator(ticker.MultipleLocator(100))
	ax_histx.tick_params(which='major', # Options for both major and minor ticks
	                top='True', # turn off top ticks
	                left='True', # turn off left ticks
	                right='False',  # turn off right ticks
	                bottom='False', # turn off bottom ticks
	                labelsize=20)
	ax_histx.tick_params(which='minor',bottom='off',left='off')

	# Radius Hist:
	ax_histy.minorticks_on()
	ax_histy.yaxis.set_major_locator(ticker.MultipleLocator(1.))
	ax_histy.xaxis.set_major_locator(ticker.MultipleLocator(100))
	ax_histy.grid(which='both',linestyle=':', linewidth='0.5', color='grey',alpha=0.2)

	ax_histy.tick_params(which='major', # Options for both major and minor ticks
	                top='False', # turn off top ticks
	                left='False', # turn off left ticks
	                right='True',  # turn off right ticks
	                bottom='True', # turn off bottom ticks
	                labelsize=20)
	ax_histy.tick_params(which='minor',bottom='False',left='False')
	
	plt.tight_layout()
	#plt.savefig('mass_radius.pdf',dpi=100)
	

    # Error Analysis:\n",
    # plt.scatter(mass,radius,s=15,c=radius_err_pos,cmap='Paired',zorder=10)\n",


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
	train_file_names =['pande_pickle_1','pande_pickle_2','pande_pickle_3','astero_final_sample_1','astero_final_sample_2','astero_final_sample_3','astero_final_sample_4']
	train_file_pickle=[i+'_memmap.pickle' for i in train_file_names]
	train_file_txt   =[i+'.txt' for i in train_file_names]

	print('Getting training data...')
	all_labels,all_data,total_stars,all_files=gettraindata(train_file_txt,train_file_pickle)
	all_labels,all_data,all_files=all_labels[start:start+end],all_data[start:start+end],all_files[start:start+end]
	
	#init_plots(testlabels,labels_m1,labels_m2)
	radii,avg_psd=get_avg_psd(all_files,all_data)
	keep_idx_1   =fit_power_radius(radii,avg_psd)
	keep_idx_2   =remove_high_chi2(chi2_vals)
	keep,badidx  =final_result(keep_idx_1,keep_idx_2,testlabels,labels_m1,labels_m2)

	#check_models(keep,badidx,testlabels,labels_m1,labels_m2,spectra_m1,all_data,all_files)
	print('=== original:',len(radii))
	print('=== after cleaning:',len(keep))
	print('=== outliers:',len(badidx))
	print('=== fraction of outliers:',len(badidx)/len(keep))
	
	radii,mass,logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err=get_mass(keep,testlabels,labels_m1,labels_m2,all_files)
	paper_plot(keep,testlabels,labels_m1,labels_m2,logg_pos_err,logg_neg_err)
	exit()
	get_mass_radius_plot(radii,mass,logg,rad_pos_err,rad_neg_err,logg_pos_err,logg_neg_err)
	



main(0)

