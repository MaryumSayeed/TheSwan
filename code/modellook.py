import pickle, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol
from astropy.io import ascii
from astropy.io import fits
from scipy.stats import chisquare
#print(plt.rcParams.keys())
plt.rcParams['xtick.labelsize']=7
plt.rcParams['ytick.labelsize']=7
plt.rcParams['axes.labelsize']='small'
plt.rcParams['ytick.major.width'] = 0
plt.rcParams['ytick.minor.width'] = 0
plt.rcParams['ytick.major.pad']='0'
plt.rcParams['xtick.major.pad']='0'
getonepdf=False
d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/pdfs/'

'''files=glob.glob(d+'*.pdf')
a=[]
for f in files:
	file=f.split('/')[-1]
	if file[0] is '4' or file[0] is '5':
		b=file.split('.pdf')[0]
		a.append(float(b))'''

# if getonepdf is True:
# 	pdfs=[3.34689, 3.18328, 3.38047, 3.13354, 3.32002, 3.41434, 3.18208, 3.08309, 3.24494, 3.43897, 3.10187, 3.04602, 3.04149, 3.0407, 3.16581, 3.23413, 3.30362, 3.07028, 3.28768, 3.03921, 3.01687, 3.19369, 3.16328, 3.28858, 3.22932, 3.29616, 3.10295, 3.21392, 3.31912, 3.22126, 3.04887, 3.18387, 3.33681, 3.35233, 3.2456, 3.25415, 3.1459, 3.17014, 3.13425, 3.23899, 3.24337, 3.24673, 3.20623, 3.16708, 3.1314, 3.08842, 3.09805, 3.04119, 3.06385, 3.07853, 3.07621, 3.015, 3.01204, 3.01932, 3.21142, 3.33733, 3.16926, 3.26263, 3.15183, 3.02131, 3.09022, 3.28585, 3.25724, 3.32499, 3.21568, 3.14694, 3.19342, 3.26604, 3.02304, 3.21622, 3.35429, 3.10867, 3.10415, 3.0485, 3.11922, 3.17635, 3.14813, 3.25707, 3.23334, 3.0797, 3.07484, 3.185, 3.20039, 3.36164, 3.00551, 3.09054, 3.27064, 3.05632, 3.30744, 3.08973, 3.11832, 3.34063, 3.00613, 3.11021, 3.0963, 3.05417, 3.27865, 3.00405, 3.17934, 3.18567, 3.30637, 3.15726, 3.34334, 3.1663, 3.21576, 3.04113, 3.1639, 3.13606, 3.30125, 3.03329, 3.1953, 3.07747, 3.14266, 3.08676, 3.26737, 3.27059, 3.02374, 3.03281, 3.12563, 3.05931]
# 	print(sorted(pdfs))
# 	merger = PdfFileMerger()
# 	for pdf in sorted(pdfs):
# 		p=d+'{}.pdf'.format(pdf)
# 		merger.append(p)
# 	merger.write(d+"RG_below3.5.pdf")
# 	exit()

# Load all files:
a=open('model_real_unweighted.pickle','rb')
model=pickle.load(a)
a.close()

a=open('test_realifft_unweighted.pickle','rb')
testdata=pickle.load(a)
a.close()
freq=np.arange(10.011574074074073,277.76620370370375,0.01157)
print(freq)

a=open('coeffs_real_unweighted.pickle','rb')
coeffs_file=pickle.load(a)
a.close()
dataall2, metaall, labels, schmoffsets, coeffs, covs, scatters, chis, chisqs=coeffs_file

pf=open('testmetaall_lum.pickle','rb') 
testmetaall,truemetaall=pickle.load(pf)
pf.close() 

kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
allkps      =list(df['kic_kepmag'])

goodfile='/Users/maryumsayeed/Desktop/HuberNess/mlearning/ACFcannon-master/RG_below3.5.txt'
badfile='/Users/maryumsayeed/Desktop/HuberNess/mlearning/ACFcannon-master/RG_below3.5.txt'
goodfile=np.loadtxt(goodfile,dtype=str)
badfile=np.loadtxt(badfile,dtype=str)
print('# of good stars',len(goodfile))
print('# of bad stars',len(badfile))

goodidx=[]
for file in testdata[-1]:
	if file in badfile:
		idx=testdata[-1].index(file)
		goodidx.append(idx)

testdatazeros=[]
testdatazeros.append([testdata[0][:,i,:] for i in goodidx])
testdatazeros.append([testdata[1][i] for i in goodidx])
testdatazeros.append(testdata[2])
testdatazeros.append([testdata[3][i] for i in goodidx])
testdatazeros.append([testdata[4][i] for i in goodidx])
model      =[model[i] for i in goodidx]
testdata   =testdatazeros
testmetaall=[testmetaall[i] for i in goodidx]
truemetaall=[truemetaall[i] for i in goodidx]

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
pdfs=[]
allms=[]

# Get loggs:
loggs=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_loggs.txt',skiprows=1,delimiter=',',usecols=[0,1])
logg_kic1=loggs[:,0]
logg_vals1=loggs[:,1]
loggs=np.genfromtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_loggs_2.txt',skip_header=True,delimiter=',',usecols=[0,1,4])#,dtype=['i4','f4','f4'])
logg_kic2=loggs[:,0]
logg_vals2=loggs[:,1]
logg_kic=list(logg_kic1)+list(logg_kic2)
logg_vals=list(logg_vals1)+list(logg_vals2)
logg_kic=[int(i) for i in logg_kic]


print('# of stars to save as pdf:',len(model))
for star in range(0,len(model)):
	plt.figure(figsize=(15,8))
	grid = plt.GridSpec(3, 4, wspace=0.2, hspace=0.2)
	gaia=ascii.read('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/DR2PapTable1.txt',delimiter='&')
	psfile=testdata[-1][star]	
	kicid=psfile.split('/')[-1].split('-')[0].split('kplr')[-1]
	#file='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/data/kplr{}-2011177032512_llc.fits'.format(kicid)
	kicid=int(kicid.lstrip('0'))
	kp =allkps[kp_kics.index(kicid)]
	
	idx=logg_kic.index(kicid)
	logg=logg_vals[idx]

	print(kicid)#,logg)
	file='/Users/maryumsayeed/Desktop/pande/pande_lcs/'+psfile.split('/')[-1][0:-3]
	

	#kicid=3115435
	#file='/Users/maryumsayeed/Desktop/test/kplr00{}-2011177032512_llc.fits'.format(kicid)
	
	data=fits.open(file)
	head=data[0].data
	dat=data[1].data
	time=dat['TIME']
	qual=dat['SAP_QUALITY']
	flux=dat['PDCSAP_FLUX']
	

	um=np.where(gaia['KIC'] == kicid)[0]
	#if (len(um) == 0.):
	#	continue
	kics=np.zeros(1,dtype='int')
	teffs=np.zeros(1)
	lums=np.zeros(1)
	i=0
	kics[i]=kicid
	teffs[i]=gaia['teff'][um[0]]
	rad=gaia['rad'][um[0]]
	lums[i]=rad**2.*(teffs[i]/5777.)**4.
	numax = 3090.*((10.**logg)/(10.**4.44) * (teffs[i]/5777.)**(-0.5))
	numax = np.log10(numax)
	
	good=np.where(qual == 0)[0]
	time=time[good]
	flux=flux[good]

	# plot the light curve
	plt.subplot(grid[0,2:])
	plt.plot(time,flux)
	plt.xlabel('Time (Days)')
	plt.ylabel('Flux (counts)')
	plt.xticks(fontsize=7)
	plt.yticks(fontsize=7)
	
	# sigma-clip outliers from the light curve and overplot it
	res=sigclip(time,flux,50,3)
	good = np.where(res == 1)[0]
	time=time[good]
	flux=flux[good]
	plt.plot(time,flux)

	# next, run a filter through the data to remove long-periodic (low frequency) variations

	width=2.
	boxsize=width/(30./60./24.)
	box_kernel = Box1DKernel(boxsize)
	smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
	# overplot this smoothed version, and then divide the light curve through it
	plt.plot(time,smoothed_flux)

	flux=flux/(smoothed_flux)
	 
	#flux=flux+np.random.randn(len(flux))*0.01

	# plot the filtered light curve
	plt.subplot(grid[1,2:])
	plt.plot(time,flux)
	plt.xlabel('Time (Days)')
	plt.ylabel('Relative flux')
	plt.xticks(fontsize=7)
	plt.yticks(fontsize=7)

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
	 
	# smooth by 2 muHz
	n=np.int(2./fres_mhz)
	 
	gauss_kernel = Gaussian1DKernel(n)
	pssm = convolve(amp, gauss_kernel)
	 
	um=np.where(freq > 10.)[0]
	start=np.where(freq>10)[0][0]
	#end=np.where(freq>50)[0][-1]
	start=864
	end=21000+start
	#print(start,end)

	toplot=[10.**(testdata[0][star][i][1]) for i in range(0,end-start)]
	#print(amp[864:864+3])
	#print(freq)
	#print(len(testdata[0][star][:][1]))
	#print(toplot[0:3])
	#print(amp[0:3])
	plt.subplot(grid[2,2:])
	plt.loglog(freq,amp,c='k',linewidth=1)
	#plt.loglog(freq,pssm,c='r',label='smoothed')
	plt.xlabel('Frequency ($\mu$Hz)')
	plt.ylabel('Power Density')
	plt.axvspan(freq[start],freq[end],alpha=0.2,color='green',label='bins used for training')
	plt.legend(loc='upper right',fontsize = 'x-small')
	plt.xlim([0.02,300])
	plt.ylim([0.01,1e8])

	# plot the power spectrum
	#plt.axvline(freq[3000],c='g',linestyle='dashed')
	plt.subplot(grid[0:3,0:2])
	plt.loglog(freq,amp,c='grey',alpha=0.4)
	plt.loglog(freq,pssm,c='k',label='smoothed-quicklook',linestyle='dashed')#,linewidth=5,alpha=0.5)
	plt.loglog(freq[start:end],toplot,label='smoothed-cannon',c='cyan',linestyle='dashed')
	plt.loglog(freq[start:end],10.**model[star][0:],label='model',c='r',linewidth=1)
	yfit=model[star][0:]
	dyfit=scatters
	plt.fill_between(freq[start:end], 10**(yfit - dyfit), 10**(yfit + dyfit),color='red',alpha=0.2,zorder=100)
	plt.xlabel('Frequency ($\mu$Hz)',fontsize=14)
	plt.ylabel('Power Density',fontsize=14)
	plt.xlim([9.,300])
	plt.ylim([0.01,1e6])
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	logg=round(logg,2)
	numax=round(numax,2)
	chi2,_=chisquare(10.**model[star][0:],toplot)
	chi2=np.log10(chi2)
	print(chi2)
	plt.suptitle('$X^2$='+str(round(chi2,2))+'    teff='+str(teffs[i])+'    logg='+str(logg)+r'    $\nu_{max}=$'+str(numax)+'    lum='+str(round(lums[i],2))+'    kp='+str(round(kp,2))+'    kicid:'+str(kicid),y=0.92)
	#a=round(10.**truemetaall[star][0],3)
	#b=round(10.**testmetaall[star][0],3)
	#ae=round(truemetaall[star][0],2)
	#be=round(testmetaall[star][0],2)
	#ldot=' L$_{\odot}$'
	#plt.text(11,10**5.5,s='true:  '+str(a)+ldot+' ({})'.format(ae))
	#plt.text(11,10**5.,s='pred: '+str(b)+ldot+' ({})'.format(be))
	a=round(truemetaall[star][0],3)
	b=round(testmetaall[star][0],3)
	predkp=round(testmetaall[star][1],2)
	truekp=round(kp,2)
	plt.text(11,10**5.5,s='true: '+str(a))#+ldot)
	plt.text(11,10**5.,s='pred: '+str(b))#+ldot)
	plt.text(20,10**5.5,s='true kp: '+str(truekp))#+ldot)
	plt.text(20,10**5.,s='pred kp: '+str(predkp))#+ldot)
	
	#plt.tight_layout()
	#plt.draw()
	plt.legend()
	#plt.show()
	#exit()
	#p=round(10.**truemetaall[star][0],4)
	a=round(truemetaall[star][0],5)
	p=a
	pdfs.append(p)
	#pdfs.append('{}.format(a,b))
	plt.savefig('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/pdfs/'+'{}.pdf'.format(p))
	#allms.append(10**model[star][0:4])
	#plt.show()
	plt.clf()
	
	#plt.savefig('wwww')
#np.savetxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/pdfs/ms.txt',allms)
print(pdfs)

exit()






# Number of plots on each page (in groups of 16):
lims=[[0,20],[20,40],[40,60],[60,80],[80,100]]
rows,cols=4,5

pdfs=[]
for i in lims:
	fig, axs = plt.subplots(rows,cols, figsize=(20, 8), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = .5, wspace=.05)
	axs = axs.ravel()
	for star in range(i[0],i[1]):
		file=testdata[-1][star]
		#print(file)
		kic=file.split('/')[-1].split('-')[0].split('kplr')[-1]
		kic=int(kic.lstrip('0'))
		pred,true=testmetaall[star],truemetaall[star]
		a=abs((pred[0]-true[0])/true[0])<0.1
		#if pred[0] < -0.4:
		print(file,true,pred)
		axs[star-i[0]].plot(freq[0:fbins],model[star][0:fbins],label='model',c='r')
		axs[star-i[0]].plot(freq[0:fbins],testdata[0][0:fbins,star,1],label='og data',c='k')
		axs[star-i[0]].set_yscale('log')
		axs[star-i[0]].set_xscale('log')
		#axs[star-i[0]].set_ylim([1.,10.])
		#axs[star-i[0]].tick_params(axis='y', which='major', pad=0)
		#if star-i[0] not in [0,4,8]:
		#	axs[star-i[0]].get_yaxis().set_visible(False)
		
		axs[star-i[0]].set_title('true: {} pr: {} ({}-{})'.format(true[0],pred[0],a,kic),fontsize=8)
		print(star)
		
		#else:
		#	continue	
	exit()
	name='modelvsdata_{}.pdf'.format(i)
	pdfs.append(name)
	fig.savefig(name)

# Merge all PDF files into one:
merger = PdfFileMerger()
for pdf in pdfs:
    merger.append(pdf)
merger.write("all_modelvsdata.pdf")
