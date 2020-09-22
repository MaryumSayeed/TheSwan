import numpy as np
import matplotlib.pyplot as plt
import pickle, re, glob
from astropy.io import fits
from astropy.io import ascii
import pandas as pd
from astropy.stats import mad_std

from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol
import matplotlib.ticker as tck


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
    #kp=kps[kp_kics.index(kic)]
    fname=file+'.ps'
    file_idx=np.where(LLR_files==fname)[0]
    kp=cannon_kp[file_idx]
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

def get_amp(file):
	'''Given star's filename, get its frequency &
	white noise corrected power spectral density.'''
	file=file[0]
	file=file[0:-3]
	kicid=int(file.split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
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

	return freq,amp

# Get Kps for all stars:
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
kps      =list(df['kic_kepmag'])
kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])
    
# choose x stars across logg range 
# get their LLR model 
# generate their possible Cannon model
# plot on PSD vs. Freq plot

# Load Cannon files & results:
d1='cannon_vs_LLR/one_label/'
d2='cannon_vs_LLR/two_labels/'
cannon_test_files=np.loadtxt(d1+'Cannon_test_stars.txt',usecols=[0],dtype='str')
pf=open(d1+'testmetaall_cannon_logg.pickle','rb') 
testmetaall,truemetaall=pickle.load(pf)
pf.close() 
cannon_true=truemetaall[:,0]
cannon_infer=testmetaall[:,0]

cannon_test_files=np.loadtxt(d2+'Cannon_test_stars.txt',usecols=[0],dtype='str')
pf=open(d2+'testmetaall_cannon_logg.pickle','rb') 
testmetaall,truemetaall=pickle.load(pf)
pf.close() 
cannon_kp=truemetaall[:,1]

# Load LLR files & results:
LLR_astero_files=np.loadtxt('LLR_seismic/astero_final_sample_full.txt',usecols=[0],dtype='str')
LLR_pande_files=np.loadtxt('LLR_gaia/pande_final_sample_full.txt',usecols=[0],dtype='str')

# Save LLR KICs:
LLR_astero_kics=[]
for file in LLR_astero_files:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	LLR_astero_kics.append(kic)

LLR_pande_kics=[]
for file in LLR_pande_files:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	LLR_pande_kics.append(kic)

LLR_pande_kics=np.array(LLR_pande_kics)
LLR_astero_kics=np.array(LLR_astero_kics)

LLR_pande_true=np.load('LLR_gaia/testlabels.npy')[0:5964]
LLR_pande_infer=np.load('LLR_gaia/labels_m1.npy')
LLR_pande_spectra=np.load('LLR_gaia/spectra_m1.npy')
LLR_astero_true=np.load('LLR_seismic/testlabels.npy')[5964:]
LLR_astero_infer=np.load('LLR_seismic/labels_m1.npy')
LLR_astero_spectra=np.load('LLR_seismic/spectra_m1.npy')

# Find Cannon stars in LLR pande. sample:
# --- get star's model & inferred logg
common_kics=[]
pande_true=[]
pande_inferred=[]
pande_models=[]
astero_true=[]
astero_inferred=[]
astero_models=[]
pande_files=[]
astero_files=[]
for file in cannon_test_files:
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic in LLR_pande_kics:
		common_kics.append(kic)
		star_idx=np.where(LLR_pande_kics==kic)[0]
		star_model=LLR_pande_spectra[star_idx]
		star_inferred=LLR_pande_infer[star_idx]
		star_true=LLR_pande_true[star_idx]
		pande_models.append(star_model)
		pande_inferred.append(star_inferred)
		pande_true.append(star_true)
		pande_files.append(LLR_pande_files[star_idx])
	elif kic in LLR_astero_kics:
		common_kics.append(kic)
		star_idx=np.where(LLR_astero_kics==kic)[0]
		star_model=LLR_astero_spectra[star_idx]
		star_inferred=LLR_astero_infer[star_idx]
		star_true=LLR_astero_true[star_idx]
		astero_models.append(star_model)
		astero_inferred.append(star_inferred)
		astero_true.append(star_true)
		astero_files.append(LLR_astero_files[star_idx])

LLR_true=pande_true+astero_true
LLR_infer=pande_inferred+astero_inferred
LLR_models=pande_models+astero_models
LLR_files=pande_files+astero_files
LLR_true=np.array(LLR_true)
LLR_infer=np.array(LLR_infer)

# plt.plot(cannon_true,cannon_true)
# plt.scatter(cannon_true,cannon_infer,c='k')
# plt.scatter(cannon_true,LLR_infer,c='r')
# plt.show(False)
for i in range(0,len(cannon_test_files)):
	file=cannon_test_files[i]
	kicid=int(file.split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
	if kicid in [6122418,3632803,11716673]:
		print(i,kicid)
# exit()
# Pick 2 stars:
giant=np.where((cannon_true>2.7) & (cannon_true<2.74) & (abs(cannon_true - cannon_infer)<0.05))[0]  #kicid: 6122418 idx=4

mid=np.where((cannon_true>3.0) & (cannon_true<3.2) & (abs(cannon_true - cannon_infer)<0.05))[0][3]   #kicid: 3632803, idx=3

dwarf=np.where((cannon_true>3.94) & (abs(cannon_true - cannon_infer)<0.1))[0][108]   #kicid: 11716673, idx=108
def get_info(stars):
	test_kics=[]
	for i in range(len(stars)):
		a=abs(cannon_true[stars][i]-cannon_infer[stars][i])
		b=abs(cannon_true[stars][i]-LLR_infer[stars][i])
		#print(i,cannon_true[dwarf][i],cannon_infer[dwarf][i],LLR_infer[dwarf][i],r'diff LLR<Cannon',b<a)
		file=cannon_test_files[stars][i]
		kicid=int(file.split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0'))
		if b<a:
		# if kicid==3735447:
			# print(kicid)
			print(i,kicid,cannon_true[stars][i],cannon_infer[stars][i],LLR_infer[stars][i],r'diff LLR<Cannon',b<a)
			test_kics.append(kicid)
	print(test_kics)
# get_info(giant)
# exit()
giant,mid,dwarf=2611,2790,1398

print('Giant',int(cannon_test_files[giant].split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0')))
print('Subgiant',int(cannon_test_files[mid].split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0')))
print('Dwarf',int(cannon_test_files[dwarf].split('/')[-1].split('-')[0].split('kplr')[-1].lstrip('0')))

# Find LLR model:
giant_model=10.**(LLR_models[giant])[0]
dwarf_model=10.**(LLR_models[dwarf])[0]
mid_model  =10.**(LLR_models[mid])[0]


freq = np.arange(0.01, 24., 0.01)
freq = 1000.*freq/86.4
freq = freq[86:86+2099]
freq_LLR=freq

# Get each star's data:

# Generate Cannon model:
a=open(d1+'coeffs_real_unweighted.pickle','rb')
coeffs_file=pickle.load(a)
a.close()

dataall2, metaall, labels, schmoffsets, coeffs, covs, scatters, chis, chisqs=coeffs_file

nstars  =3
nlabels =1    #2
offsets =[0.] #[0,0]

Params_all = np.zeros((nstars, nlabels))  #16 rows x 2 cols
true_loggs = [cannon_true[giant],cannon_true[mid],cannon_true[dwarf]]
#for 2 labels:
#kps = [[cannon_infer[giant],cannon_kp[giant]],[cannon_infer[mid],cannon_kp[mid]],[cannon_infer[dwarf],cannon_kp[dwarf]]]
#for 1 label:
kps = [[cannon_infer[giant]],[cannon_infer[mid]],[cannon_infer[dwarf]]]
for i in range(0,nstars):
    Params_all[i,:]=kps[i]

labels           = Params_all
features_data    = np.ones((nstars, 1))   #16 rows x 1 col
features_data    = np.hstack((features_data, labels - offsets))  #16 rows x 3 cols
newfeatures_data = np.array([np.outer(m, m)[np.triu_indices(nlabels)] for m in (labels - offsets)])
features_data    = np.hstack((features_data, newfeatures_data))  #16 rows x 6 cols  (6 cols=6 coefficients)

model_all=[]
for jj in range(nstars):
	# coeffs:          2099 rows x 6 cols (2099 frequency bins x 6 coefficients)
	# features_data.T:     6 rows x 16 cols (6 coefficients x 16 stars)
    model_gen = np.dot(coeffs,features_data.T[:,jj]) 
    model_all.append(model_gen) 
model_all = model_all # 16 rows x 2099 cols: 16 stars & 2099 frequency bins
# freq    =np.arange(10.011574074074073,277.76620370370375,0.01157)
legend_labels=['Cannon Best Fit Model:','Cannon Best Fit Model:','Cannon Best Fit Model:']
COLORS=['blue','red','green']
#plt.figure(figsize=(8,6))

print('Kp values:',kps)

fig=plt.figure(figsize=(7,10))
# plt.subplots_adjust(hspace = 0)
# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
plt.rc('font', size=15)                  # controls default text sizes
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend',fontsize=12)
fig.text(0.0, 0.5, 'Power Spectral Density [ppm$^2/\mathrm{\mu}$Hz]', va='center', rotation='vertical')

LW=2

ax = fig.add_subplot(111)    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])
# ax.set_ylabel('PSD (ppm$^2/\mathrm{\mu}$Hz)',fontsize=20,labelpad=30)
# 

llrcolor  ='#404788FF'
cancolor  ='tab:green'
llrlabel  ='LLM-LC Best Fit Model'
canlabel  ='Cannon Best Fit Model'

ax1 = fig.add_subplot(3,1,1)
spaces=''
i=0
f,amp=get_amp(LLR_files[giant])
amp=10.**(np.log10(amp))
ax1.loglog(f,amp,c='lightgrey',lw=1,alpha=1,zorder=0)
ax1.loglog(freq[0:2099],10.**(model_all[i]),label=canlabel,color=cancolor,linestyle='-',linewidth=LW)
ax1.loglog(freq_LLR,10.**(np.log10(giant_model)),color=llrcolor,label=llrlabel,linestyle='--',linewidth=LW)
ax1.set_ylim([1e-2,1e5])
ax1.set_xlim([9.,300])
ax1.set_xticks([])
# plt.legend(loc='lower right')

true_label=str(round(cannon_true[giant],2))
sl=len('True Logg: ')
s1='True Logg: {0:.2f} dex'.format(float(true_label))
s2='Cannon: '.ljust(sl+1)+'{0:.2f} dex'.format(labels[i][0])
s3='LLM-LC: '.ljust(sl+2)+'{0:.2f} dex'.format(LLR_infer[giant][0])
STR=s1+'\n'+s2+'\n'+s3
print(STR)
# exit()
# STR='True Logg: {} dex'.format(true_label)+'\n'+'Cannon: {0:.2f} dex'.format(labels[i][0])+'\n'+'LLR: {0:.2f} dex'.format(LLR_infer[giant][0])
t=ax1.text(0.03,0.2,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
t.set_bbox(dict(facecolor='white',edgecolor='grey'))#, alpha=0.5, edgecolor='red'))
ax1.tick_params(which='minor',axis='y',pad=4)# 

ax2 = fig.add_subplot(3,1,2)

i=1
f,amp=get_amp(LLR_files[mid])
amp=10.**(np.log10(amp))
ax2.loglog(f,amp,c='lightgrey',lw=1,alpha=1,zorder=0)
ax2.loglog(freq[0:2099],10.**(model_all[i]),label=canlabel,color=cancolor,linestyle='-',linewidth=LW)
ax2.loglog(freq_LLR,10.**(np.log10(mid_model)),color=llrcolor,label=llrlabel,linestyle='--',linewidth=LW)
ax2.set_ylim([1e-2,1e5])
ax2.set_xlim([9.,300])
ax2.set_xticks([])
# ax2.set_ylabel('Power Spectral Density [ppm$^2/\mathrm{\mu}$Hz]')
# plt.legend(loc='upper right')

true_label=str(round(cannon_true[mid],2))
sl=len('True Logg: ')
s1='True Logg: {0:.2f} dex'.format(float(true_label))
s2='Cannon: '.ljust(sl+1)+'{0:.2f} dex'.format(labels[i][0])
s3='LLM-LC: '.ljust(sl+2)+'{0:.2f} dex'.format(LLR_infer[mid][0])
STR=s1+'\n'+s2+'\n'+s3
print(STR)
t=ax2.text(0.03,0.2,s=STR,color='k',ha='left',va='center',transform = ax2.transAxes)
t.set_bbox(dict(facecolor='white',edgecolor='grey'))#, alpha=0.5, edgecolor='red'))

ax3 = fig.add_subplot(3,1,3)

i=2
f,amp=get_amp(LLR_files[dwarf])
amp=10.**(np.log10(amp))
ax3.loglog(f,amp,c='lightgrey',lw=1,alpha=1,zorder=0)
ax3.loglog(freq[0:2099],10.**(model_all[i]),label=canlabel,color=cancolor,linestyle='-',linewidth=LW)
ax3.loglog(freq_LLR,10.**(np.log10(dwarf_model)),color=llrcolor,label=llrlabel,linestyle='--',linewidth=LW)
ax3.set_ylim([1e-2,1e5])
ax3.set_xlim([9.,300])
ax3.set_xlabel('Frequency [$\mathrm{\mu}$Hz]')
plt.legend(loc='upper right')

true_label=str(round(cannon_true[dwarf],2))
sl=len('True Logg: ')
s1='True Logg: {0:.2f} dex'.format(float(true_label))
s2='Cannon: '.ljust(sl+1)+'{0:.2f} dex'.format(labels[i][0])
s3='LLM-LC: '.ljust(sl+2)+'{0:.2f} dex'.format(LLR_infer[dwarf][0])
STR=s1+'\n'+s2+'\n'+s3
print(STR)

t=ax3.text(0.03,0.2,s=STR,color='k',ha='left',va='center',transform = ax3.transAxes)
t.set_bbox(dict(facecolor='white',edgecolor='grey'))#, alpha=0.5, edgecolor='red'))
 
ax1.get_shared_x_axes().join(ax1, ax2)
ax2.get_shared_x_axes().join(ax2, ax3)

plt.tight_layout()
# plt.subplots_adjust(hspace=0)
plt.savefig('cannon_vs_LLR.png',dpi=100,bbox_inches='tight')
# plt.savefig('/Users/maryumsayeed/Desktop/HuberNess/iPoster/cannon_vs_LLR_sep.png',dpi=50,bbox_inches='tight')
plt.show(False)




