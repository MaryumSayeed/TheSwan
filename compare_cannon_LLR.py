import numpy as np
import matplotlib.pyplot as plt
import pickle, re, glob
import pyfits
import pandas as pd
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol


# Get Kps for all stars:
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
kps      =list(df['kic_kepmag'])

# choose x stars across logg range 
# get their LLR model 
# generate their possible Cannon model
# plot on PSD vs. Freq plot

# Load Cannon files & results:
cannon_test_files=np.loadtxt('Cannon_test_stars.txt',usecols=[0],dtype='str')
pf=open('testmetaall_cannon_logg.pickle','rb') 
testmetaall,truemetaall=pickle.load(pf)
pf.close() 
cannon_true=truemetaall[:,0]
cannon_infer=testmetaall[:,0]
cannon_kp=truemetaall[:,1]

# Load LLR files & results:
LLR_astero_files=np.loadtxt('astero_final_sample_full.txt',usecols=[0],dtype='str')
LLR_pande_files=np.loadtxt('pande_final_sample_full.txt',usecols=[0],dtype='str')

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

LLR_pande_true=np.load('jan2020_pande_sample/testlabels.npy')[0:6135]
LLR_pande_infer=np.load('jan2020_pande_sample/labels_m1.npy')
LLR_pande_spectra=np.load('jan2020_pande_sample/spectra_m1.npy')
LLR_astero_true=np.load('jan2020_astero_sample/testlabels.npy')[6135:]
LLR_astero_infer=np.load('jan2020_astero_sample/labels_m1.npy')
LLR_astero_spectra=np.load('jan2020_astero_sample/spectra_m1.npy')

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

# Pick 2 stars:
giant=np.where((cannon_true<2.5) & abs((cannon_true - cannon_infer)<0.5))[0][7]
dwarf=np.where((cannon_true>3.5) & (cannon_true<4.) & abs((cannon_true - cannon_infer)<0.5))[0][7]
mid=np.where((cannon_true>3.) & (cannon_true<3.5) & abs((cannon_true - cannon_infer)<0.5))[0][7]

#print(cannon_true[giant],cannon_infer[giant],LLR_true[giant],LLR_infer[giant])
#print(cannon_true[mid],cannon_infer[mid],LLR_true[mid],LLR_infer[mid])
#print(cannon_true[dwarf],cannon_infer[dwarf],LLR_true[dwarf],LLR_infer[dwarf])

# Find LLR model:
giant_model=10.**(LLR_models[giant])[0]
dwarf_model=10.**(LLR_models[dwarf])[0]
mid_model  =10.**(LLR_models[mid])[0]


freq = np.arange(0.001, 24., 0.001)
freq_LLR = 1000.*freq/86.4
freq_LLR=freq_LLR[864:864+21000]

# Get each star's data:


# Generate Cannon model:
a=open('coeffs_real_unweighted.pickle','rb')
coeffs_file=pickle.load(a)
a.close()

dataall2, metaall, labels, schmoffsets, coeffs, covs, scatters, chis, chisqs=coeffs_file

nstars  =3
nlabels =2
offsets =[0.,0.]

Params_all = np.zeros((nstars, nlabels))  #16 rows x 2 cols
true_loggs = [cannon_true[giant],cannon_true[mid],cannon_true[dwarf]]
kps = [[cannon_infer[giant],cannon_kp[giant]],[cannon_infer[mid],cannon_kp[mid]],[cannon_infer[dwarf],cannon_kp[dwarf]]]
for i in range(0,nstars):
    Params_all[i,:]=kps[i]

labels           = Params_all
features_data    = np.ones((nstars, 1))   #16 rows x 1 col
features_data    = np.hstack((features_data, labels - offsets))  #16 rows x 3 cols
newfeatures_data = np.array([np.outer(m, m)[np.triu_indices(nlabels)] for m in (labels - offsets)])
features_data    = np.hstack((features_data, newfeatures_data))  #16 rows x 6 cols  (6 cols=6 coefficients)

model_all=[]
for jj in range(nstars):
	# coeffs:          21000 rows x 6 cols (21000 frequency bins x 6 coefficients)
	# features_data.T:     6 rows x 16 cols (6 coefficients x 16 stars)
    model_gen = np.dot(coeffs,features_data.T[:,jj]) 
    model_all.append(model_gen) 
model_all = model_all # 16 rows x 21000 cols: 16 stars & 21000 frequency bins
freq    =np.arange(10.011574074074073,277.76620370370375,0.01157)
legend_labels=['Cannon Inference:','Cannon Inference:','Cannon Inference:']
COLORS=['blue','red','green']
#plt.figure(figsize=(8,6))


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
	data=pyfits.open(file)
	head=data[0].data
	dat=data[1].data
	time=dat['TIME']
	qual=dat['SAP_QUALITY']
	flux=dat['PDCSAP_FLUX']
	ndays=time[-1]-time[0]
	nmins=ndays*24.*60.
	expected_points=nmins/30.
	observed_points=len(time)

	# if observed_points < expected_points*0.5:
	# 	print(kicid,'below')
	# 	continue

	#um=np.where(gaia['KIC'] == kicid)[0]
	# if (len(um) == 0.):
	# 	continue

	# if (teffs[i] == 0):
	# 	continue

	# only keep data with good quality flags
	good=np.where(qual == 0)[0]
	time=time[good]
	flux=flux[good]

	
	# sigma-clip outliers from the light curve and overplot it
	res=sigclip(time,flux,50,3)
	good = np.where(res == 1)[0]
	time=time[good]
	flux=flux[good]	
	
	# next, run a filter through the data to remove long-periodic (low frequency) variations
	 
	width=1.
	boxsize=width/(30./60./24.)
	box_kernel = Box1DKernel(boxsize)
	smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
	# overplot this smoothed version, and then divide the light curve through it

	flux=flux/(smoothed_flux)
	 
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

	#White noise correction:
	# wnoise=getkp(file)
	# amp_wn=np.zeros(len(amp))
	# for p in range(0,len(amp)):
	# 	a=amp[p]
	# 	if a-wnoise < 0.:
	# 		amp_wn[p]=amp[p]
	# 	if a-wnoise > 0.:
	# 		amp_wn[p]=a-wnoise
	
	return freq,amp

fig=plt.figure(figsize=(7,10))
plt.subplots_adjust(hspace = 0)
# plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend',fontsize=12)

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
import matplotlib.ticker as tck

llrcolor  ='#404788FF'
ax1 = fig.add_subplot(3,1,1)
spaces=''
i=0
offset = 0
f,amp=get_amp(LLR_files[giant])
amp=10.**(np.log10(amp)+offset)
LABEL=legend_labels[i]+' {} dex'.format(str(round(labels[i][0],2)))
ax1.loglog(f,amp,c='lightgrey',lw=1,alpha=0.7,zorder=0)
ax1.loglog(freq[0:21000],10.**(model_all[i]+offset),label=LABEL,color='tab:green',linestyle='-',linewidth=LW)
LABEL='LLR Inference: {}{} dex'.format(spaces,str(round(LLR_infer[giant][0],2)))
ax1.loglog(freq_LLR,10.**(np.log10(giant_model)+offset),color=llrcolor,label=LABEL,linewidth=LW)
ax1.set_ylim([10**1.5,1e5])
ax1.set_xlim([9.,255])
ax1.set_xticks([])
plt.legend(loc='upper right')
true_label=str(round(cannon_true[giant],2))
ax1.text(10,50,'True Logg: {} dex'.format(true_label),bbox=dict(facecolor='white'),va='bottom',fontsize=15)
ax1.tick_params(which='minor',axis='y',pad=4)# 
#ax1.yaxis.set_minor_locator(tck.AutoMinorLocator())
# plt.minorticks_on()

ax2 = fig.add_subplot(3,1,2)

i=1
offset = 0
f,amp=get_amp(LLR_files[mid])
amp=10.**(np.log10(amp)+offset)
LABEL=legend_labels[i]+' {} dex'.format(str(round(labels[i][0],2)))
ax2.loglog(f,amp,c='lightgrey',lw=1,alpha=0.7,zorder=0)
ax2.loglog(freq[0:21000],10.**(model_all[i]+offset),label=LABEL,color='tab:green',linestyle='-',linewidth=LW)
LABEL='LLR Inference: {}{} dex'.format(spaces,str(round(LLR_infer[mid][0],2)))
ax2.loglog(freq_LLR,10.**(np.log10(mid_model)+offset),color=llrcolor,label=LABEL,linewidth=LW)
ax2.set_ylim([3e1,10**3.5])
ax2.set_xlim([9.,255])
ax2.set_xticks([])
plt.legend(loc='upper right')
true_label=str(round(cannon_true[mid],2))
ax2.text(10,40,'True Logg: {} dex'.format(true_label),bbox=dict(facecolor='white'),va='bottom',fontsize=15)
ax2.set_ylabel('PSD [ppm$^2/\mathrm{\mu}$Hz]',fontsize=20)#,labelpad=30)

ax3 = fig.add_subplot(3,1,3)

i=2
offset = 0
f,amp=get_amp(LLR_files[dwarf])
amp=10.**(np.log10(amp)+offset)
LABEL=legend_labels[i]+' {} dex'.format(str(round(labels[i][0],2)))
ax3.loglog(f,amp,c='lightgrey',lw=1,alpha=0.7,zorder=0)
ax3.loglog(freq[0:21000],10.**(model_all[i]+offset),label=LABEL,color='tab:green',linestyle='-',linewidth=LW)
LABEL='LLR Inference: {}{} dex'.format(spaces,str(round(LLR_infer[dwarf][0],2)))
ax3.loglog(freq_LLR,10.**(np.log10(dwarf_model)+offset),color=llrcolor,label=LABEL,linewidth=LW)
ax3.set_ylim([10**(0.5),1e3])
ax3.set_xlim([9.,255])
ax3.set_xlabel('Frequency [$\mathrm{\mu}$Hz]',fontsize=20)
plt.legend(loc='upper right')

true_label=str(round(cannon_true[dwarf],2))
ax3.text(10,5e0,'True Logg: {} dex'.format(true_label),bbox=dict(facecolor='white'),va='bottom',fontsize=15)
 
# plt.ylabel('PSD (ppm$^2/\mathrm{\mu}$Hz)',fontsize=20)
ax1.get_shared_x_axes().join(ax1, ax2)
ax2.get_shared_x_axes().join(ax2, ax3)

plt.tight_layout()
# plt.savefig('/Users/maryumsayeed/Desktop/HuberNess/iPoster/cannon_vs_LLR_sep.pdf',dpi=50)
plt.show(False)




