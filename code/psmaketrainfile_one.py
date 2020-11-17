import scipy
import numpy as np
import pandas as pd
import pickle,re,math
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io import ascii
from scipy import interpolate 
from scipy import ndimage 
# Get Kps for all stars:
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
kps      =list(df['kic_kepmag'])

d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'
# NAME='baseline_cuts/65_days/LLR_gaia/pande_pickle_3'
# NAME='baseline_cuts/65_days/LLR_seismic/astero_final_sample_4'
# NAME='short_cadence/training_cannon'
# NAME='cannon_vs_LLR/other_labels/testing_cannon_lum'
NAME='cannon_vs_LLR/other_labels/training_cannon_lum'
# NAME='LLR_seismic/astero_final_sample_4'
# NAME='LLR_gaia/pande_pickle_3'
filename=d+'{}.txt'.format(NAME)
a = open(filename, 'r')  #fitsfile names
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')
numtrain = 7000
# numtrain = 20
numtest=numtrain
al = a.readlines()[0:numtrain]

xgrid = np.arange(10.011574074074073,277.76620370370375,0.01157)
t_o = 1/0.00027
t_max = 1/0.000000009
t_n = len(xgrid)
t_diff = (t_max-t_o)/t_n
factor = 1./3600/24.
xgrid_ac = np.arange(t_o*factor,t_max*factor,t_diff*factor)
diff_freq = np.diff(xgrid)
Per = 1/diff_freq/(10**-6)
diff_time = Per/len(xgrid)

def interpolate_to_grid(xdata, ydata,xgrid):
  f = interpolate.interp1d(xdata, ydata)
  new_ydata= f(xgrid)
  return xgrid, new_ydata

def getclosest(num,collection):
    '''Given a number and a list, get closest number in the list to number given.'''
    return min(collection,key=lambda x:abs(x-num))

al2 = [] 
for each in al:
    al2.append(each.split()[0]) 

freqall,fluxall = [],[] 
counter = 0

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

shapes=[]
for file in al2:
    print(counter)
    b = ascii.read(file)
    freq = b['freq'] 
    flux = b['power']
    #wnoise = getkp(file)
    freqall.append(freq)
    fluxall.append(np.array(flux,dtype='float64'))
    shapes.append(len(freq))
    counter = counter+1
    
# end=21000 #oversampling=10

# end=np.where(freq>8400)[0][0]     #short cadence
# end=np.where(freq>2000)[0][0]     #short cadence
end=np.where(freq>253)[0][0]    
start=0
# start=np.where(freq>100.)[0][0]
print('# of freq. bins to feed:',end)
ifluxall     =np.array(fluxall)

print('Get logarithmic values...')
# tc_fluxa_log = (np.log10(np.abs(ifluxall)))[:,0:21000] # for teff label

tc_fluxa_log=[]
for i in ifluxall:
    tc_fluxa_log.append((np.log10(np.abs(i[start:end]))))
    
print('Get linear values...')
# tc_fluxa     = (abs(np.asarray(ifluxall)))[:,0:21000]
tc_fluxa = []
for i in ifluxall:
    tc_fluxa.append(abs(i[start:end]))

tc_fluxa_log=np.array(tc_fluxa_log)
tc_fluxa=np.array(tc_fluxa)

# Sample every nth pixel:
tc_flux = tc_fluxa_log#[:,0::4] 
tc_flux_linear = tc_fluxa#[:,0::4]

tc_wavelx = [] 
tc_error  = []
print('Get error...')
for each in tc_flux_linear:
    tc_error.append(1./each**0.5) # this gives best performance
    tc_wavelx.append(np.arange(0, len(each), 1))

error_take = np.array(tc_error) 
# bad        = np.isinf(error_take) 
labels     = ['logg']

nmeta      = len(labels) 
logg       = np.loadtxt(filename, usecols = (1), unpack =1) 
logg_train = logg[0:numtrain]

tc_names = al2[0:numtrain] 
metaall  = np.ones((len(tc_names), nmeta))
countit  = np.arange(0,len(tc_flux),1)
newwl    = np.arange(0,len(tc_flux),1) 
npix     = np.shape(tc_flux[0])[0]
 
print('Constructing dataall...')

dataall = np.zeros((npix, len(tc_names), 3)) # use for oversampling=10
for a,b,c,jj in zip(tc_wavelx, tc_flux, tc_error, countit):
    dataall[:,jj,0] = a
    dataall[:,jj,1] = b
    dataall[:,jj,2] = c

for k in range(0,len(logg_train)):
# for k in range(len(al)):
    metaall[k,0] = logg_train[k]

print('Saving file...')
data_to_save=dataall

print(np.shape(data_to_save))

# Used for LLR:
# fp = np.memmap(d+'{}_memmap.pickle'.format(NAME), mode='w+', dtype=np.float32,shape=np.shape(data_to_save))
# fp[:]=data_to_save[:]
# del fp

# # Use for Cannon with 1 label:
file_in = open(d+'{}.pickle'.format(NAME), 'wb') 
pickle.dump((dataall, metaall, labels, tc_names, al2),  file_in)
file_in.close()
# 
# pickle.dump(w, open(d+'training_realifft_unweighted_py2.pickle',"wb"), protocol=2)

# file_in = open(d+'astero_final_sample_4.pickle', 'wb') 
# pickle.dump((dataall,metaall),  file_in)
# file_in.close()