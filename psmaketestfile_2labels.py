import scipy
import numpy as np
import pickle,re
import pandas as pd
from astropy.io import fits
from astropy.io import ascii
from scipy import interpolate 
from scipy import ndimage 
import matplotlib.pyplot as plt
# Get Kps for all stars:
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
kps      =list(df['kic_kepmag'])

# Get white noise value:
d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'
whitenoise=np.loadtxt('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/whitenoisevalues.txt',skiprows=1,delimiter=',')

filename=d+'Cannon_test_stars.txt'

a = open(filename, 'r') 
al = a.readlines()

xgrid = np.arange(10.011574074074073,277.76620370370375,0.01157)
t_o = 1/0.00027
t_max = 1/0.000000009
t_n = len(xgrid)
t_diff = (t_max-t_o)/t_n
factor = 1./3600/24.
xgrid_ac = np.arange(t_o*factor,t_max*factor,t_diff*factor)

def interpolate_to_grid(xdata, ydata,xgrid):
  f = interpolate.interp1d(xdata, ydata)
  new_ydata= f(xgrid)
  return xgrid, new_ydata
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

al2 = [] 
for each in al:
    al2.append(each.split()[0]) 

numtest = len(al2)

freqall,fluxall = [],[] 
counter = 0

for file in al2:
    print(counter)
    #print(file)
    b = ascii.read(file)

    freq = b['freq'] 
    flux = b['power']
    wnoise = getkp(file)
    #flux = flux-wnoise
    xgrid = np.linspace(np.nanmin(freq),np.nanmax(freq),len(freq)) 
    newx,newy = interpolate_to_grid(freq,flux,xgrid)
    freqall.append(newx)
    fluxall.append(newy)
    # plt.loglog(freq,flux)
    # plt.xlim([9,300])
    # plt.ylim([0.01,1e6])
    # plt.show()
    counter = counter+1


ifluxall = [] 
print('Performing FFT...')
for each in fluxall[0:numtest]: # used to use the full 10,000 but now only use the first 2000 
    #ifluxall.append(np.fft.ifft(each)) 
    ifluxall.append(each)

#tc_fluxa_log = np.log(abs(np.real(ifluxall)))[:,0:780] # for logg, numax and deltanu inference
#tc_fluxa_log = np.log10(abs(np.asarray(ifluxall)))[:,0:21000] # for teff inference

#tc_fluxa = abs(np.real(ifluxall))[:,0:780] 
#tc_fluxa = (abs(np.asarray(ifluxall)))[:,0:21000]
tc_fluxa_log=[]
print('get tc_fluxa_log...')
for i in ifluxall:
    a=abs(np.asarray(i))[0:21000]
    tc_fluxa_log.append(np.log10(a))
#tc_fluxa_log = np.log10(abs(np.asarray(ifluxall)))[:,0:3000] # for teff inference

#tc_fluxa = abs(np.real(ifluxall))[:,0:780] 
print('get tc_fluxa (linear)...')
tc_fluxa=[]
for i in ifluxall:
    a=abs(np.asarray(i))[0:21000]
    tc_fluxa.append(a)
tc_flux = tc_fluxa_log
tc_flux_linear = tc_fluxa

tc_wavelx = [] 
tc_error = []
for each in tc_flux_linear:
    tc_error.append(1./each**0.5) # this gives best performance
    tc_wavelx.append(np.arange(0, len(each), 1))

#tc_error=[0.5]*len(tc_flux_linear)
#tc_error=np.ones(len(tc_flux_linear))
error_take = np.array(tc_error) 
#bad = np.isinf(error_take) 
labels = ['logg','teff']
nmeta = len(labels) 

logg,teff= np.loadtxt(filename, usecols = (1,2), unpack =1) 
logg_test  = logg
teff_test = teff
#kp_test   = kp

tc_names = al2
tc_names_test = al2
metaall = np.ones((len(tc_names), nmeta))
countit = np.arange(0,len(tc_flux),1)
newwl = np.arange(0,len(tc_flux),1) 
npix = np.shape(tc_flux[0]) [0]
 
dataall = np.zeros((npix, len(tc_names), 3))
for a,b,c,jj in zip(tc_wavelx, tc_flux, tc_error, countit):
    dataall[:,jj,0] = a
    dataall[:,jj,1] = b
    dataall[:,jj,2] = c

nstars = np.shape(dataall)[1]
for k in range(0,len(tc_names)):
    metaall[k,0] = logg_test[k]
    metaall[k,1] = teff_test[k]
    #metaall[k,2] = kp_test[k]
    #metaall[k,2] = kp_test[k]
#with open(d+'test_realifft_unweighted.pkl', "rb") as f:
#    w = pickle.load(f)

#pickle.dump(w, open('test_realifft_unweighted_py2.pickle',"wb"), protocol=2)

file_in = open(d+'testing_cannon.pickle', 'wb') 
pickle.dump((dataall[:,0:numtest,:], metaall[0:numtest,:], labels, tc_names_test[0:numtest], tc_names_test[0:numtest]),  file_in)
file_in.close()
