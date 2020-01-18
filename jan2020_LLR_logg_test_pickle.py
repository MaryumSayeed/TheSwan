#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle, os
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from astropy.io import ascii
import time
from numpy.linalg import lstsq
from astropy.stats import mad_std
import pickle, glob, re
import numpy as np
import pandas as pd
import datetime
from astropy.io import ascii
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.stats import LombScargle
from scipy.signal import savgol_filter as savgol
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.stats import mad_std
from sklearn.metrics import r2_score
import matplotlib.image as mpimg
from scipy.stats import chisquare
rc('text', usetex=True)


# In[2]:


plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.size']=18
plt.rcParams['mathtext.default']='regular'
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='8'


# In[3]:


def predict_labels_1(spectrum, P, L):
    '''
    Predicts label of test spectrum assuming that the power-spectrum error is more important.

    input:
    spectrum (array): spectrum of the test star
    P (list of lists): spectra of knn (k-nearest neighbours) [shape: (knn,npix)]
    L (list):          labels of knn [shape: (knn,nlabel)]

    Returns:
    label (float): inferred label 
    model (array): model fit for test star [shape: (npix,)]
    '''
    muL = np.mean(L, 0)             #mean of labels
    sL  = np.std(L, 0)              #std of labels
    L   = (L-muL)/sL                #pivot the labels (z-score)
    muP = np.mean(P, 0)             #mean of spectra
    sP  = np.std(P, 0)              #std of spectra
    P   = (P-muP)/sP                #pivot all closest spectra (z-score)
    spectrum = (spectrum-muP)/sP    #pivot the test spectrum (z-score) [shape: (npix,)]
    T   = lstsq(L, P)[0]            #solves for x in Lx=P for knn [shape: (nlabel,npix)]
    l   = lstsq(T.T, spectrum)[0]   #solves for x in Px=L for knn [shape: (npix,nlabel)]
    return (l*sL + muL), np.dot(T.T, l)*sP + muP

def predict_labels_2(spectrum, P, L):
    '''
    Predicts label of test spectrum assuming that the label error is more important

    input:
    spectrum (array):  spectrum of the test star
    P (list of lists): spectra of knn (k-nearest neighbours) [shape: (knn,npix)]
    L (list):          labels of knn [shape: (knn,nlabel)]

    Returns:
    label (float): inferred label [shape: (nlabel,)]
    '''
    muL = np.mean(L, 0)             #mean of labels        
    sL  = np.std(L, 0)              #std of labels
    L   = (L-muL)/sL                #pivot the labels (z-score)
    muP = np.mean(P, 0)             #mean of spectra
    sP  = np.std(P, 0)              #std of spectra
    P   = (P-muP)/sP                #pivot all closest spectra (z-score)
    spectrum = (spectrum-muP)/sP    #pivot the test spectrum (z-score) [shape: (npix,)]
    T   = lstsq(P, L)[0]            #solves for x in Px=L for knn [shape: (npix,nlabel)]
    return np.dot(spectrum, T)*sL + muL


# In[4]:
    

def getlowestchi2(chi2_values,teststar):
    lowest_chi2 = sorted(chi2_values)[0:11]
    chi2_idx    = np.array([np.where(chi2_values==i)[0][0] for i in lowest_chi2]) # find index of 10 lowest chi2 values
    print('before',chi2_idx)
    chi2_idx    = chi2_idx[chi2_idx != teststar]  #don't inlcude the test-star itself in training
    print('after',chi2_idx)

    min_chi2    =np.min(chi2_values[chi2_idx])
    return chi2_idx,min_chi2

def getclosestspectra(chi2_idx,test_spectrum,traindata,trainlabels):
    closest_spectra  = [] #10 closest spectra in training data
    for i in range(len(chi2_idx)):
        closest_spectra.append(traindata[chi2_idx[i],:])
    closest_labels = [[i] for i in trainlabels[chi2_idx]]

    label_m1,spectrum_m1 = predict_labels_1(test_spectrum, closest_spectra, closest_labels)
    label_m1=label_m1[0]
    
    label_m2=predict_labels_2(test_spectrum, closest_spectra, closest_labels)
    label_m2=label_m2[0]
    return label_m1,spectrum_m1,label_m2

def getinferredlabels(trainlabels,traindata,nstars,allfiles):
    print('Getting test data...')
    #a=open(testdata_file,'rb')
    #testdata=pickle.load(a)
    #a.close()
    #testlabels=testdata[1]
    #testlabels=testlabels[:,0]
    s1=time.time()

    traindata      =np.array(traindata)
    print('Shape of alldata:',np.shape(traindata))
    trainlabels    =np.array(trainlabels)

    testdata,testlabels=traindata,trainlabels     
    print('...took: {} s.'.format(time.time()-s1))

    # Number of stars to analyze:
    ngaia          =6135

    print('Number of total stars:',nstars)
    infer_avg     =np.zeros(ngaia)
    infer_m1      =np.zeros(ngaia)
    infer_m2      =np.zeros(ngaia)
    min_chi2      =np.zeros(ngaia)
    model_spectra =[]
    
    totalstart=time.time()
    print('Begin:',datetime.datetime.now())
    nast=0
    #for teststar in range(nast,nast+ngaia):
    for teststar in range(nast,nast+ngaia):
        file=allfiles[teststar]
        kic=re.search('kplr(.*)-', file).group(1)
        kic=int(kic.lstrip('0'))
        print('Star #',teststar,kic)
        try:
            s0=time.time()

            test_spectrum     =testdata[teststar,:] #1 in the last column contains the flux btw
            
            s1=time.time()
            #print(type(traindata[:,:]),np.shape(traindata[:,:]))
            allchi2           =np.sum((traindata[:,:]-test_spectrum)**2.,1)
            print('   get all chi2',time.time()-s1)

            s1=time.time()
            chi2_idx,smallest_chi2 =getlowestchi2(allchi2,teststar)
            print(traindata[chi2_idx,:])
            
            print('   getlowestchi2',time.time()-s1)

            s1=time.time()
            label_m1,spectrum_m1,label_m2=getclosestspectra(chi2_idx,test_spectrum,traindata,trainlabels)
            print('   getclosestspectra',time.time()-s1)
            avg_logg = np.average(trainlabels[chi2_idx])                           # find avg of 10 loggs with lowest chi2 values
            scatter_avg=round(testlabels[teststar]-avg_logg,2)
            scatter_m1 =round(testlabels[teststar]-label_m1,2)
            scatter_m2 =round(testlabels[teststar]-label_m2,2)
            print(teststar,'true:',testlabels[teststar])
            print(' ','avg ',round(avg_logg,2),'diff:',abs(scatter_avg))
            print(' ','mod1',round(label_m1,2),'diff:',abs(scatter_m1))
            print(' ','mod2',round(label_m2,2),'diff:',abs(scatter_m2))
            #infer_avg[teststar-nast]   =avg_logg
            infer_avg[teststar-nast]   =avg_logg
            infer_m1[teststar-nast]    =label_m1
            infer_m2[teststar-nast]    =label_m2
            min_chi2[teststar-nast]    =smallest_chi2
            model_spectra.append(spectrum_m1)
            print('  star: {}'.format(round(time.time()-s0,2)))
            print('\n')
        except:
            # if model can't find a best fit:
            print('problem with:',teststar)
            infer_avg[teststar-nast]   =-99
            infer_m1[teststar-nast]    =-99
            infer_m2[teststar-nast]    =-99
            min_chi2[teststar-nast]    =-99
            model_spectra.append([-99])
    print('\n','total time taken:',time.time()-totalstart)
    print('End:',datetime.datetime.now())
    return testlabels,infer_avg,infer_m1,model_spectra,infer_m2,min_chi2


'''def getinferredlabels(trainlabels,traindata,testdata_file):
    print('Getting test data...')
    a=open(testdata_file,'rb')
    testdata=pickle.load(a)
    a.close()
    testlabels=testdata[1]
    testlabels=testlabels[:,0]

    nstars        =np.shape(testdata)[1]
    infer_avg     =np.zeros(nstars)
    infer_m1      =np.zeros(nstars)
    infer_m2      =np.zeros(nstars)
    model_spectra   =[]
    
    totalstart=time.time()
    print('Begin:',datetime.datetime.now())
    for teststar in range(len(infer_avg)):
        print('Star #',teststar)
        s1=time.time()
        allchi2=np.zeros(len(trainlabels))
        test  =testdata[0][:,teststar,1] #compare log(PSD) values
        for trainstar in range(len(trainlabels)):
            train =traindata[:,trainstar,1]
            chi2,_=chisquare(f_obs=test,f_exp=train)
            allchi2[trainstar] = chi2
        lowest_chi2 = sorted(allchi2)[0:10]
        chi2_idx    = np.array([np.where(allchi2==i)[0][0] for i in lowest_chi2]) # find index of 10 lowest chi2 values
        avg_logg    = np.average(trainlabels[chi2_idx])                           # find avg of 10 loggs with lowest chi2 values
        min_idx     = np.where(allchi2==np.min(allchi2))[0][0]
        closest_logg  = trainlabels[min_idx]
        closest_spectra = [] #10 closest spectra in training data
        for i in range(len(chi2_idx)):
            closest_spectra.append(traindata[:,chi2_idx[i],1])
        closest_labels = [[i] for i in trainlabels[chi2_idx]]
    
        label_m1,spectrum_m1 = predict_labels_1(test, closest_spectra, closest_labels)
        label_m1=label_m1[0]
        
        label_m2=predict_labels_2(test, closest_spectra, closest_labels)
        label_m2=label_m2[0]
        best_logg =avg_logg

        scatter_avg=round(testlabels[teststar]-best_logg,2)
        scatter_m1 =round(testlabels[teststar]-label_m1,2)
        scatter_m2 =round(testlabels[teststar]-label_m2,2)
        print(teststar,'true:',testlabels[teststar])
        print(' ','avg ',round(best_logg,2),'diff:',abs(scatter_avg))
        print(' ','mod1',round(label_m1,2),'diff:',abs(scatter_m1))
        print(' ','mod2',round(label_m2,2),'diff:',abs(scatter_m2))
        infer_avg[teststar]   =best_logg
        infer_m1[teststar]    =label_m1
        infer_m2[teststar]    =label_m2
        model_spectra.append(spectrum_m1)
        print('  star: {}'.format(round(time.time()-s1,2)))
        print('\n')

    print('\n','total time taken:',time.time()-totalstart)
    print('End:',datetime.datetime.now())
    return testlabels,infer_avg,infer_m1,model_spectra,infer_m2'''

# In[5]:

#def gettraindata(text_file,pickle_file,nstars):
def gettraindata(text_files,pickle_files):
    begintime=datetime.datetime.now()
    trainlabels=[]
    alldata=[]
    allpickledata=[]
    allfiles=[]
    star_count=0
    data_lengths=[]
    for i in range(0,len(text_files)):
        print(i,'getting data from:',pickle_files[i])
        # pf=open(pickle_files[i],'rb')
        # data,_=pickle.load(pf)
        # pf.close()
        
        # allpickledata.append(data[:,:,1].transpose())
        labels=np.loadtxt(text_files[i],delimiter=' ',usecols=[1])
        files=np.loadtxt(text_files[i],delimiter=' ',usecols=[0],dtype=str)
        stars= len(labels)
        data = np.memmap(pickle_files[i],dtype=np.float32,mode='r',shape=(21000,stars,3))
        trainlabels.append(labels)
        traindata=data[:,:,1].transpose()
        print(traindata[0,:])
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
    #traindata=list(alldata[0])+list(alldata[1])+list(alldata[2]) #rows=nstars x columns=npix
    
    # Uncomment below if data NOT saved:
    # print('Concatenating data...')
    # alldata=np.concatenate((allpickledata),axis=0) #this step was taking up memory so switched to np.memmap. It can be used for smaller files.

    # print('Saving alldata into array...')
    # np.save('alldata.npy',alldata)
    
    # Uncomment section below if data already saved:
    
    #alldata=np.load('alldata.npy')

    #labels=np.concatenate((trainlabels))

    #alldata=np.concatenate((alldata),axis=0)
    print('Shape of whole data set (stars, bins):',np.shape(alldata))
    print('Shape of labels:',np.shape(labels))
    print('Shape of files:',np.shape(allfiles))
    start,end=6300,6307
    #print(allfiles[start:end])
    #print(labels[start:end])

    total_stars=star_count
    print('Total # of stars:',total_stars)
    print('     ','took {}s.'.format(datetime.datetime.now()-begintime))
    return labels,alldata,total_stars,allfiles




train_file_names =['pande_pickle_1','pande_pickle_2','pande_pickle_3','astero_final_sample_1','astero_final_sample_2','astero_final_sample_3','astero_final_sample_4']
train_file_pickle=[i+'_memmap.pickle' for i in train_file_names]
train_file_txt   =[i+'.txt' for i in train_file_names]


# In[6]:
print('Getting training data...')
train_labels,train_data,total_stars,all_files=gettraindata(train_file_txt,train_file_pickle)
print('Beginning inference...')
testlabels,average,labelm1,modelm1,labelm2,min_chi2=getinferredlabels(train_labels,train_data,total_stars,all_files)
dirr='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/jan2020_pande_LLR4/'
np.save(dirr+'testlabels.npy',testlabels)
np.save(dirr+'average.npy',average)
np.save(dirr+'labels_m1.npy',labelm1)
np.save(dirr+'labels_m2.npy',labelm2)
np.save(dirr+'spectra_m1.npy',modelm1)
np.save(dirr+'min_chi2.npy',min_chi2)
#print(min_chi2)
# In[8]:


def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms


# In[12]:


def plotresults(testlabels,infer,model_1,model_2):
    std_avg=mad_std(infer-testlabels)
    std_m1=mad_std(model_1-testlabels)
    std_m2=mad_std(model_2-testlabels)
    bias_av,rms_av=returnscatter(testlabels-infer)
    bias_m1,rms_m1=returnscatter(testlabels-model_1)
    bias_m2,rms_m2=returnscatter(testlabels-model_2)
    plt.figure(figsize=(20,15))
    plt.subplot(221)
    x=np.linspace(1.5,4.8,100)
    plt.plot(x,x,c='k',linestyle='dashed')
    plt.scatter(testlabels,infer,facecolors='none',edgecolors='r',label='Average',zorder=10)
    plt.scatter(testlabels,model_1,facecolors='none', edgecolors='g',label='Model-1')
    plt.scatter(testlabels,model_2,facecolors='none', edgecolors='b',label='Model-2')
    plt.xlabel('True')
    plt.ylabel('Inferred')
    plt.legend(loc=2)
    
    plt.subplot(222)
    x=np.linspace(1.5,4.8,100)
    plt.plot(x,x,c='k',linestyle='dashed')
    plt.scatter(testlabels,infer,facecolors='none',edgecolors='r',label='Average',zorder=10)
    plt.text(1.5,4.5,s='m.std={}'.format(round(std_avg,3)))
    plt.text(1.5,4.2,s='RMS={}'.format(round(rms_av,2)))
    plt.text(1.5,3.9,s='Bias={}'.format(round(bias_av,3)))
    plt.xlabel('True')
    plt.ylabel('Inferred')
    plt.legend(loc=1)
    
    plt.subplot(223)
    x=np.linspace(1.5,4.8,100)
    plt.plot(x,x,c='k',linestyle='dashed')
    plt.scatter(testlabels,model_1,facecolors='none', edgecolors='g',label='Model-1')
    plt.text(1.5,4.5,s='m.std={}'.format(round(std_m1,3)))
    plt.text(1.5,4.2,s='RMS={}'.format(round(rms_m1,2)))
    plt.text(1.5,3.9,s='Bias={}'.format(round(bias_m1,3)))
    plt.xlabel('True')
    plt.ylabel('Inferred')
    plt.legend(loc=1)
    
    plt.subplot(224)
    x=np.linspace(1.5,4.8,100)
    plt.plot(x,x,c='k',linestyle='dashed')
    plt.scatter(testlabels,model_2,facecolors='none', edgecolors='b',label='Model-2')
    plt.text(1.5,4.5,s='m.std={}'.format(round(std_m2,3)))
    plt.text(1.5,4.2,s='RMS={}'.format(round(rms_m2,2)))
    plt.text(1.5,3.9,s='Bias={}'.format(round(bias_m2,3)))
    plt.xlabel('True')
    plt.ylabel('Inferred')
    plt.legend(loc=1)
    
    #plt.suptitle('Logg - 10 neighbours')
    plt.savefig(dirr+'part1.pdf')
    # plt.show()
    print('average',std_avg,rms_av,bias_av)
    print('model_1',std_m1,rms_m1,bias_m1)
    print('model_2',std_m2,rms_m2,bias_m2)


# In[13]:


#plotresults(testlabels[6000+5412:6000+5412+len(average)],average,labelm1,labelm2)
start=0
plotresults(testlabels[start:start+len(average)],average,labelm1,labelm2)

# In[11]:


exit()



















#np.memmap info: DO NOT DELETE
'''print('Making memmap file...')
    memmap_filename='test.array'

    a=np.memmap(memmap_filename,dtype=np.float64, mode='w+', shape=(21000,data_lengths[0],3))
    a[:,:,:]=allpickledata[0]
    del a
    
    print('Appending to memmap file...')
    for i in range(1,len(data_lengths)):
        b=np.memmap(memmap_filename, dtype=np.float64, mode='r+', shape=(21000,data_lengths[i],3),order='F')
        b[:,:,:]=allpickledata[i]
        del b
        
    alldata=np.memmap(memmap_filename, dtype=np.float64, mode='r+', shape=(21000,star_count,3))
    print('Shape of memmap file:',np.shape(alldata))
    a=np.memmap('alldata.array',dtype=np.float64, mode='w+', shape=(21000,star_count,3))
    a[:,:,:]=alldata
    del a
    alldata=np.memmap('alldata.array',dtype=np.float64, mode='r', shape=(21000,star_count,3))
    '''


# In[ ]:


# Get Kps for all stars:
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
allkps      =list(df['kic_kepmag'])
gaia     =ascii.read('/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/DR2PapTable1.txt',delimiter='&')

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


# In[ ]:


print('bias','RMS')
print(returnscatter(testlabels-av))
print(returnscatter(testlabels-labelm1))
print(returnscatter(testlabels-labelm2))


idx=np.where((abs(testlabels-labelm1)>0.2))[0]
print(len(idx))
idx=idx[0:5]

a=open(test_file,'rb')
testdata=pickle.load(a)
a.close()

for star in idx:
    file=testdata[-1][star][0:-3]
    kic=re.search('kplr(.*)-', file).group(1)
    kic=int(kic.lstrip('0'))
    kp =allkps[kp_kics.index(kic)]
    idx=np.where(gaia['KIC']==kic)
    t=gaia['teff'][idx][0]
    r=gaia['rad'][idx][0]
    l=r**2.*(t/5777.)**4.
    
    time,flux,f,amp,pssm=getps(file,1)
    kic=file.split('/')[-1].split('-')[0].split('kplr')[-1]
    kic=int(kic.lstrip('0'))
    test =10.**testdata[0][:,star,1] #compare log(PSD) values
    
    model=10.**modelm1[star] #compare log(PSD) values
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    plt.plot(time,flux,linewidth=1)
    plt.title('lightcurve of test star',fontsize=20)
    plt.subplot(122)
    plt.loglog(f,amp,c='grey',alpha=0.3,linewidth=1)
    plt.loglog(f[864:21000+864],test,c='k',label='smoothed')
    plt.loglog(f[864:864+21000],pssm[864:864+21000],c='cyan',label='smoothed')
    plt.loglog(f[864:864+21000],model,c='r',label='model')
    #plt.title('KICID: {}'.format(str(kic)),fontsize=20)
    plt.title('KICID: {} Teff: {} Lum: {} Kp: {}'.format(str(kic),t,round(l,2),round(kp,2)),fontsize=20)

    a=round(testlabels[star],2)
    b=round(labelm1[star],2)
    plt.text(10,10**4.,s='True: {} Pred.: {}'.format(a,b),fontsize=20,ha='left')
    plt.xlim([8,300])
    plt.ylim([0.1,1e5])
    plt.xlabel('Frequency ($\mu$Hz)')
    plt.ylabel(r'PSD (ppm$^2$/$\mu$Hz)')
    plt.legend()
    plt.tight_layout()
    d='/Users/maryumsayeed/plotsfornextweek/plotforjuly19/logg/draft2/bad/'
    #plt.savefig(d+str(kic)+'.pdf')
    plt.show()
    plt.clf()


# In[ ]:


stop
import os,glob
from PyPDF2 import PdfFileMerger
d='/Users/maryumsayeed/plotsfornextweek/plotforjuly19/logg/draft2/bad/'
pdfs=glob.glob(d+'*.pdf')
merger = PdfFileMerger()
for pdf in pdfs:
    p=pdf
    merger.append(p)
merger.write(d+"sigma>0.15.pdf")


# In[ ]:





# In[ ]:


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
    res =sigclip(time,flux,50,3)
    good=np.where(res == 1)[0]
    time=time[good]
    flux=flux[good]
    width=day
    boxsize=width/(30./60./24.)
    box_kernel = Box1DKernel(boxsize)
    smoothed_flux = savgol(flux,int(boxsize)-1,1,mode='mirror')
    flux=flux/(smoothed_flux)

    yq=1./(30./60./24.)
    fres=1./90./0.0864

    fres_cd=0.001
    fres_mhz=fres_cd/0.0864

    freq = np.arange(0.001, 24., 0.001)
    amp = LombScargle(time,flux).power(freq)

    # unit conversions
    freq = 1000.*freq/86.4
    bin = freq[1]-freq[0]
    amp = 2.*amp*np.var(flux*1e6)/(np.sum(amp)*bin)

    # smooth by 2 muHz
    n=np.int(2./fres_mhz)

    gauss_kernel = Gaussian1DKernel(n)
    pssm = convolve(amp, gauss_kernel)
#     wnoise=getkp(file)
#     print('white noise:',wnoise)
#     pssm = pssm-wnoise
    
    return time,flux,freq,amp,pssm


# In[ ]:




