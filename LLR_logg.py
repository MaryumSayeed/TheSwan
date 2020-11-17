#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle, os, sys
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
# rc('text', usetex=True)
import argparse

# In[2]:

# Create the parser
my_parser = argparse.ArgumentParser(description='Run LLR on given sample.')

# Add the arguments
my_parser.add_argument('-sample','--sample',
                       metavar='sample',
                       type=str,required=True,
                       help='The sample (Gaia or Seismic) to run LLR on.')

my_parser.add_argument('-d','--directory',
                       metavar='directory',
                       type=str,required=True,
                       help='Next sub-directory (in ~/powerspectrum) to save results in.')


# Execute the parse_args() method
args = my_parser.parse_args()

sample = args.sample
dirr= args.directory

if sample.lower() =='gaia':
    NGAIA=5964  # number of stars of given sample (Gaia=5964, Seismic=14168)
    NAST=0      # index where seismic stars start. 0 if not the seismic sample.
    START=0     # for plotting: index to start plotting true logg values
elif sample.lower() =='seismic':
    NGAIA=14168 # number of stars of given sample (Gaia=5964, Seismic=14168)
    NAST=5964   # index where seismic stars start. 0 if not the seismic sample.
    START=5964  # for plotting: index to start plotting true logg values
else:
    print('Sample not recognized. Try "Gaia" or "Seismic".')


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
    # if analyzing Pande. sample:  ngaia = 5964 
    # if analyzing astero. sample: ngaia = 14168 
    # if testing, let ngaia=10
    ngaia         =NGAIA#14168#5964

    print('Number of total stars:',nstars)
    infer_avg     =np.zeros(ngaia)
    infer_m1      =np.zeros(ngaia)
    infer_m2      =np.zeros(ngaia)
    min_chi2      =np.zeros(ngaia)
    model_spectra =[]
    
    totalstart=time.time()
    print('Begin:',datetime.datetime.now())
    
    # if analyzing Pande. sample:  nast = 0
    # if analyzing astero. sample: nast = 5964
    nast=NAST#5964

    #for teststar in range(nast,nast+ngaia):
    #delete lines below after done simul wnoise stuff
    # print(np.shape(trainlabels))
    # t2=trainlabels[6135:]
    # print('len(t2):',len(t2))
    # i1=np.where((t2>2.) & (t2<2.5))[0][0:200]
    # i2=np.where((t2>2.5) & (t2<3.))[0][0:200]
    # i3=np.where((t2>3.) & (t2<3.5))[0][0:200]
    # i4=np.where((t2>3.5) & (t2<4.))[0][0:200]
    # i5=np.where((t2>4.) & (t2<4.5))[0][0:200]
    # i6=np.where((t2>4.5))[0][0:200]
    
    # want_idx=np.concatenate([i1,i2,i3,i4,i5,i6])
    # print('len(want_idx)=',len(want_idx))
    # want_idx=want_idx+6135
    #delete lines above after done simul wnoise stuff
    for teststar in range(nast,nast+ngaia):
        #if teststar in want_idx:#delete this line after done simul wnoise stuff
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
            #print(traindata[chi2_idx,:])
            
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
        #data = np.memmap(pickle_files[i],dtype=np.float32,mode='r',shape=(21000,stars,3)) #for oversampling=10
        data = np.memmap(pickle_files[i],dtype=np.float32,mode='r',shape=(2099,stars,3)) #for oversampling=1: 2099, oversampling=5: 10498, oversampling=10: 20995

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
    # start,end=6300,6307
    #print(allfiles[start:end])
    #print(labels[start:end])

    total_stars=star_count
    print('Total # of stars:',total_stars)
    print('     ','took {}s.'.format(datetime.datetime.now()-begintime))
    return labels,alldata,total_stars,allfiles

#print(min_chi2)
# In[8]:


def returnscatter(diffxy):
    #diffxy = inferred - true label values
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
    plt.savefig(savedir+'results.png')
    # plt.show()
    print('average',std_avg,rms_av,bias_av)
    print('model_1',std_m1,rms_m1,bias_m1)
    print('model_2',std_m2,rms_m2,bias_m2)

# In[13]:

#plotresults(testlabels[6000+5412:6000+5412+len(average)],average,labelm1,labelm2)
# if analyzing pande. sample: start = 0
# if analyzing astero. sample: start = 5964

if __name__ == '__main__':
    savedir='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/{}/'.format(dirr)
    print(savedir)
    if not os.path.isdir(savedir):
        print('The path specified does not exist:',savedir)
        sys.exit()

    dp='LLR_gaia/'
    da='LLR_seismic/'
    train_file_names =[dp+'pande_pickle_1',dp+'pande_pickle_2',dp+'pande_pickle_3',da+'astero_final_sample_1',da+'astero_final_sample_2',da+'astero_final_sample_3',da+'astero_final_sample_4']    
    #train_file_names =['pande_pickle_1','pande_pickle_2','pande_pickle_3','astero_final_sample_1','astero_final_sample_2','astero_final_sample_3','astero_final_sample_4']
    train_file_pickle=[i+'_memmap.pickle' for i in train_file_names]
    train_file_txt   =[i+'.txt' for i in train_file_names]

    # In[6]:
    print('Getting training data...')
    train_labels,train_data,total_stars,all_files=gettraindata(train_file_txt,train_file_pickle)
    print('Beginning inference...')
    testlabels,average,labelm1,modelm1,labelm2,min_chi2=getinferredlabels(train_labels,train_data,total_stars,all_files)
    
    np.save(savedir+'testlabels.npy',testlabels)
    np.save(savedir+'average.npy',average)
    np.save(savedir+'labels_m1.npy',labelm1)
    np.save(savedir+'labels_m2.npy',labelm2)
    np.save(savedir+'spectra_m1.npy',modelm1)
    np.save(savedir+'min_chi2.npy',min_chi2)

    start=START#5964
    plotresults(testlabels[start:start+len(average)],average,labelm1,labelm2)


exit()
