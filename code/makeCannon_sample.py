import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Load Kps:
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
kps      =list(df['kic_kepmag'])

pande_stars=np.loadtxt('LLR_gaia/pande_final_sample_full.txt',usecols=[1])
astero_stars=np.loadtxt('LLR_seismic/astero_final_sample_full.txt',usecols=[1])

print('# of total Pande stars:',len(pande_stars),'\n','# of total Astero. stars',len(astero_stars))


b1,b2,b3,b4,b5,b6=[],[],[],[],[],[]
for i in range(0,len(pande_stars)):
	logg=pande_stars[i]
	if logg < 2.5:
		b1.append(i)
	elif logg >= 2.5 and logg < 3.:
		b2.append(i)
	elif logg >= 3. and logg < 3.5:
		b3.append(i)
	elif logg >= 3.5 and logg < 4:
		b4.append(i)
	elif logg >= 4. and logg < 4.5:
		b5.append(i)
	elif logg >= 4.5 and logg < 5.:
		b6.append(i)

n=600
pande_stars_Cannon_idx=b1[0:n]+b2[0:n]+b3[0:n]+b4[0:n]+b5[0:n]+b6[0:n]
# print('pande_stars_Cannon_idx')
# for i in [b1[0:n],b2[0:n],b3[0:n],b4[0:n],b5[0:n],b6[0:n]]:
# 	print('   ',len(i))

b1,b2,b3,b4,b5,b6,b7,b8,b9,b10=[],[],[],[],[],[],[],[],[],[]
for i in range(0,len(astero_stars)):
	logg=astero_stars[i]
	# if logg < 0.5:
	# 	b1.append(i)
	# elif logg >= 0.5 and logg < 1.:
	# 	b2.append(i)
	# elif logg >= 1. and logg < 1.5:
	# 	b3.append(i)
	# elif logg >= 1.5 and logg < 2.:
	# 	b4.append(i)
	if logg >= 2. and logg < 2.5:
		b5.append(i)
	elif logg >= 2.5 and logg < 3.:
		b6.append(i)
	elif logg >= 3. and logg < 3.5:
		b7.append(i)
	elif logg >= 3.5 and logg < 4.:
		b8.append(i)
	elif logg >= 4. and logg < 4.5:
		b9.append(i)
	elif logg >= 4.5 and logg < 5.:
		b10.append(i)

astero_stars_Cannon_idx=b1[0:n]+b2[0:n]+b3[0:n]+b4[0:n]+b5[0:n]+b6[0:n]+b7[0:n]+b8[0:n]+b9[0:n]+b10[0:n]
pande_stars_files =np.loadtxt('LLR_gaia/pande_final_sample_full.txt',usecols=[0],dtype='str')
astero_stars_files=np.loadtxt('LLR_seismic/astero_final_sample_full.txt',usecols=[0],dtype='str')

# print('astero_stars_Cannon_idx')
# for i in [b1[0:n],b2[0:n],b3[0:n],b4[0:n],b5[0:n],b6[0:n],b7[0:n],b8[0:n],b9[0:n],b10[0:n]]:
# 	print('   ',len(i))
# # Create training samples:
pande_stars_Cannon_train=[]
train_sample=[]
for idx in pande_stars_Cannon_idx:
	logg=pande_stars[idx]
	file=pande_stars_files[idx]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	kp=kps[kp_kics.index(kic)]
	pande_stars_Cannon_train.append([file,logg,kp])
	
astero_stars_Cannon_train=[]
for idx in astero_stars_Cannon_idx:
	logg=astero_stars[idx]
	file=astero_stars_files[idx]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	kp=kps[kp_kics.index(kic)]
	astero_stars_Cannon_train.append([file,logg,kp])
	
# Create testing samples:
# First, get stars NOT used for training:
pande_idx=np.arange(0,len(pande_stars),1)
pande_idx_test=list(set(pande_idx)-set(pande_stars_Cannon_idx))
print('this should add up')
print(len(pande_idx),'=',len(pande_stars_Cannon_idx),'+',len(pande_idx_test))

astero_idx=np.arange(0,len(astero_stars),1)
astero_idx_test=list(set(astero_idx)-set(astero_stars_Cannon_idx))
print('this should add up')
print(len(astero_idx),'=',len(astero_stars_Cannon_idx),'+',len(astero_idx_test))

b1,b2,b3,b4,b5,b6=[],[],[],[],[],[]
for i in pande_idx_test:
	logg=pande_stars[i]
	if logg < 2.5:
		b1.append(i)
	elif logg >= 2.5 and logg < 3.:
		b2.append(i)
	elif logg >= 3. and logg < 3.5:
		b3.append(i)
	elif logg >= 3.5 and logg < 4:
		b4.append(i)
	elif logg >= 4. and logg < 4.5:
		b5.append(i)
	elif logg >= 4.5 and logg < 5.:
		b6.append(i)

n=800
pande_stars_test_idx=b1[0:n]+b2[0:n]+b3[0:n]+b4[0:n]+b5[0:n]+b6[0:n]


# print('pande_stars_test_idx')
# for i in [b1[0:n],b2[0:n],b3[0:n],b4[0:n],b5[0:n],b6[0:n]]:
# 	print('   ',len(i))
# # print('...',[print(len(i)) for i in [b1[0:n],b2[0:n],b3[0:n],b4[0:n],b5[0:n],b6[0:n]]])

b1,b2,b3,b4,b5,b6,b7,b8,b9,b10=[],[],[],[],[],[],[],[],[],[]
for i in astero_idx_test:
	logg=astero_stars[i]
	# if logg < 0.5:
	# 	b1.append(i)
	# elif logg >= 0.5 and logg < 1.:
	# 	b2.append(i)
	# elif logg >= 1. and logg < 1.5:
	# 	b3.append(i)
	# elif logg >= 1.5 and logg < 2.:
	# 	b4.append(i)
	if logg >= 2. and logg < 2.5:
		b5.append(i)
	elif logg >= 2.5 and logg < 3.:
		b6.append(i)
	elif logg >= 3. and logg < 3.5:
		b7.append(i)
	elif logg >= 3.5 and logg < 4.:
		b8.append(i)
	elif logg >= 4. and logg < 4.5:
		b9.append(i)
	elif logg >= 4.5 and logg < 5.:
		b10.append(i)
	
astero_stars_test_idx=b1[0:n]+b2[0:n]+b3[0:n]+b4[0:n]+b5[0:n]+b6[0:n]+b7[0:n]+b8[0:n]+b9[0:n]+b10[0:n]
# print('astero_stars_test_idx')
# for i in [b1[0:n],b2[0:n],b3[0:n],b4[0:n],b5[0:n],b6[0:n],b7[0:n],b8[0:n],b9[0:n],b10[0:n]]:
# 	print('   ',len(i))

# Pick stars each from astero and Pande samples for testing :
pande_stars_Cannon_test=[]
for idx in pande_stars_test_idx:
	logg=pande_stars[idx]
	file=pande_stars_files[idx]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	kp=kps[kp_kics.index(kic)]
	pande_stars_Cannon_test.append([file,logg,kp])
	
astero_stars_Cannon_test=[]
for idx in astero_stars_test_idx:
	logg=astero_stars[idx]
	file=astero_stars_files[idx]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	kp=kps[kp_kics.index(kic)]
	astero_stars_Cannon_test.append([file,logg,kp])
	
print(len(astero_stars_Cannon_train), len(astero_stars_Cannon_test), len(pande_stars_Cannon_train), len(pande_stars_Cannon_test), )

# Create one common training sample:
train_sample=pande_stars_Cannon_train+astero_stars_Cannon_train
# train_sample=astero_stars_Cannon_train

print('# of training stars',len(train_sample))

# Create one common testing sample:
test_sample=pande_stars_Cannon_test+astero_stars_Cannon_test
# test_sample=astero_stars_Cannon_test
print('# of testing stars',len(test_sample))

# Make sure the stars in training and testing samples are in final astero/pande samples:
filename='LLR_seismic/Seismic_Catalogue.txt'
dfa = pd.read_csv(filename,index_col=False,delimiter=';')
mykicsa=np.array(dfa['KICID'])

filename='LLR_gaia/Gaia_Catalogue.txt'
dfp = pd.read_csv(filename,index_col=False,delimiter=';')
mykicsp=np.array(dfp['KICID'])

mykics=np.concatenate([mykicsa,mykicsp])

# c=0
# for kic in training_kics:
#     if kic in mykics:
#         c=c+1
# d=0
# for kic in testing_kics:
#     if kic in mykics:
#         d=d+1
        
# print(c,d)


# Double check the stars in both training and testing are not overlapping:
training_kics=[]
for i in range(0,len(train_sample)):
	file=train_sample[i][0]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	training_kics.append(kic)

testing_kics=[]
for i in range(0,len(test_sample)):
	file=test_sample[i][0]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	testing_kics.append(kic)

common_kics=list(set(training_kics) & set(testing_kics))
print('# of stars common in train/test samples:',len(common_kics))
print(common_kics)

keep_training_idx=[]
for i in range(len(training_kics)):
	kic=training_kics[i]
	if kic in mykics:
		keep_training_idx.append(i)

keep_testing_idx=[]
for i in range(len(testing_kics)):
	kic=testing_kics[i]
	if kic in mykics:
		keep_testing_idx.append(i)

# Take out the common KIC from test sample:
new_test_sample=[]
for i in range(0,len(test_sample)):
	file=test_sample[i][0]
	kic=re.search('kplr(.*)-', file).group(1)
	kic=int(kic.lstrip('0'))
	if kic not in common_kics:
		new_test_sample.append(i)

ts=[]
for i in new_test_sample:
	ts.append(test_sample[i])

test_sample=ts

train_sample,test_sample=np.array(train_sample),np.array(test_sample)
train_sample=train_sample[keep_training_idx]
test_sample =test_sample[keep_testing_idx]


new_test_sample=train_sample # i switched them because i wanted more in training than testing
new_train_sample=test_sample
test_sample,train_sample=new_test_sample,new_train_sample
print('Training stars',len(train_sample))
print('Testing stars',len(test_sample))

test_logg=np.array([float(i[1]) for i in test_sample])
train_logg=np.array([float(i[1]) for i in train_sample])

# Visualize distribution of logg for testing and training stars:
train_giants=np.where(train_logg<3.5)[0]
train_dwarfs=np.where(train_logg>=3.5)[0]
test_giants=np.where(test_logg<3.5)[0]
test_dwarfs=np.where(test_logg>=3.5)[0]

print('===Training giants',len(train_giants))
print('===Training dwarfs',len(train_dwarfs))
print('===Testing giants',len(test_giants))
print('===Testing dwarfs',len(test_dwarfs))


c1  ='#404788FF'
c2  ='#55C667FF'
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('font', size=15)                  # controls default text sizes
plt.rc('axes', titlesize=15)             # fontsize of the axes title
plt.rc('axes', labelsize=15)             # fontsize of the x and y labels

plt.rcParams['font.size']=15
plt.rc('legend',fontsize=15)
bins=np.arange(2.,4.8,0.2)
plt.hist(train_logg,label='Train Sample',edgecolor=c2,facecolor='none',bins=bins,linestyle='dashed',linewidth=2)
plt.hist(test_logg,label='Test Sample',edgecolor=c1,facecolor='none',bins=bins,linestyle='dashed',linewidth=2)

plt.xlabel(r'$\log g$ [dex]')
plt.ylabel('Counts')
plt.legend()
plt.tight_layout()
# plt.savefig('/Users/maryumsayeed/Desktop/HuberNess/iPoster/Cannon_sample_hist.pdf',dpi=50)
plt.savefig('cannon_hist.png',dpi=100,bbox_inches='tight')
plt.show(False)

# if training on logg only, uncomment below:
test_sample =np.array([[i[0],i[1]] for i in test_sample])
train_sample=np.array([[i[0],i[1]] for i in train_sample])

d1=''
# Save testing and training samples:
# np.savetxt('Cannon_train_stars.txt',train_sample,fmt='%s')
# np.savetxt('Cannon_test_stars.txt',test_sample,fmt='%s')

