import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Load Kps:
kpfile   ='/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/KIC_Kepmag_Berger2018.csv'
df       =pd.read_csv(kpfile,usecols=['KIC','kic_kepmag'])
kp_kics  =list(df['KIC'])
kps      =list(df['kic_kepmag'])

pande_stars=np.loadtxt('pande_final_sample.txt',usecols=[1])
astero_stars=np.loadtxt('astero_final_sample_full.txt',usecols=[1])




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

n=200
pande_stars_Cannon_idx=b1[0:n]+b2[0:n]+b3[0:n]+b4[0:n]+b5[0:n]+b6[0:n]


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
pande_stars_files =np.loadtxt('pande_final_sample.txt',usecols=[0],dtype='str')
astero_stars_files=np.loadtxt('astero_final_sample_full.txt',usecols=[0],dtype='str')


# Create training samples:
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
print(len(pande_idx),len(pande_stars_Cannon_idx),len(pande_idx_test))

astero_idx=np.arange(0,len(astero_stars),1)
astero_idx_test=list(set(astero_idx)-set(astero_stars_Cannon_idx))
print('this should add up')
print(len(astero_idx),len(astero_stars_Cannon_idx),len(astero_idx_test))

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

n=650
pande_stars_test_idx=b1[0:n]+b2[0:n]+b3[0:n]+b4[0:n]+b5[0:n]+b6[0:n]

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
	

# Create one common training sample:
train_sample=pande_stars_Cannon_train+astero_stars_Cannon_train
print(len(train_sample))

# Create one common testing sample:
test_sample=pande_stars_Cannon_test+astero_stars_Cannon_test
print(len(test_sample))


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

print(len(test_sample))
print(len(new_test_sample))
# print(new_test_sample)
test_sample=ts

# Save testing and training samples:
np.savetxt('Cannon_train_stars.txt',train_sample,fmt='%s')
np.savetxt('Cannon_test_stars.txt',test_sample,fmt='%s')

