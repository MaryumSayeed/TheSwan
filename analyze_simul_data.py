import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename='Astero_Catalogue.txt'
df = pd.read_csv(filename,index_col=False,delimiter=';')
df = df[df['Outlier']==0]
mykics=np.array(df['KICID'])

d='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'

def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms

trueog=np.load(d+'wnoise_level/0.001_result/testlabels.npy')[6135:]

d0='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/jan2020_astero_sample/'
true0=np.load(d0+'testlabels.npy')[6135:6135+50]
idx=np.where(true0>1)[0]

true0=true0[idx]
pred0=np.load(d0+'labels_m1.npy')[0:50][idx]

true01=np.load(d+'wnoise_level/0.0001_result/testlabels.npy')[6135:6135+50][idx]
pred01=np.load(d+'wnoise_level/0.0001_result/labels_m1.npy')[0:50][idx]

true02=np.load(d+'wnoise_level/0.0002_result/testlabels.npy')[6135:6135+50][idx]
pred02=np.load(d+'wnoise_level/0.0002_result/labels_m1.npy')[0:50][idx]

true03=np.load(d+'wnoise_level/0.0003_result/testlabels.npy')[6135:6135+50][idx]
pred03=np.load(d+'wnoise_level/0.0003_result/labels_m1.npy')[0:50][idx]

true04=np.load(d+'wnoise_level/0.0004_result/testlabels.npy')[6135:6135+50][idx]
pred04=np.load(d+'wnoise_level/0.0004_result/labels_m1.npy')[0:50][idx]

true05=np.load(d+'wnoise_level/0.0005_result_old/testlabels.npy')[6135:6135+50][idx]
pred05=np.load(d+'wnoise_level/0.0005_result_old/labels_m1.npy')[0:50][idx]

# true05=np.load(d+'wnoise_level/0.0005_result/testlabels.npy')[6135:6135+50][idx]
# pred05=np.load(d+'wnoise_level/0.0005_result/labels_m1.npy')[0:50][idx]


true1=np.load(d+'wnoise_level/0.001_result/testlabels.npy')[6135:6135+50][idx]
pred1=np.load(d+'wnoise_level/0.001_result/labels_m1.npy')[0:50][idx]

true2=np.load(d+'wnoise_level/0.002_result/testlabels.npy')[6135:6135+50][idx]
pred2=np.load(d+'wnoise_level/0.002_result/labels_m1.npy')[0:50][idx]

true5=np.load(d+'wnoise_level/0.005_result/testlabels.npy')[6135:6135+50][idx]
pred5=np.load(d+'wnoise_level/0.005_result/labels_m1.npy')[0:50][idx]


b0,r0=returnscatter((pred0-true0))
b01,r01=returnscatter((pred01-true01))
b02,r02=returnscatter((pred02-true02))
b03,r03=returnscatter((pred03-true03))
b04,r04=returnscatter((pred04-true04))
b05,r05=returnscatter((pred05-true05))
b1,r1=returnscatter((pred1-true1))
b2,r2=returnscatter((pred2-true2))
b5,r5=returnscatter((pred5-true5))

print(b0,b01,b02,b05,b1,b5)
print(r0,r01,r02,r05,r1,r5)
print(b1,r1)
exit()
# plt.plot(trueog,trueog)
# plt.scatter(true0,pred0)
# plt.show()

# print(true1)
# print(true5)


plt.figure(figsize=(15,5))
plt.rc('lines', markersize=5)  

x=[0,0.0001,0.0002,0.0003,0.0004,0.0005,0.001,0.002,0.005]
y=[r0,r01,r02,r03,r04,r05,r1,r2,r5]/r0

for i in range(0,len(x)):
	print(x[i],y[i])

plt.subplot(121)
plt.plot(trueog,trueog,linestyle='--',c='k',linewidth=1,zorder=0)
plt.scatter(true0,pred0,label='Original: '+str('{0:.2f}'.format(r0)))
plt.scatter(true01,pred01,label='0.0001: '+str('{0:.2f}'.format(r01)))
plt.scatter(true02,pred02,label='0.0002: '+str('{0:.2f}'.format(r02)))
plt.scatter(true03,pred03,label='0.0003: '+str('{0:.2f}'.format(r03)))
plt.scatter(true04,pred04,label='0.0004: '+str('{0:.2f}'.format(r04)))
plt.scatter(true05,pred05,label='0.0005: '+str('{0:.2f}'.format(r05)))
plt.scatter(true1,pred1,label='0.001: '+str('{0:.2f}'.format(r1)))
plt.scatter(true2,pred2,label='0.002: '+str('{0:.2f}'.format(r2)))
plt.scatter(true5,pred5,label='0.005: '+str('{0:.2f}'.format(r5)))
plt.xlabel('Asteroseismic Logg')
plt.ylabel('Inferred Logg')
plt.xlim(2,3.6)
plt.ylim(2,3.6)
plt.legend()

ax2=plt.subplot(122)
plt.grid()
plt.grid(which='major',c='lightgrey',axis='x')
plt.grid(which='minor',c='lightgrey',axis='x',linestyle='--')
ax2.set_axisbelow(True)
plt.scatter(x,y)
plt.xlabel('Noise Factor')
plt.ylabel('Fractional increase relative to no noise added')
plt.xlim(-0.0005,0.006)
plt.minorticks_on()
plt.tight_layout()
plt.show()


