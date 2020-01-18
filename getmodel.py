import pickle, glob
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger

#Load file containing coefficients from fitting the power spectrum of 500 stars:
a=open('coeffs_real_unweighted.pickle','rb')
coeffs_file=pickle.load(a)
a.close()

dataall2, metaall, labels, schmoffsets, coeffs, covs, scatters, chis, chisqs=coeffs_file

nstars  =4
nlabels =2
offsets =[0.,0.]

Params_all = np.zeros((nstars, nlabels))  #16 rows x 2 cols

# Create a grid of 16 different pairs of labels.
#   ie: logg of 3 & Kp range 7-15, stepsize=0.5
#kps=[[3.,i] for i in np.linspace(7,15,nstars+1)] 
kps = [[2.5,8.], [2.5,13.], [4.5,8.], [4.5,13.]]
# kps = [[2.],[2.5],[3.],[3.5],[4.],[4.5]]
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

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
freq    =np.arange(10.011574074074073,277.76620370370375,0.01157)
pdfs=[]
for i in range(0,len(model_all)):
	logg,kp=str(labels[i][0]),str(labels[i][1])
	LABEL=r'$\log g$: {} Kp: {}'.format(logg,kp)
	plt.loglog(freq[0:21000],10.**model_all[i],label=LABEL)
	#plt.title(str(labels[i]))
	plt.ylim([0.001,1e5])
	plt.xlim([9.,300])
	plt.legend()
	plt.xlabel(r'Frequency ($\mu$Hz)',fontsize=20)
	plt.ylabel(r'PSD (ppm$^2/\mu$Hz)',fontsize=20)
	#plt.title('trained on: BO')
	name='{}.pdf'.format(i)
	pdfs.append(name)
	#plt.savefig(d+name)
	#plt.clf()
plt.tight_layout()
savedir='/Users/maryumsayeed/Desktop/HuberNess/iPoster/'
plt.savefig(savedir+'jan2020_cannon_model2.pdf',dpi=50)
plt.show()
