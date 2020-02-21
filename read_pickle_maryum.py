import pickle, glob, re
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.stats import mad_std
from sklearn.metrics import r2_score
import matplotlib.image as mpimg
from scipy.stats import chisquare

rc('text', usetex=True)
dd='/Users/maryumsayeed/Desktop/HuberNess/mlearning/powerspectrum/'
#pf=open(dd+'testmetaall_lum.pickle','rb') 
#testmetaall,truemetaall=pickle.load(pf)
#pf.close() 

teststars=np.loadtxt(dd+'Cannon_test_stars.txt',usecols=[0],dtype='str')
file=teststars[0]
f     = ascii.read(file)
freq  = f['freq'] 
print(freq[-1])

def returnscatter(diffxy):
    #diffxy = inferred - true label value
    rms = (np.sum([ (val)**2.  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms


plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.size']=14
plt.rcParams['mathtext.default']='regular'
plt.rcParams['xtick.major.pad']='3'
plt.rcParams['ytick.major.pad']='4'
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

#true=truemetaall[:,1]
#infer=testmetaall[:,1]

#print('TRUE',np.min(true),np.max(true))
#print('INFER',np.min(infer),np.max(infer))

q=True
if q is True:
	pf=open(dd+'testmetaall_cannon_logg.pickle','rb') 
	testmetaall,truemetaall=pickle.load(pf)
	pf.close() 
	true=truemetaall[:,0]
	infer=testmetaall[:,0]
	std=mad_std(infer-true)
	r2=r2_score(true,infer)
	bias,rms=returnscatter(true-infer)

	# a=open(dd+'model_real_unweighted.pickle','rb')
	# model=pickle.load(a)
	# a.close()
	# a=open(dd+'testing_cannon.pickle','rb')
	# testdata=pickle.load(a)
	# a.close()
	#data=testdata[0][:,star,1]
	
	# truekp=truemetaall[:,1]
	# inferkp=truemetaall[:,1]

	# plt.subplot(121)
	#plt.figure(figsize=(10,7))
	#TITLE='Bias: {} RMS: {} ({})'.format(round(bias,2),round(rms,2),len(infer))

	#plt.title(TITLE)
	plt.scatter(true,infer,facecolors='grey', edgecolors='k',s=10)
	plt.plot(true,true,linestyle='dashed',c='k')
	plt.text(2,4.3,'RMS = '+str(round(rms,2)), fontsize=15, ha='left',va='bottom')
	plt.text(2,4.,'Bias = '+str(round(bias,2)), fontsize=15, ha='left',va='bottom')
	plt.xlabel('True (dex)',fontsize=20)
	plt.ylabel('Inferred (dex)',fontsize=20)

	below=np.where(true<=3.5)[0]
	bias_below,rms_below=returnscatter(true[below]-infer[below])
	above=np.where(true>3.5)[0]
	bias_above,rms_above=returnscatter(true[above]-infer[above])
	plt.text(2.8,2.3,r'$\log g < 3.5$: RMS = '+str(round(rms_below,2))+' Bias = '+str(round(bias_below,2)), fontsize=15, ha='left',va='bottom')
	plt.text(2.8,2.,r'$\log g > 3.5$: RMS = '+str(round(rms_above,2))+' Bias = '+str(round(bias_above,2)), fontsize=15, ha='left',va='bottom')
	plt.tight_layout()
	savedir='/Users/maryumsayeed/Desktop/HuberNess/iPoster/'
	plt.savefig(savedir+'cannon_results.pdf',dpi=50)
	plt.show()
	
	

	exit()
	plt.title('coloured by true-Kp')

	plt.subplot(122)
	plt.scatter(true,infer)
	plt.plot(true,true,linestyle='dashed',c='k')
	plt.title('coloured by pred-Kp')
	plt.show()
	exit()
	


	chi2s=[]
	for star in range(0,len(model)):
		exp  =10.**(testdata[0][:,star,1])
		obs  =10.**(model[star])
		chi2,_=chisquare(f_obs=obs,f_exp=exp)
		chi2  =np.log10(chi2)
		chi2s.append(chi2)
	idx=np.where(np.array(chi2s)<8.57)[0]
	trues,infers=[],[]
	ts,lgs,cs=[],[],[]
	for i in idx:
		trues.append(true[i])
		infers.append(infer[i])
		ts.append(teffs[i])
		lgs.append(loggs[i])
		cs.append(chi2s[i])
	cs=np.array(cs)

	f = plt.figure()
	ax1 = f.add_subplot(111)
	plt.scatter(trues,infers,c='k')#,c=cs,cmap='jet',s=10)
	plt.plot(trues,trues,linestyle='dashed',c='k')
	plt.show()
	exit()
	idx=np.where(np.array(cs)<5.5)[0]
	#np.savetxt('rg_chi2_below5.5.txt',teststars[idx],fmt='%s')
	t,i=[],[]
	for ii in idx:
		t.append(trues[ii])
		i.append(infers[ii])
	plt.scatter(t,i,c='blue')
	idx=np.where((np.array(cs)>6.2) & (np.array(infers)<2.8))[0] 
	t,i=[],[]
	for ii in idx:
		t.append(trues[ii])
		i.append(infers[ii])
	#np.savetxt('rg_chi2_above6.2.txt',teststars[idx],fmt='%s')
	plt.scatter(t,i,c='red')
	plt.plot(true,true,linestyle='dashed',c='k')
	plt.text(0.02, 0.9,'$r^2$: '+str(round(r2,2)), fontsize=10, ha='left',va='bottom', transform=ax1.transAxes)
	plt.text(0.02, 0.85,'m.$\sigma$: '+str(round(std,2)), fontsize=10, ha='left',va='bottom', transform=ax1.transAxes)
	plt.xlabel('True')
	plt.ylabel('Inferred')
	#plt.colorbar(label=r'$\rm{Log_{10}(X^2)}$')
	plt.show()
	exit()
	#plt.scatter(teffs,loggs,c=chi2s,s=15,cmap='jet')
	plt.scatter(ts,lgs,c=cs,s=15,cmap='jet')
	plt.colorbar(label=r'$\rm{Log_{10}(X^2)}$')
	#plt.yscale('log')
	plt.xlabel('Teff (K)')
	plt.ylabel('Log(g)')
	plt.gca().invert_yaxis()
	plt.xlim([7500,4000])
	#plt.ylim([10**-1.,300.])
	plt.title('Trained on: RG')
	plt.tight_layout()
	plt.show()
	
	

	#print(len(model))
	exit()


q=False
if q is True:
	ratio=true-infer
	nus=np.array(nus)
	plt.scatter(teffs,ratio,alpha=0.4,s=15)#,c=teffs,cmap='jet_r')
	plt.axhline(0,linestyle='dashed',c='k')
	#idx=np.where((nus>1.87) & (nus<2.5))[0]
	#idx=np.where(np.array(teffs)>6500)[0]
	#idx=np.where((abs(true-infer)<0.1))[0]
	#idx=idx[0:100]
	
	#np.savetxt('tabove6500.1.txt',teststars[idx],fmt='%s')
	#exit()
	#plt.scatter(nus[idx],ratio[idx],alpha=0.5,c='r',s=15)#,c=ratio,cmap='jet')
	#plt.axvspan(np.log10(10.),np.log10(253.),alpha=0.2,color='green')
	#plt.axvline(np.log10(284.),c='k',linestyle='dashed')
	#plt.text(np.log10(284.+10),5.8,s='Nyquist Freq.')
	#plt.gca().invert_yaxis()
	plt.gca().invert_xaxis()
	#plt.xlabel(r'$\rm{\nu_{max}}$')
	plt.xlabel('Teff (K)')
	#plt.ylabel(r'$\rm{Log(g)_{TRUE}-Log(g)_{INFER}}$')
	plt.ylabel(r'$\rm{Kp_{TRUE}-Kp_{INFER}}$')
	#plt.colorbar(label='Teff (K)')
	plt.show()
	exit()


q=False
if q is True:
	plt.subplot(121)
	idx=np.where(true<3.5)[0]
	ifs,ks,rs=[],[],[]
	for i in idx:
		ks.append(kps[i])
		rs.append(ratio[i])
		ifs.append(infer[i])
	plt.scatter(ks,rs,s=25,c=ifs,cmap='jet_r')
	plt.axhline(0,linestyle='dashed',c='k')
	plt.xlabel('Kp',fontsize=20)
	plt.ylabel(r'$\rm{Log(g)_{TRUE}-Log(g)_{INFER}}$')
	plt.title(r'$\rm{Log(g)_{TRUE} < 3.5}$')
	plt.colorbar(orientation='horizontal',label='Inferred log(g)')
	plt.subplot(122)
	idx=np.where(true>3.5)[0]
	ks,rs=[],[]
	ifs,ks,rs=[],[],[]
	for i in idx:
		ks.append(kps[i])
		rs.append(ratio[i])
		ifs.append(infer[i])
	plt.scatter(ks,rs,s=25,c=ifs,cmap='jet_r')
	plt.axhline(0,linestyle='dashed',c='k')
	plt.xlabel('Kp',fontsize=20)
	plt.title(r'$\rm{Log(g)_{TRUE} > 3.5}$')
	plt.colorbar(orientation='horizontal',label='Inferred log(g)')
	plt.tight_layout()
	plt.show()
	exit()

# True (input) values:


# PLOT RESULTS:
q=True
if q is True:
	rms=np.sqrt(np.mean(infer**2.))
	std=mad_std(infer-true)
	rss=np.sum((true-infer)**2.)
	bias=np.sum(infer-true)
	r2=r2_score(true,infer)
	var=np.var(infer)
	nus=np.array(nus)
	f = plt.figure()
	ax1 = f.add_subplot(111)
	plt.scatter(true,infer,marker='o',s=7)#,c=teffs,cmap='gist_rainbow')
	#plt.scatter(true[idx],infer[idx],c='g',s=7,label='s.1/2sigma')
	#print('# of bad stars:',len(idx))
	#np.savetxt('badkp.txt',teststars[idx],fmt='%s')
	#print('# of good stars:',len(idx))
	#plt.scatter(true[idx],infer[idx],c='g',s=7,label='s.1/2sigma')
	#np.savetxt('good.txt',teststars[idx],fmt='%s')
	#plt.colorbar(label='Teff (K)')
	plt.plot(true,true,c='k',linewidth=1,linestyle='dashed')
	plt.minorticks_on()
	plt.xlabel('True')
	plt.ylabel('Inferred')
	#plt.legend()
	plt.title('logg (1 label)')
	#plt.ylim([1.9,4.6])
	plt.text(0.02, 0.9,'$r^2$: '+str(round(r2,2)), fontsize=10, ha='left',va='bottom', transform=ax1.transAxes)
	plt.text(0.02, 0.85,'m.$\sigma$: '+str(round(std,2)), fontsize=10, ha='left',va='bottom', transform=ax1.transAxes)
	plt.text(0.02, 0.8,'RMS: '+str(round(rms,2)), fontsize=10, ha='left',va='bottom', transform=ax1.transAxes)
	plt.tight_layout()
	plt.show()
	exit()


#LAG PLOTS:

'''
lag=5000
plt.figure(figsize=(7,5))
plt.subplot(121)
plt.scatter(teffs,allpower[:,lag],c=kps,s=5)
#plt.xlim([7000,4500])
plt.xlabel('Teff',fontsize=20)
plt.ylabel('PSD',fontsize=20)
plt.gca().invert_xaxis()
plt.colorbar(orientation='horizontal',label=('Kp'))
plt.subplot(122)
plt.scatter(teffs,allpower[:,lag],c=lums,s=5,cmap='jet')
plt.xlabel('Teff',fontsize=20)
plt.gca().invert_xaxis()
plt.colorbar(orientation='horizontal',label='Luminosity (solar)')
plt.suptitle(str(lag)+' '+ str(round(freq[lag],2)),fontsize=10)
#plt.savefig('/Users/maryumsayeed/Desktop/pande/{}.png'.format(str(lag)))
plt.tight_layout()
plt.show()
exit()'''

#GET & PLOT METALLICITY:

findidx=[]
truef,inferf,fehf=[],[],[]
for i in fehs:
	kic,feh=i[0],i[1]
	b=kics.index(kic)
	truef.append(true[b])
	inferf.append(infer[b])
	fehf.append(feh)
if kic in allfeh:
	idx=np.where(allfeh[:,0]==kic)[0]
	feh=allfeh[idx,1][0]
	fehs.append([kic,feh])







'''
ax2 = f.add_subplot(232)
plt.scatter(true,infer,marker='o',s=7,alpha=0.5,c=kps)
plt.plot(true,true,c='k',linewidth=1,linestyle='dashed')
plt.xlim([mmin,mmax])
plt.ylim([mmin,mmax])
plt.minorticks_on()
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
#plt.ylabel('Inferred')

ax2 = f.add_subplot(233)
plt.scatter(true,infer,marker='o',s=7,alpha=0.5,c=teffs,cmap='jet_r')
plt.plot(true,true,c='k',linewidth=1,linestyle='dashed')
plt.xlim([mmin,mmax])
plt.ylim([mmin,mmax])
plt.minorticks_on()
ax2.get_xaxis().set_visible(False)
plt.ylabel('Inferred',rotation=270)
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()


ax4 = f.add_subplot(334,sharex=ax1)
plt.scatter(true,infer-true,marker='o',s=7,alpha=0.5)
ax4.axhline(0,c='k',linestyle='--',linewidth=1)
ax4.get_xaxis().set_visible(False)
plt.minorticks_on()
plt.ylabel('Residual')
plt.ylim([-1,1])


ax4 = f.add_subplot(335,sharex=ax1)
plt.scatter(true,infer-true,marker='o',s=7,alpha=0.5,c=kps)
ax4.axhline(0,c='k',linestyle='--',linewidth=1)
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
plt.minorticks_on()
#plt.ylabel('Residual')
plt.ylim([-1,1])

ax4 = f.add_subplot(336,sharex=ax1)
plt.scatter(true,infer-true,marker='o',s=7,alpha=0.5,c=teffs,cmap='jet_r')
ax4.axhline(0,c='k',linestyle='--',linewidth=1)
ax4.get_xaxis().set_visible(False)
plt.minorticks_on()
plt.ylabel('Residual',rotation=270)
plt.ylim([-1,1])
ax4.yaxis.set_label_position('right')
ax4.yaxis.tick_right()'''

ax2 = f.add_subplot(234,sharex=ax1)
plt.scatter(true,infer,marker='o',s=7,alpha=0.5)
plt.plot(true,true,c='k',linewidth=1,linestyle='dashed')
idx=np.where(abs(infer-true)>std)
plt.scatter(true[idx],infer[idx],marker='o',s=7,alpha=0.5,c='orange',label='$1\sigma$')
idx=np.where(abs(infer-true)>2.*std)
plt.scatter(true[idx],infer[idx],marker='o',s=7,alpha=0.5,c='red',label='$2\sigma$')
plt.legend()
plt.minorticks_on()
ax2.get_yaxis().set_visible(True)
plt.xlabel('True')
plt.ylabel('Inferred')#,rotation=270)
plt.ylim([mmin,mmax])
#ax2.yaxis.set_label_position('right')
#ax2.yaxis.tick_right()

ax3 = f.add_subplot(235,sharex=ax1)
ax3.axhline(0,c='k',linestyle='--',linewidth=1)
plt.scatter(true,true/infer,c=kps,s=7)
plt.xlabel('True')
plt.ylim([-6,6])
plt.minorticks_on()
ax3.get_yaxis().set_visible(False)
cbaxes = f.add_axes([0.05, 0.1, 0.01, 0.8])  # This is the position for the colorbar
cb=plt.colorbar(orientation='vertical',cax=cbaxes)
cb.ax.set_ylabel('Kp')#, rotation=-90, va="bottom")
cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_label_position('left')

#cb.set_label('Kp')#, rotation=270)

ax4 = f.add_subplot(236,sharex=ax1)
ax4.axhline(0,c='k',linestyle='--',linewidth=1)
plt.scatter(true,true/infer,c=teffs,s=7,cmap='jet_r')
plt.xlabel('True')
plt.ylabel('True/Infer',rotation=270)
ax4.yaxis.set_label_position('right')
ax4.yaxis.tick_right()
plt.ylim([-6,6])
plt.minorticks_on()
cbaxes = f.add_axes([0.95, 0.1, 0.01, 0.8])  # This is the position for the colorbar
cb=plt.colorbar(orientation='vertical',cax=cbaxes)
cb.set_label('Teff (K)')#, rotation=270)
cb.ax.yaxis.set_ticks_position('right')
cb.ax.yaxis.set_label_position('right')

f.subplots_adjust(hspace=0,wspace=0)
#plt.tight_layout()

plt.show()
exit()

#idx=np.where(abs(true-infer)>std)
#plt.scatter(true[idx],infer[idx],c='r',s=7,marker='o',label='$1\sigma$')#,c=truekp)

#idx=np.where(infer<0.)
#plt.scatter(true[idx],infer[idx],c='g',s=7,marker='o',label='$1\sigma$')#,c=truekp)

#plt.legend()
#plt.colorbar()
#plt.ylim([-2.4,-2.1])
#plt.savefig('/Users/maryumsayeed/Desktop/testing_choices/pix_choices/'+name.replace('-','_')+'.png')
plt.show()
exit()

