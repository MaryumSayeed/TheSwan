import pandas as pd
import numpy as np
import csv
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])


filename='Pande_Catalogue.txt'

print('Loading...',filename)

df = pd.read_csv(filename,index_col=False,delimiter=';')
df = df[df['Outlier']==0]
mykics=np.array(df['KICID'])

print('Stars',len(mykics))

teff=np.array(df['Teff'])
rad=np.array(df['Radius'])
true=np.array(df['True_Logg'])
pred=np.array(df['Inferred_Logg'])
lum=np.array(df['Radius']**2.*(df['Teff']/5777.)**4.)
mass=np.array(df['Inferred_Mass'])
tmass=np.array(df['True_Mass'])

wnoise_frac=pd.read_csv('pande_wnoise_fraction.txt',delimiter=' ',names=['KICID','More','Less'],skiprows=1)

frac_good=[]
for kic in mykics:
	kic=int(kic)
	if kic in np.array(wnoise_frac['KICID']):
		row   =wnoise_frac.loc[wnoise_frac['KICID']==kic]
		frac=float(row['More'].item())
		frac_good.append(frac)
	else:
		continue 

frac_good=np.array(frac_good)

lim=0
idx=np.where(frac_good>lim)[0]

# original:
trueo,predo,teffo,rado,lumo,masso,tmasso=true,pred,teff,rad,lum,mass,tmass
# after cut:
true,pred,teff,rad,lum,mass,tmass=true[idx],pred[idx],teff[idx],rad[idx],lum[idx],mass[idx],tmass[idx]
print(len(true)/len(frac_good))

print('Plotting...')
plt.rc('font', size=12)                  # controls default text sizes
plt.rc('axes', titlesize=12)             # fontsize of the axes title
plt.rc('axes', labelsize=12)             # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
plt.rc('figure', titlesize=12)           # fontsize of the figure title
plt.rc('axes', linewidth=1)    
plt.rc('lines', markersize = 4)

ss=2

def getplot(true,pred,teff,rad,lum,mass,tmass):
	mlim=2
	idx=np.where(mass<mlim)[0]

	fig=plt.figure(figsize=(20,10))

	ax1=plt.subplot(241)
	plt.plot(true,true,c='k',linestyle='--')
	plt.scatter(true,pred,s=ss,c='r',alpha=0.2,label='M$_{pred}$'+'>{}'.format(mlim))
	im1=plt.scatter(true[idx],pred[idx],c=mass[idx])
	plt.xlabel('Gaia Logg [dex]')
	plt.ylabel('Inferred Logg [dex]')
	lgnd=plt.legend(loc='lower right')
	lgnd.legendHandles[0]._sizes = [60]
	STR='fraction > {}'.format(lim) + '\n' + '# of stars: {}/{}'.format(len(teff),len(teffo))+'={}%'.format(round(len(teff)/len(teffo)*100))
	t=ax1.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax1.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
	

	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferred Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax2=plt.subplot(242)
	bins=np.arange(0.2,6,0.05)
	#bins=70
	xm,ym,_=plt.hist(mass,bins=bins,alpha=0.5,label='Inferred')#facecolor='none',edgecolor='k')
	plt.hist(tmass,bins=bins,alpha=0.5,label='True')#facecolor='none',edgecolor='r')
	mass_peak   =ym[np.argmax(xm)]
	plt.legend()
	plt.xlabel('Mass')
	plt.ylabel('Count')

	STR='M$_{pred}$'+'='+str(round(mass_peak,1))
	t=ax2.text(0.25,0.9,s=STR,color='k',ha='right',va='center',transform = ax2.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
	
	# plt.text(mass_peak,50,s='I: '+,ha='right')
	

	ax3=plt.subplot(243)
	plt.scatter(teffo,lumo,s=ss,c='lightcoral',label='Stars below cut')
	im3=plt.scatter(teff[idx],lum[idx],c=mass[idx],s=10)
	plt.ylim(0.2,200)
	plt.gca().invert_xaxis()
	plt.yscale('log')
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Luminosity [L$_{\\odot}$]')
	lgnd=plt.legend(loc='lower left')
	lgnd.legendHandles[0]._sizes = [60]
	

	ax1_divider = make_axes_locatable(ax3)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im3, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferred Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax4 =plt.subplot(244)
	xy = np.vstack([mass,np.log10(lum)])
	z  = gaussian_kde(xy)(xy)

	plt.scatter(masso,lumo,c='lightcoral',label='Stars below cut',s=ss)
	im4 =plt.scatter(mass,lum,c=z)
	plt.ylim(0.2,200)
	plt.gca().invert_xaxis()
	plt.yscale('log')
	plt.xlabel('Mass [M$_{\\odot}$]')
	plt.ylabel('Luminosity [L$_{\\odot}$]')
	lgnd=plt.legend(loc='lower left')
	lgnd.legendHandles[0]._sizes = [60]
	

	ax1_divider = make_axes_locatable(ax4)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im4, cax=cax1, orientation="horizontal")
	cb1.set_label('log$_{10}$(Count)')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax5 =plt.subplot(245)
	xy = np.vstack([mass,rad])
	z  = gaussian_kde(xy)(xy)

	plt.scatter(masso,rado,c='lightcoral',s=ss,label='Stars below cut')
	im5=plt.scatter(mass,rad,c=z,s=10)
	
	plt.xlabel('Mass [M$_{\\odot}$]')
	plt.ylabel('Radius [R$_{\\odot}$]')
	plt.ylim(0.5,5)
	plt.xlim(0,5)
	plt.gca().invert_xaxis()
	lgnd=plt.legend(loc='lower left')
	lgnd.legendHandles[0]._sizes = [60]
	

	ax1_divider = make_axes_locatable(ax5)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im5, cax=cax1, orientation="horizontal")
	cb1.set_label('log$_{10}$(Count)')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax6 = plt.subplot(246)

	gmass=np.where(tmass>0)[0] # find masses above 0

	xy = np.vstack([tmass[gmass],mass[gmass]])
	z  = gaussian_kde(xy)(xy)

	plt.scatter(tmasso[gmass],masso[gmass],c='lightcoral',s=ss,label='Stars below cut')
	plt.plot(tmass[gmass],tmass[gmass],c='k',linestyle='--')
	im6=plt.scatter(tmass[gmass],mass[gmass],c=z,s=10)
	plt.xlabel('True Mass')
	plt.ylabel('Our Mass')
	lgnd=plt.legend(loc='upper left')
	lgnd.legendHandles[0]._sizes = [60]
	
	ax1_divider = make_axes_locatable(ax6)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im6, cax=cax1, orientation="horizontal")
	cb1.set_label('Count')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax7 =plt.subplot(247)
	plt.scatter(teffo,trueo,c='lightcoral',s=ss,label='Stars below cut')
	im7=plt.scatter(teff[idx],pred[idx],c=mass[idx],s=10)
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Inferred Logg [dex]')
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()
	lgnd=plt.legend(loc='upper left')
	lgnd.legendHandles[0]._sizes = [60]

	ax1_divider = make_axes_locatable(ax7)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im7, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferred Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	# good=np.array(list(set(idx).intersection(gmass)))
	# mass_diff=tmass[good]-mass[good]

	# cmap = plt.get_cmap('cool', 5)  #6=number of discrete cmap bins
	# plt.plot(true,true,c='k',linestyle='--')
	# im7=plt.scatter(true[good],pred[good],c=mass_diff,cmap=cmap,s=10,vmin=-2,vmax=2)
	# plt.xlabel('Gaia Logg [dex]')
	# plt.ylabel('Inferred Logg [dex]')
	# STR='# of stars ' + 'M$_{pred}$'+'<{}'.format(mlim)+' = {}'.format(len(good))
	# t=ax7.text(0.03,0.85,s=STR,color='k',ha='left',va='center',transform = ax7.transAxes)
	# t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))

	# ax1_divider = make_axes_locatable(ax7)
	# cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	# cb1 = fig.colorbar(im7, cax=cax1, orientation="horizontal")
	# cb1.set_label('True - Inferred Mass')
	# cax1.xaxis.set_ticks_position('top')
	# cax1.xaxis.set_label_position('top')
	# cb1.ax.tick_params()

	ax8 =plt.subplot(248)
	plt.scatter(teffo,trueo,c='lightcoral',s=ss,label='Stars below cut')
	im8=plt.scatter(teff[idx],true[idx],c=mass[idx],s=10)
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('True Logg [dex]')
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()
	lgnd=plt.legend(loc='upper left')
	lgnd.legendHandles[0]._sizes = [60]

	ax1_divider = make_axes_locatable(ax8)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im8, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferred Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	plt.tight_layout()
	plt.savefig('cut_{}.png'.format(lim))
	plt.show(False)


getplot(true,pred,teff,rad,lum,mass,tmass)

