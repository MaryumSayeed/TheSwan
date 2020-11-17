import pandas as pd
import numpy as np
import csv
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])

pande=False

if pande is True:
	filename='Pande_Catalogue.txt'
	df = pd.read_csv(filename,index_col=False,delimiter=';')
	df = df[df['Outlier']==0]
	mykics=np.array(df['KICID'])

	print('Stars',len(mykics))
	fig=plt.figure(figsize=(20,10))
	print('Plotting...')
	plt.rc('font', size=12)                  # controls default text sizes
	plt.rc('axes', titlesize=12)             # fontsize of the axes title
	plt.rc('axes', labelsize=12)             # fontsize of the x and y labels
	plt.rc('xtick', labelsize=12)            # fontsize of the tick labels
	plt.rc('ytick', labelsize=12)            # fontsize of the tick labels
	plt.rc('figure', titlesize=12)           # fontsize of the figure title
	plt.rc('axes', linewidth=1)    
	plt.rc('lines', markersize = 4)

	teff=np.array(df['Teff'])
	pred=np.array(df['Inferred_Logg'])
	true=np.array(df['True_Logg'])
	mass=np.array(df['Inferred_Mass'])
	tmass=np.array(df['True_Mass'])
	

	lum=df['Radius']**2.*(df['Teff']/5777.)**4.
	lum=np.array(lum)

	idx=np.where((mass<2) & (tmass<2))[0]
	#idx=np.where(mass<3)[0]
	print(np.min(tmass[idx]),np.max(tmass[idx]))
	print(np.min(mass[idx]),np.max(mass[idx]))
	print(np.min(true),np.min(pred))
	print(np.max(true),np.max(pred))
	# exit()
	ax1=plt.subplot(241)
	plt.scatter(teff,lum,c='r',s=5,alpha=0.2)
	im1=plt.scatter(teff[idx],lum[idx],c=mass[idx],s=10,vmin=np.min(mass[idx]),vmax=np.max(mass[idx]))
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Luminosity [L$_{\\odot}$]')
	plt.yscale('log')
	plt.xlim(4225,7500)
	plt.gca().invert_xaxis()

	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferered Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax2=plt.subplot(242)
	plt.scatter(teff,lum,s=10,c='r',alpha=0.2)
	im2=plt.scatter(teff[idx],lum[idx],c=tmass[idx],s=10,vmin=np.min(mass[idx]),vmax=np.max(mass[idx]))
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Luminosity [L$_{\\odot}$]')
	plt.yscale('log')
	plt.xlim(4225,7500)
	plt.gca().invert_xaxis()

	ax1_divider = make_axes_locatable(ax2)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im2, cax=cax1, orientation="horizontal")
	cb1.set_label('True Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax3=plt.subplot(243,sharey=ax2)
	plt.scatter(teff,lum,c='grey',s=10,alpha=0.2)#,c=tmass,s=10)
	idx2=np.where((mass<1.2) & (mass>0.8))[0]

	im3=plt.scatter(teff[idx2],lum[idx2],c=mass[idx2],s=10,vmin=np.min(mass[idx]),vmax=np.max(mass[idx]))
	plt.yscale('log')
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Luminosity [L$_{\\odot}$]')
	STR='M$_{pred}$=0.8-1.2'+'\n'+'N={} stars'.format(len(idx2))
	t=ax3.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax3.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
	plt.xlim(4225,7500)
	plt.gca().invert_xaxis()

	ax1_divider = make_axes_locatable(ax3)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im3, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferred Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	plt.subplot(244)
	bins=np.arange(0.5,6,0.2)
	# bins=50
	plt.hist(mass,bins=bins,alpha=0.5,label='Inferred')#facecolor='none',edgecolor='k')
	plt.hist(tmass,bins=bins,alpha=0.5,label='True (isoclassify)')#facecolor='none',edgecolor='r')
	plt.xlabel('Mass [M$_{\\odot}$]')
	plt.ylabel('Count')
	plt.legend()


	ax4=plt.subplot(245)
	plt.scatter(teff,true,c='r',s=10,alpha=0.2)#,c=tmass,s=10)
	im4=plt.scatter(teff[idx],true[idx],c=mass[idx],s=10,vmin=np.min(mass[idx]),vmax=np.max(mass[idx]))
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('True Logg [dex]')
	plt.xlim(4225,7500)
	plt.ylim(2.1,4.7)
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()

	ax1_divider = make_axes_locatable(ax4)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im4, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferred Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax5=plt.subplot(246)
	plt.scatter(teff,pred,c='r',s=10,alpha=0.2)#,c=tmass,s=10)
	im5=plt.scatter(teff[idx],pred[idx],c=mass[idx],s=10,vmin=np.min(mass[idx]),vmax=np.max(mass[idx]))
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Inferred Logg [dex]')
	plt.xlim(4225,7500)
	plt.ylim(2.1,4.7)
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()

	ax1_divider = make_axes_locatable(ax5)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im5, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferred Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax7=plt.subplot(247)
	plt.scatter(teff,true,s=10,c='r',alpha=0.2)
	im7=plt.scatter(teff[idx],true[idx],c=tmass[idx],s=10,vmin=np.min(mass[idx]),vmax=np.max(mass[idx]))
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('True Logg [dex]')
	plt.xlim(4225,7500)
	plt.ylim(2.1,4.7)
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()

	ax1_divider = make_axes_locatable(ax7)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im7, cax=cax1, orientation="horizontal")
	cb1.set_label('True Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax8=plt.subplot(248)
	plt.scatter(teff,pred,s=10,c='r',alpha=0.2)
	im8=plt.scatter(teff[idx],pred[idx],c=tmass[idx],s=10,vmin=np.min(mass[idx]),vmax=np.max(mass[idx]))
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Inferred Logg [dex]')
	plt.xlim(4225,7500)
	plt.ylim(2.1,4.7)
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()

	ax1_divider = make_axes_locatable(ax8)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im8, cax=cax1, orientation="horizontal")
	cb1.set_label('True Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	plt.tight_layout()
	plt.savefig('11.png')
	plt.show(False)

# exit()
grav_const  =6.67e-8   #cm^3*g^-1*s^-2
solar_radius=6.956e10  #cm
solar_mass  =1.99e33     #g
	
astero=True
if astero is True:
	filename='Astero_Catalogue.txt'
	print('Loading...',filename)

	df = pd.read_csv(filename,index_col=False,delimiter=';')
	df = df[df['Outlier']==0]
	mykics=np.array(df['KICID'])


	print('Calculating inferred and true mass...')

	print('Loading astero. catalogues...')
	astero1=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/labels_full.txt',delimiter=' ',skiprows=1,usecols=[0,1,2,3],names=['KIC','Teff','Logg','Lum'])
	astero2=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/rg_yu.txt',delimiter='|',skiprows=1,usecols=[0,1,3,7,9],names=['KIC','Teff','Logg','Mass','Radius'])
	chaplin=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/Chaplin_2014.tsv',skiprows=35,delimiter=';',names=['KIC','Mass','Errp','Errn'])

	teff=np.array(df['Teff'])
	rad=np.array(df['Radius'])
	true_logg=np.array(df['True_Logg'])
	pred_logg=np.array(df['Inferred_Logg'])
	lum=np.array(df['Radius']**2.*(df['Teff']/5777.)**4.)
	mass=np.array(df['Inferred_Mass'])
	idx=np.where(mass<3)[0]
	true_mass=np.array(df['True_Mass'])

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

	fig=plt.figure(figsize=(20,10))
	ax1=plt.subplot(241)
	plt.plot(true_logg,true_logg,c='k',linestyle='--')
	plt.scatter(true_logg,pred_logg,s=ss,c='r')
	im1=plt.scatter(true_logg[idx],pred_logg[idx],c=mass[idx])
	plt.xlabel('Asteroseismic Logg [dex]')
	plt.ylabel('Inferred Logg [dex]')

	ax1_divider = make_axes_locatable(ax1)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferred Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	plt.subplot(242)
	bins=np.arange(0.2,6,0.05)
	#bins=70
	xm,ym,_=plt.hist(mass,bins=bins,alpha=0.5,label='Inferred')#facecolor='none',edgecolor='k')
	plt.hist(true_mass,bins=bins,alpha=0.5,label='True')#facecolor='none',edgecolor='r')
	mass_peak   =ym[np.argmax(xm)]
	plt.text(mass_peak,1000,s=str(round(mass_peak,1)),ha='right')
	plt.legend()
	plt.xlabel('Mass')
	plt.ylabel('Count')

	ax3=plt.subplot(243)
	plt.scatter(teff,lum,s=ss,c='r')
	im3=plt.scatter(teff[idx],lum[idx],c=mass[idx])
	plt.gca().invert_xaxis()
	plt.yscale('log')
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('Luminosity [L$_{\\odot}$]')


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
	im4 =plt.scatter(mass,lum,c=z)
	plt.gca().invert_xaxis()
	plt.yscale('log')
	plt.xlabel('Mass [M$_{\\odot}$]')
	plt.ylabel('Luminosity [L$_{\\odot}$]')


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

	im5=plt.scatter(mass,rad,c=z)
	plt.gca().invert_xaxis()
	plt.xlabel('Mass [M$_{\\odot}$]')
	plt.ylabel('Radius [R$_{\\odot}$]')


	ax1_divider = make_axes_locatable(ax5)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im5, cax=cax1, orientation="horizontal")
	cb1.set_label('log$_{10}$(Count)')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax6 = plt.subplot(246)
	
	gmass=np.where(true_mass>0)[0] # find masses above 0

	xy = np.vstack([true_mass[gmass],mass[gmass]])
	z  = gaussian_kde(xy)(xy)

	plt.plot(true_mass[gmass],true_mass[gmass],c='k',linestyle='--')
	im6=plt.scatter(true_mass[gmass],mass[gmass],c=z)
	plt.xlabel('True Mass')
	plt.ylabel('Our Mass')

	ax1_divider = make_axes_locatable(ax6)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im6, cax=cax1, orientation="horizontal")
	cb1.set_label('Count')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax7 =plt.subplot(247)

	good=np.array(list(set(idx).intersection(gmass)))
	mass_diff=true_mass[good]-mass[good]

	cmap = plt.get_cmap('cool', 5)  #6=number of discrete cmap bins
	im7=plt.scatter(true_logg[good],pred_logg[good],c=mass_diff,cmap=cmap,s=10,vmin=-2,vmax=2)
	plt.xlabel('Asteroseismic Logg [dex]')
	plt.ylabel('Inferred Logg [dex]')
	STR='M$_{pred}$<3'+' N={} stars'.format(len(good))
	t=ax7.text(0.03,0.85,s=STR,color='k',ha='left',va='center',transform = ax7.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))



	ax1_divider = make_axes_locatable(ax7)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im7, cax=cax1, orientation="horizontal")
	cb1.set_label('True - Inferred Mass')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	ax8 =plt.subplot(248)

	im8=plt.scatter(teff[idx],true_logg[idx],c=mass[idx],s=10)
	plt.xlabel('Effective Temperature [K]')
	plt.ylabel('True Logg [dex]')
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()


	ax1_divider = make_axes_locatable(ax8)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im8, cax=cax1, orientation="horizontal")
	cb1.set_label('Inferred Mass [M$_{\\odot}$]')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	plt.tight_layout()
	plt.savefig('new.png')
	plt.show(False)




