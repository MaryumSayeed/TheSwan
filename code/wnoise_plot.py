import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.gridspec as gridspec
from astropy.stats import mad_std


NAME='astero'
NAME='pande'
wnoise_frac=pd.read_csv('LLR_gaia/{}_wnoise_frac.txt'.format(NAME),delimiter=' ',names=['KICID','Fraction','Radius'],skiprows=1)

# idx=np.where(wnoise_frac['More']>0.01)[0]

df=pd.read_csv('LLR_gaia/Gaia_Catalogue.txt',index_col=False,delimiter=';')
# df=pd.read_csv('LLR_seismic_final/Astero_Catalogue.txt',index_col=False,delimiter=';')

keep=np.array(df[df['Outlier']==0]['KICID'])
outliers=np.array(df[df['Outlier']==1]['KICID'])

true=np.array(df[df['Outlier']==0]['True_Logg'])
pred=np.array(df[df['Outlier']==0]['Inferred_Logg'])

trueo=np.array(df[df['Outlier']==1]['True_Logg'])
predo=np.array(df[df['Outlier']==1]['Inferred_Logg'])

teff=np.array(df[df['Outlier']==0]['Teff'])
teffo=np.array(df[df['Outlier']==1]['Teff'])

rad=np.array(df[df['Outlier']==0]['Radius'])
rado=np.array(df[df['Outlier']==1]['Radius'])

snr=np.array(df[df['Outlier']==0]['SNR'])
snro=np.array(df[df['Outlier']==1]['SNR'])

tloggp=np.array(df[df['Outlier']==0]['TLoggp'])
tloggn=np.array(df[df['Outlier']==0]['TLoggn'])



mass=np.array(df[df['Outlier']==0]['Inferred_Mass'])
radius=np.array(df[df['Outlier']==0]['Radius'])

cmap = plt.get_cmap('viridis', 8)  #6=number of discrete cmap bins	
plt.scatter(mass,radius,s=10,c=abs(true-pred),cmap=cmap)
plt.xlim(0,5)
plt.ylim(0.5,5)
plt.colorbar()
plt.gca().invert_xaxis()
# plt.show()

i1=np.where(mass>3.5)[0]
print(len(i1))
i2=np.where(abs(true-pred)[i1]>0.25)[0]
print(len(i2))
exit()

# i2=np.where( (mass >2) & (radius<1.1) )[0]
print(true[i2],pred[i2],true[i2]-pred[i2])
print(mass[i2],radius[i2])
print(len(i2))
exit()

# plt.scatter(tloggp, true-pred,s=10)
# plt.show()
# true=np.array(df['True_Logg'])
# pred=np.array(df['Inferred_Logg'])
# tloggp=np.array(df['TLoggp'])
# tloggn=np.array(df['TLoggn'])


# i1=np.where(tloggp<=0.1)[0]
# plt.plot(true,true,c='k')
# plt.scatter(true,pred,s=10)
# plt.scatter(true[i1],pred[i1],s=10,label='Errors<=0.1')
# plt.legend()

# plt.hist(snr,bins=np.arange(0.1,1.,0.1),normed=True,alpha=0.5)
# plt.hist(snro,bins=np.arange(0.1,1.,0.1),normed=True,edgecolor='k',facecolor='lightgrey',alpha=0.5)
# plt.show()
exit()


# true=np.array(df['True_Logg'])
# pred=np.array(df['Inferred_Logg'])
# snr=np.array(df['SNR'])

# diff=pred-true

# i1=np.where(diff>0.2)[0]
# i2=np.where(diff<0.2)[0]
# print(np.median(snr[i1]))
# print(np.median(snr[i2]))
# plt.scatter(true,pred,c=snr,s=10)
# plt.colorbar()
# plt.show()
# exit()

# print(np.min(snr),np.min(snro))
# print(np.max(snr),np.max(snro))
# exit()
# plt.scatter(true,pred,c=snr,vmin=0,vmax=1)
# plt.scatter(trueo,predo,c=snro,marker='^',vmin=0,vmax=1)
# plt.colorbar()
# plt.show()
# exit()

lum=rad**2.*(teff/5777.)**4.
lumo=rado**2.*(teffo/5777.)**4.
allstars=np.concatenate([keep,outliers])

frac_good=snr
frac_bad=snro

allfrac=np.concatenate([frac_bad,frac_good])
print(len(frac_good),len(frac_bad))
# exit()
# plt.figure(figsize=(8,6))

# plt.scatter(teff,lum,c=frac_good,s=10)
# plt.scatter(teffo,lumo,c=frac_bad,marker='^',s=10)
# plt.yscale('log')
# plt.xlabel('Effective Temperature [K]')
# plt.ylabel('Luminosity [solar]')

# plt.colorbar()
# plt.gca().invert_xaxis()
# plt.tight_layout()
# plt.show(False)
# plt.savefig('wnoise_hr.png')
# exit()
fig=plt.figure(figsize=(6,6))
gs = gridspec.GridSpec(4, 4,hspace=0)
ms=5
a,b,c,d=0,3,0,4
lim=0.2
idx1=np.where(frac_good>lim)[0]
idx2=np.where(frac_bad>lim)[0]
ax=plt.subplot(gs[a:b,c:d])
plt.plot(true,true,c='k',linestyle='--')
im=plt.scatter(true[idx1],pred[idx1],c=frac_good[idx1],s=ms,vmin=lim,vmax=allfrac.max())
plt.scatter(trueo,predo,c='grey',alpha=0.5,s=ms,marker="^",vmin=lim,vmax=allfrac.max(),label='Outliers')
im=plt.scatter(trueo[idx2],predo[idx2],c=frac_bad[idx2],s=ms,marker="^",vmin=lim,vmax=allfrac.max(),label='Outliers')
plt.xticks([])

std=mad_std(true[idx1]-pred[idx1])
str1='SNR > {} '.format(lim)+'({0:2.0f}%)'.format((len(idx1)+len(idx2))*100/(len(keep)+len(outliers)))# + '\n' + str('{0:2.0f}'.format()+'%'
str2=r'$\sigma_{\mathrm{mad}}$ = '+str('{0:.2f}'.format(std))
STR=str1+'\n'+str2+' dex'
t=ax.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax.transAxes)
t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
a=((len(idx1)+len(idx2))/(len(keep)+len(outliers)))*100

ax1_divider = make_axes_locatable(ax)
cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
cb1 = fig.colorbar(im, cax=cax1, orientation="horizontal")
# cb1.set_label('Fraction of power above white noise')
cax1.xaxis.set_ticks_position('top')
cax1.xaxis.set_label_position('top')
cb1.ax.tick_params()
a,b,c,d=3,4,0,4

ax2 = plt.subplot(gs[a:b, c:d])
ax2.scatter(true[idx1],true[idx1]-pred[idx1],s=ms,c=frac_good[idx1],vmin=lim,vmax=allfrac.max(),zorder=10)
ax2.scatter(trueo[idx2],trueo[idx2]-predo[idx2],s=ms,c=frac_bad[idx2],vmin=lim,vmax=allfrac.max(),marker='^')
ax2.axhline(0,c='k',linestyle='dashed')
if NAME=='astero':
	ax.set_xlim(1,4.7)
	ax.set_ylim(1,4.7)
	ax2.set_xlim(1,4.7)
	ax2.set_ylim(-0.7,0.7)
else:
	ax2.set_ylim(-0.7,0.7)
# if n==2 or n==3:
ax.set_ylabel('Inferred Logg [dex]',labelpad=15)
ax2.set_ylabel('$\Delta$ Log g [dex]')
ax2.set_xlabel('True Log g [dex]')
d='/Users/maryumsayeed/LLR_updates/Aug24/'
plt.savefig(d+'{}.png'.format(lim),bbox_inches='tight')
plt.show(True)
exit()
ms=2
def get_plot(lim,n):
	if n==1:
		a,b,c,d=0,3,0,4
	if n==2:
		a,b,c,d=0,3,4,8
	if n==3:
		a,b,c,d=5,8,0,4
	if n==4:
		a,b,c,d=5,8,4,8
	idx1=np.where(frac_good>lim)[0]
	idx2=np.where(frac_bad>lim)[0]
	ax=plt.subplot(gs[a:b,c:d])
	plt.plot(true,true,c='k',linestyle='--')
	im=plt.scatter(true[idx1],pred[idx1],c=frac_good[idx1],s=ms,vmin=lim,vmax=allfrac.max())
	#im=plt.scatter(trueo[idx2],predo[idx2],c=frac_bad[idx2],s=ms,marker="^",vmin=lim,vmax=allfrac.max(),label='Outliers')
	plt.xticks([])

	std=mad_std(true[idx1]-pred[idx1])
	str1='SNR > {} '.format(lim)+'({0:2.0f}%)'.format((len(idx1)+len(idx2))*100/(len(keep)+len(outliers)))# + '\n' + str('{0:2.0f}'.format()+'%'
	str2=r'$\sigma_{\mathrm{mad}}$ = '+str('{0:.2f}'.format(std))
	STR=str1+'\n'+str2+' dex'
	t=ax.text(0.03,0.9,s=STR,color='k',ha='left',va='center',transform = ax.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
	a=((len(idx1)+len(idx2))/(len(keep)+len(outliers)))*100

	ax1_divider = make_axes_locatable(ax)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im, cax=cax1, orientation="horizontal")
	# cb1.set_label('Fraction of power above white noise')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	if n==1:
		a,b,c,d=3,4,0,4
	if n==2:
		a,b,c,d=3,4,4,8
	if n==3:
		a,b,c,d=8,9,0,4
	if n==4:
		a,b,c,d=8,9,4,8

	ax2 = plt.subplot(gs[a:b, c:d])
	ax2.scatter(true[idx1],true[idx1]-pred[idx1],s=ms,c=frac_good[idx1],zorder=10)
	# ax2.scatter(trueo[idx2],trueo[idx2]-predo[idx2],s=ms,c=frac_bad[idx2],marker='^')
	ax2.axhline(0,c='k',linestyle='dashed')
	if NAME=='astero':
		ax.set_xlim(1,4.7)
		ax.set_ylim(1,4.7)
		ax2.set_xlim(1,4.7)
		ax2.set_ylim(-0.7,0.7)
	else:
		ax2.set_ylim(-0.7,0.7)
	# if n==2 or n==3:
		ax.set_ylabel('Inferred Logg [dex]',labelpad=15)
		ax2.set_ylabel('$\Delta$ Log g [dex]')

fig=plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(9, 8,hspace=0)

if NAME=='astero':
	get_plot(0.5,1)
	get_plot(0.6,2)
	get_plot(0.7,3)
	get_plot(0.8,4)
	fig.text(0.5, 0.0,'Asteroseismic Logg [dex]' , ha='center')

else:
	get_plot(0.4,1)
	get_plot(0.45,2)
	get_plot(0.5,3)
	get_plot(0.55,4)
	fig.text(0.5, 0.0,'Gaia Logg [dex]' , ha='center')

# fig.text(0.5, 0.98,'Power above white noise' , ha='center')
# fig.text(0.0, 0.5,'Power above white noise' , ha='center',va='center',rotation='vertical')

plt.tight_layout()

plt.show(False)
# exit()
plt.savefig('{}_wnoise_fraction.png'.format(NAME),dpi=100,bbox_inches='tight')
exit()
tmass=np.array(df[df['Outlier']==0]['True_Mass'])
imass=np.array(df[df['Outlier']==0]['Inferred_Mass'])

tmasso=np.array(df[df['Outlier']==1]['True_Mass'])
imasso=np.array(df[df['Outlier']==1]['Inferred_Mass'])

def get_mass_plot(lim,true,pred,trueo,predo,n):
	if n==1:
		a,b,c,d=0,3,0,4
	if n==2:
		a,b,c,d=0,3,4,8
	if n==3:
		a,b,c,d=5,8,0,4
	if n==4:
		a,b,c,d=5,8,4,8
	idx1=np.where(frac_good>lim)[0]
	idx2=np.where(frac_bad>lim)[0]

	ax=plt.subplot(gs[a:b,c:d])
	plt.plot(true,true,c='k',linestyle='--')
	im=plt.scatter(true[idx1],pred[idx1],c=frac_good[idx1],s=ms,vmin=lim,vmax=allfrac.max(),)
	plt.ylim(0,7)
	# im=plt.scatter(trueo[idx2],predo[idx2],c=frac_bad[idx2],s=ms,marker="^",vmin=lim,vmax=allfrac.max(),label='Outliers')
	
	STR='fraction > {}'.format(lim) + '\n' + str('{0:2.0f}'.format((len(idx1)+len(idx2))*100/(len(keep)+len(outliers))))+'%'
	t=ax.text(0.03,0.92,s=STR,color='k',ha='left',va='center',transform = ax.transAxes)
	t.set_bbox(dict(facecolor='none',edgecolor='none'))#, alpha=0.5, edgecolor='red'))
	a=((len(idx1)+len(idx2))/(len(keep)+len(outliers)))*100

	ax1_divider = make_axes_locatable(ax)
	cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
	cb1 = fig.colorbar(im, cax=cax1, orientation="horizontal")
	# cb1.set_label('Fraction of power above white noise')
	cax1.xaxis.set_ticks_position('top')
	cax1.xaxis.set_label_position('top')
	cb1.ax.tick_params()

	if n==1:
		a,b,c,d=3,4,0,4
	if n==2:
		a,b,c,d=3,4,4,8
	if n==3:
		a,b,c,d=8,9,0,4
	if n==4:
		a,b,c,d=8,9,4,8


	ax2 = plt.subplot(gs[a:b, c:d])
	ax2.scatter(true[idx1],true[idx1]-pred[idx1],s=ms,c=frac_good[idx1],zorder=10)
	# ax2.scatter(trueo[idx2],trueo[idx2]-predo[idx2],s=ms,c=frac_bad[idx2],marker='^')
	ax2.axhline(0,c='k',linestyle='dashed')
	ax2.set_ylim(-5,2)

	ax.set_xlim(0.2,np.max(true)+0.5)
	ax2.set_xlim(0.2,np.max(true)+0.5)

plt.clf()
get_mass_plot(0,tmass,imass,tmasso,imasso,1)
get_mass_plot(0.5,tmass,imass,tmasso,imasso,2)
get_mass_plot(0.6,tmass,imass,tmasso,imasso,3)
get_mass_plot(0.7,tmass,imass,tmasso,imasso,4)
plt.tight_layout()
plt.show(False)
plt.savefig('astero_wnoise_mass.png')





