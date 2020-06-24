import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.gridspec as gridspec

wnoise_frac=pd.read_csv('pande_wnoise_fraction.txt',delimiter=' ',names=['KICID','More','Less'],skiprows=1)

# idx=np.where(wnoise_frac['More']>0.01)[0]

df=pd.read_csv('Pande_Catalogue.txt',index_col=False,delimiter=';')

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

lum=rad**2.*(teff/5777.)**4.
lumo=rado**2.*(teffo/5777.)**4.
allstars=np.concatenate([keep,outliers])

frac_good=[]
for kic in keep:
	# kic=str(kic.lstrip('0'))
	kic=str(kic)
	if kic in np.array(wnoise_frac['KICID']):
		row   =wnoise_frac.loc[wnoise_frac['KICID']==kic]
		frac=float(row['More'].item())
		frac_good.append(frac)
	else:
		continue 

frac_bad=[]#np.zeros(len(outliers))
for kic in outliers:
	# kic=str(kic.lstrip('0'))
	kic=str(kic)
	if kic in np.array(wnoise_frac['KICID']):
		row   =wnoise_frac.loc[wnoise_frac['KICID']==kic]
		frac=float(row['More'].item())
		frac_bad.append(frac)
	else:
		continue 
frac_good=np.array(frac_good)
frac_bad=np.array(frac_bad)

allfrac=np.concatenate([frac_bad,frac_good])

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


ms=9
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
	plt.scatter(true[idx1],pred[idx1],c=frac_good[idx1],s=ms)
	
	im=plt.scatter(trueo[idx2],predo[idx2],c=frac_bad[idx2],s=ms,marker="^",vmin=lim,vmax=allfrac.max(),label='Outliers')
	
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
	ax2.scatter(trueo[idx2],trueo[idx2]-predo[idx2],s=ms,c=frac_bad[idx2],marker='^')
	ax2.axhline(0,c='k',linestyle='dashed')
	ax2.set_ylim(-0.5,0.5)

fig=plt.figure(figsize=(8,9))
gs = gridspec.GridSpec(9, 8,hspace=0)

get_plot(0,1)
get_plot(0.4,2)
get_plot(0.5,3)
get_plot(0.6,4)
plt.tight_layout()
plt.show(False)
# exit()
plt.savefig('pande_wnoise_fraction.png')
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

plt.clf()
get_mass_plot(0,tmass,imass,tmasso,imasso,1)
get_mass_plot(0.4,tmass,imass,tmasso,imasso,2)
get_mass_plot(0.5,tmass,imass,tmasso,imasso,3)
get_mass_plot(0.6,tmass,imass,tmasso,imasso,4)
plt.tight_layout()
plt.show(False)
# plt.savefig('pande_wnoise_mass.png')





