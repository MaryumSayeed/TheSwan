import pandas as pd
import numpy as np
import csv
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])

sample='Astero'
filename='{}_Catalogue.txt'.format(sample)
chi2_vals=pd.read_csv('{}_chi2.txt'.format(sample),index_col=False,delimiter=' ')#,names=['KICID','Chi2'])
df = pd.read_csv(filename,index_col=False,delimiter=';')

if sample=='Pande':
	xlim,ylim=2.,4.8
if sample=='Astero':
	xlim,ylim=1.,4.7

mykics=np.array(df['KICID'])
true=df[df['Outlier']==0]['True_Logg']
pred=df[df['Outlier']==0]['Inferred_Logg']
kp  =df[df['Outlier']==0]['Kp']
teff=df[df['Outlier']==0]['Teff']
rad =df[df['Outlier']==0]['Radius']
mass=np.log10(df[df['Outlier']==0]['Inferred_Mass'])
lum =np.log10(rad**2.*(teff/5777.)**4.)

trueo=df[df['Outlier']==1]['True_Logg']
predo=df[df['Outlier']==1]['Inferred_Logg']
kpo  =df[df['Outlier']==1]['Kp']
teffo=df[df['Outlier']==1]['Teff']
rado =df[df['Outlier']==1]['Radius']
masso=np.log10(df[df['Outlier']==1]['Inferred_Mass'])
lumo =np.log10(rado**2.*(teffo/5777.)**4.)


good_flags=[]  # not outliers
bad_flags=[]   # outliers
for i in range(0,len(mykics)):
	kic=int(mykics[i])
	row   =df.loc[df['KICID']==kic]
	flag=float(row['Outlier'].item())
	if flag == 0:
		good_flags.append(i)
	else:
		bad_flags.append(i)

chi2=np.log10(chi2_vals['Chi2'])

print(len(chi2),len(true),len(trueo),len(good_flags),len(bad_flags))

print('Plotting...')

plt.rc('font', size=10)                  # controls default text sizes
plt.rc('axes', titlesize=10)             # fontsize of the axes title
plt.rc('axes', labelsize=10)             # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)            # fontsize of the tick labels
plt.rc('ytick', labelsize=10)            # fontsize of the tick labels
plt.rc('figure', titlesize=12)           # fontsize of the figure title
plt.rc('axes', linewidth=1)    
plt.rc('lines', markersize=4)
plt.rc('xtick.major',pad=2)
plt.rc('ytick.major',size=2)
plt.rc('ytick.major',pad=2)
# plt.rc("legend.marker",facecolor='k')
ms=4

def get_cbar(ax,im,label,n):
	if (n % 2) == 0:
		dc='right'
		pad='12%'
	else:
		dc='left'
		pad='15%'
	pad='18%'
	ax1_divider = make_axes_locatable(ax)
	cax1 = ax1_divider.append_axes(dc, size="5%", pad=pad)
	cb1 = fig.colorbar(im, cax=cax1)#, orientation="horizontal")
	cb1.set_label(label)
	cax1.yaxis.set_ticks_position(dc)
	cax1.yaxis.set_label_position(dc)
	cb1.ax.tick_params()

def get_plot(n,clist,olist,label):
	ax=plt.subplot(3,2,n)#,sharex=ax0)

	plt.plot(true,true,c='k',linestyle='--')
	plt.grid(color='gray', linestyle='dashed')
	ax.set_xlim(xlim,ylim)
	ax.set_ylim(xlim,ylim)
	ax.set_axisbelow(True)

	im=plt.scatter(true,pred,c=clist,s=ms)
	plt.scatter(trueo,predo,c=olist,marker='^',vmin=clist.min(),vmax=clist.max(),s=ms*3)
	
	if (n % 2) == 0:
		dc='right'
	else:
		dc='left'
		# plt.ylabel('Inferred Logg')

	ax.yaxis.set_ticks_position(dc)
	ax.yaxis.set_label_position(dc)
	
	if (n==5) or (n==6):
		ax.xaxis.set_ticks_position('bottom')	
	else:
		locs,labels=plt.xticks()
		ax.tick_params(which='major',axis='x',direction='inout')
		ax.set_xticklabels([]*len(labels))
		ax.xaxis.set_ticks_position('bottom')
	
	if n ==4 :
		legend_elements = [Line2D([0], [0],  marker='^',  color='white', \
			label='Outliers',markerfacecolor='none',markeredgecolor='k', markersize=10)]
		lgnd=plt.legend(handles=legend_elements,loc='lower right')

	get_cbar(ax,im,label,n)

fig=plt.figure(figsize=(8,8))

get_plot(1,teff,teffo,'Effective Temperature [K]')#,1)
get_plot(2,rad,rado,'Radius [R$_{\\odot}$]')#,1)
get_plot(3,lum,lumo,'log$_{10}$(Luminosity [L$_{\\odot}$])')#,1)
get_plot(4,kp,kpo,'Apparent Magnitude')
get_plot(5,mass,masso,'log$_{10}$(Inferred Mass [M$_{\\odot}$])')
get_plot(6,chi2[good_flags],chi2[bad_flags],'log$_{10}$($\chi^2$)')

# fig.text(0.2, 0.5, 'Inferred Logg', va='center', rotation='vertical')
# fig.text(0.5, 0.01, 'Gaia Logg', va='center')
plt.tight_layout()
plt.subplots_adjust(hspace=.1,wspace=0.0)
plt.savefig('{}_appendix.pdf'.format(sample),dpi=50)
plt.show(False)
