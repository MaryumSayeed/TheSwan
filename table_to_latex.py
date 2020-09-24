import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
filename='LLR_gaia/Gaia_Sample_v1.csv'
# kics = np.loadtxt(filename,skiprows=1,usecols=[0],dtype=str)
# dat = np.loadtxt(filename,skiprows=1,dtype=float)
w=[]
#='RMS: '+str('{0:.2f}'.format(rmsa))
header = ['KICID','Kp', 'Teff', 'Radius','Radp','Radn','True_Logg','Loggp','Loggn','Inferred_Logg','ILoggp','ILoggn','True_Mass','TMassp','TMassn','Inferred_Mass','IMassp','IMassn','SNR','Outlier'] 
df = pd.read_csv(filename,names=header,skiprows=1)

print('Unsorted:',df.head(10))
df.sort_values(by=['KICID'], inplace=True)

# Check the values make sense:

# plt.plot(df['True_Logg'],df['True_Logg'])
# plt.scatter(df['True_Logg'],df['Inferred_Logg'],s=5)
# plt.show(False)


df_short=df.head(10)
# dat=np.array(df_short)
print('Sorted',df_short)# exit()
w=[]
for i in df.index[0:10]:
	d= ' & '
	kic=str(int(df['KICID'][i]))
	teff=str(int(df['Teff'][i]))
	kp = float(df['Kp'][i])
	kp = str('{0:.2f}'.format(kp))

	r,rp,rn=df['Radius'][i],df['Radp'][i],df['Radn'][i]
	r,rp,rn=float(r),float(rp),float(rn)

	r,rp,rn=str('{0:.2f}'.format(r)),'{+'+str('{0:.2f}'.format(rp))+'}','{'+str('{0:.2f}'.format(rn))+'}'

	l,lp,ln=df['True_Logg'][i],df['Loggp'][i],df['Loggn'][i]
	l,lp,ln=float(l),float(lp),float(ln)
	l,lp,ln=str('{0:.2f}'.format(l)),'{+'+str('{0:.2f}'.format(lp))+'}','{'+str('{0:.2f}'.format(ln))+'}'

	il,ilp,iln=df['Inferred_Logg'][i],df['ILoggp'][i],df['ILoggn'][i]
	il,ilp,iln=float(il),float(ilp),float(iln)
	il,ilp,iln=str('{0:.2f}'.format(il)),'{+'+str('{0:.2f}'.format(ilp))+'}','{-'+str('{0:.2f}'.format(iln))+'}'

	tm,tmp,tmn=df['True_Mass'][i],df['TMassp'][i],df['TMassn'][i]
	tm,tmp,tmn=float(tm),float(tmp),float(tmn)
	tm,tmp,tmn=str('{0:.2f}'.format(tm)),'{+'+str('{0:.2f}'.format(tmp))+'}','{'+str('{0:.2f}'.format(tmn))+'}'

	m,mp,mn=df['Inferred_Mass'][i],df['IMassp'][i],df['IMassn'][i]
	m,mp,mn=float(m),float(mp),float(mn)
	m,mp,mn=str('{0:.2f}'.format(m)),'{+'+str('{0:.2f}'.format(mp))+'}','{-'+str('{0:.2f}'.format(mn))+'}'

	out=str(int(df['Outlier'][i]))


	snr=str('{0:.2f}'.format(df['SNR'][i]))	

	line=kic+d+kp+d+teff+d+r'${}^{}_{}$'.format(r,rp,rn)+d+'${}^{}_{}$'.format(l,lp,ln)+d+'${}^{}_{}$'.format(il,ilp,iln)+d+'${}^{}_{}$'.format(tm,tmp,tmn)+d+'${}^{}_{}$'.format(m,mp,mn)+d+snr+d+out+'\\\\'
	print(line)
	w.append(line)

# Save format in file:
# outF = open("Pande_latex_table_for_paper.txt", "w")
outF=open('test_pande_table.txt','w')
for line in w:
  # write line to output file
  outF.write(line)
  outF.write("\n")
outF.close()

filename='LLR_seismic/Seismic_Sample_v1.csv'
header = ['KICID','Kp', 'Teff', 'Radius','Radp','Radn','True_Logg','Loggp','Loggn','Inferred_Logg','ILoggp','ILoggn','True_Mass','TMassp','TMassn','Inferred_Mass','IMassp','IMassn','SNR','Outlier'] 
df = pd.read_csv(filename,names=header,skiprows=1)

print('Unsorted:',df.head(10))
df.sort_values(by=['KICID'], inplace=True)

df_short=df.head(10)
# dat=np.array(df_short)
print('Sorted',df_short)

# df=df[df['Outlier']==0]

w=[]
for i in df.index[0:10]:
	d= ' & '
	kic=str(int(df['KICID'][i]))
	teff=str(int(df['Teff'][i]))
	kp = float(df['Kp'][i])
	kp = str('{0:.2f}'.format(kp))

	r,rp,rn=df['Radius'][i],df['Radp'][i],df['Radn'][i]
	r,rp,rn=float(r),float(rp),float(rn)

	r,rp,rn=str('{0:.2f}'.format(r)),'{+'+str('{0:.2f}'.format(rp))+'}','{'+str('{0:.2f}'.format(rn))+'}'

	l,lp,ln=df['True_Logg'][i],df['Loggp'][i],df['Loggn'][i]
	l,lp,ln=float(l),float(lp),float(ln)
	l,lp,ln=str('{0:.2f}'.format(l)),'{+'+str('{0:.2f}'.format(lp))+'}','{-'+str('{0:.2f}'.format(ln))+'}'

	il,ilp,iln=df['Inferred_Logg'][i],df['ILoggp'][i],df['ILoggn'][i]
	il,ilp,iln=float(il),float(ilp),float(iln)
	il,ilp,iln=str('{0:.2f}'.format(il)),'{+'+str('{0:.2f}'.format(ilp))+'}','{-'+str('{0:.2f}'.format(iln))+'}'

	tm,tmp,tmn=df['True_Mass'][i],df['TMassp'][i],df['TMassn'][i]
	tm,tmp,tmn=float(tm),float(tmp),float(tmn)
	tm,tmp,tmn=str('{0:.2f}'.format(tm)),'{+'+str('{0:.2f}'.format(tmp))+'}','{-'+str('{0:.2f}'.format(tmn))+'}'

	m,mp,mn=df['Inferred_Mass'][i],df['IMassp'][i],df['IMassn'][i]
	m,mp,mn=float(m),float(mp),float(mn)
	m,mp,mn=str('{0:.2f}'.format(m)),'{+'+str('{0:.2f}'.format(mp))+'}','{-'+str('{0:.2f}'.format(mn))+'}'

	out=str(int(df['Outlier'][i]))
	snr=str('{0:.2f}'.format(df['SNR'][i]))	

	line=kic+d+kp+d+teff+d+r'${}^{}_{}$'.format(r,rp,rn)+d+'${}^{}_{}$'.format(l,lp,ln)+d+'${}^{}_{}$'.format(il,ilp,iln)+d+'${}^{}_{}$'.format(tm,tmp,tmn)+d+'${}^{}_{}$'.format(m,mp,mn)+d+snr+d+out+'\\\\'
	w.append(line)

# Save format in file:
# outF = open("Astero_latex_table_for_paper.txt", "w")
outF=open('test_astero_table.txt','w')
for line in w:
  # write line to output file
  outF.write(line)
  outF.write("\n")
outF.close()
