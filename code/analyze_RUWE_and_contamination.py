#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


kepler_catalogue=pd.read_csv('/Users/maryumsayeed/Desktop/HuberNess/mlearning/hrdmachine/GKSPC_InOut_V4.csv')#,skiprows=1,delimiter=',',usecols=[0,1])


# In[3]:


kepler_catalogue


# In[4]:


pande_cat=pd.read_csv('LLR_gaia/Gaia_Sample_v3.csv')
pande_cat


# In[5]:


bad_kics=pande_cat[pande_cat['Outlier']==1]['KICID']
good_kics=pande_cat[pande_cat['Outlier']==0]['KICID']
bad_kics,good_kics=np.array(bad_kics),np.array(good_kics)


# In[6]:


bruwe_vals=[]
for kic in bad_kics:
    row=kepler_catalogue[kepler_catalogue['KIC']==kic]
    ruwe=row['RUWE'].item()
    bruwe_vals.append(ruwe)
    
gruwe_vals=[]
for kic in good_kics:
    row=kepler_catalogue[kepler_catalogue['KIC']==kic]
    ruwe=row['RUWE'].item()
    gruwe_vals.append(ruwe)


# In[7]:


bruwe_vals,gruwe_vals=np.array(bruwe_vals),np.array(gruwe_vals)


# In[8]:


print(len(np.where(gruwe_vals>1.2)[0])+len(np.where(bruwe_vals>1.2)[0]),len(bruwe_vals)+len(gruwe_vals))
print(502/4605*100)


# In[9]:


# plt.hist(gruwe_vals,np.arange(0.8,3.6,0.2))
plt.hist(bruwe_vals,np.arange(0.8,3.6,0.2))
above=np.where(bruwe_vals>1.2)[0]
print('# of outliers',len(bad_kics),'# with RUWE above 1.2:',len(above),len(above)/len(bruwe_vals)*100)
# print('# with RUWE above 1.2:',len(above),len(above)/len(bruwe_vals)*100)
print('max RUWE',np.max(bruwe_vals))
print(bad_kics[above])


# In[12]:


# np.savetxt('LLR_gaia/pande_outliers_check_contamination.txt',bad_kics,fmt='%s')
MAST=pd.read_csv('LLR_gaia/pande_contam_vals_from_MAST.txt',delimiter='\t')
con_vals=np.array(MAST['contamination'][1:]).astype(float)
plt.hist(con_vals)
print('Max value:',np.max(con_vals))


# In[ ]:




