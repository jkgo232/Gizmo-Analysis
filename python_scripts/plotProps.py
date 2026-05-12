import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

font = {'weight': 'normal','size': 28}

import matplotlib as mpl
mpl.use('pdf')

def read_file(filename, flag):
 if flag==1:
  num=33
 if flag==0:
  num=31
 data=np.loadtxt(str(filename), skiprows=num)
 snap_num2=data[:,0] 
 z2=data[:,1] 
 t_age2=data[:,2]
 stellar_mass2=data[:,3] 
 gas_mass2=data[:,4] 
 sfr2=data[:5]
 Z_star2=data[:,6] 
 Z_gas2=data[:,7] 
 log_OH2=data[:,8]
 Sigma_star2=data[:,9] 
 Sigma_gas2=data[:,10]
 r_half_star2=data[:,11] 
 r_half_gas2=data[:,12]
 T_gas_median2=data[:,13] 
 rho_gas_median2=data[:,14]
 disk_mass2=data[:,15] 
 bulge_mass2=data[:,16] 
 disk_r_half2=data[:,17] 
 bulge_r_half2=data[:,18]
 bar_radius2=data[:,19] 
 bar_ellipticity2=data[:,20]
 sigma_star_los2=data[:,21]
 sigma_gas_x2=data[:,22] 
 sigma_gas_y2=data[:,23] 
 sigma_gas_z2=data[:,24]
 n_star2=data[:,25] 
 n_gas2=data[:,26] 
 if flag==1:
  BH_mass2=data[:,27] 
  BH_lum2=data[:,28]

 return z2, t_age2, stellar_mass2, gas_mass2 

red_10, time_10, prop_10, prop2_10 = read_file('galaxy_properties_zm1.txt', 0)
red_00, time_00, prop_00, prop2_00 = read_file('galaxy_properties_zm0.txt', 0)
red_12, time_12, prop_12, prop2_12 = read_file('galaxy_properties_2BH_zm1.txt', 0)
red_02, time_02, prop_02, prop2_02 = read_file('galaxy_properties_2BH_zm0.txt', 0)

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(time_02, np.log10(prop2_02), label='$M_{gas}$, 2BH, ID=0', c='red')
#ax.plot(time_00, np.log10(prop2_00), label='$M_{gas}$ noBH, ID=0', c='red')
ax.plot(time_02, np.log10(prop_02), label='$M_{*}$, 2BH, ID=0', c='red', linestyle='dotted')
#ax.plot(time_00, np.log10(prop_00), label='$M_{*}$ noBH, ID=0', c='red', linestyle='dotted')

ax.plot(time_12, np.log10(prop2_12), label='$M_{gas}$, 2BH, ID=1', c='black')
#ax.plot(time_10, np.log10(prop2_10), label='$M_{gas}$ noBH, ID=1', c='black')
ax.plot(time_12, np.log10(prop_12), label='$M_{*}$, 2BH, ID=1', c='black', linestyle='dotted')
#ax.plot(time_10, np.log10(prop_10), label='$M_{*}$ noBH, ID=1', c='black', linestyle='dotted')

#print(time, np.log10(prop2), np.log10(prop))
#print(time, np.log10(prop2), np.log10(prop))
ax.set_ylabel('$M/M_{\odot}$', fontdict=font)
ax.set_xlabel('$\mathrm{Time [Gyr]}$', fontdict=font)
#add redshift to top
ages=[0.4719, 1.1689, 3.2802, 4.2750, 5.8585, 8.5991, 10.7578, 13.7]
age=[10, 5, 2, 1.5, 1, 0.5, 0.3, 0]
ax3 = ax.twiny()
ax3.set_xticks(ages)
ax3.tick_params(labelsize=22, which='both', length=8, width=2, right=True, direction='in')
ax3.set_xticklabels(age)
ax3.set_xlabel('$\mathrm{Redshift}$', fontdict=font)
ax3.set_xlim(np.min(0.3), 13.8)
ax.set_xlim(np.min(0.3), 13.8)
ax.tick_params(labelsize=22, length=8, width=2, direction='in', right=True)
#plt.xlabel('$Time [Gyr]$')
ax.set_ylim(6, 11.2)
#ax.set_xlim(9, 0)
#plt.xlim(0.5, 14)
ax.set_title('$HOP$')
ax.legend(fontsize=20)
plt.savefig('78956masses_hop.png')

