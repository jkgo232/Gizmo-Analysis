import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.308)

cl=2.998
m_p=1.6726e-24
kb=1.3806e-16
xx=0.59 *(5./3.-1.)*((m_p/kb)*(2.93363e+53/2.93363e+43))
gamma=2./3.
print('snap, red, time, log_stellar_mass, r4max, log_gas_mass, r0max, eff_r, log_sfr, log_sfrinr0, log_dens4, np.log10(Z40/0.02), log_temp0g, log_dens0g, log_press0, log_entropy0, log_mol00, log_HII00, log_OH, np.log10(Z00/0.02), log(BHmass), log(BHpmass), logfedd, logacc_gs, logacc_msyr, logpower, logLmech, nump_g, nump_all')

#gas <KeysViewHDF5 ['Acceleration', 'CoolingRate', 'Coordinates', 'Density', 'ElectronAbundance', 'HeatingRate', 'HydroHeatingRate', 'InternalEnergy', 'Masses', 'MetalCoolingRate', 'Metallicity', 'MolecularMassFraction', 'NetHeatingRateQ', 'NeutralHydrogenAbundance', 'ParticleChildIDsNumber', 'ParticleIDGenerationNumber', 'ParticleIDs', 'Potential', 'RateOfChangeOfInternalEnergy', 'SmoothingLength', 'StarFormationRate', 'Temperature', 'TimeStep', 'Velocities', 'VelocityDivergence', 'Vorticity']>
#stars <KeysViewHDF5 ['Acceleration', 'Coordinates', 'DensityAtParticleLocation', 'Masses', 'Metallicity', 'ParticleChildIDsNumber', 'ParticleIDGenerationNumber', 'ParticleIDs', 'Potential', 'StellarFormationTime', 'TimeStep', 'Velocities']>
#SMBH <KeysViewHDF5 ['Acceleration', 'BH_Mass', 'BH_Mdot', 'BH_NProgs', 'BH_Specific_AngMom', 'Coordinates', 'DensityAtParticleLocation', 'Masses', 'Metallicity', 'ParticleChildIDsNumber', 'ParticleIDGenerationNumber', 'ParticleIDs', 'Potential', 'SinkInitialMass', 'StellarFormationTime', 'TimeStep', 'Velocities']>

data=np.loadtxt('./outputs/HaloProps15.txt')
snap0=data[:,0]
z0=data[:,1]
rvir0=data[:,2]
Mvir0=data[:,3]
hx0=data[:,4]
hy0=data[:,5]
hz0=data[:,6]

tdata=np.loadtxt('./outputs/lookuptable.txt')
ta=tdata[:,0]
tz=tdata[:,1]
tage=tdata[:,2]


snaps=np.arange(1951,2005,1)[0::5]
for j in snaps:
 ind=np.where(snap0==j)[0]
 rvir=rvir0[ind]
 rlim=0.1*rvir

 if j<774:
  f=h5py.File("/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_45og/snapshot_"+str(j)+".hdf5", 'r')
 else:
  f=h5py.File("/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_45og/snapshot_"+str(j)+".hdf5", 'r')
 pos5=np.array(f['PartType5']['Coordinates'])
 mass5=np.array(f['PartType5']['BH_Mass'])
 pmass5=np.array(f['PartType5']['Masses'])
 acc5=np.array(f['PartType5']['BH_Mdot'])
 fedd15=acc5[0] / ( (4*np.pi* (6.672e-8) * (1.6726e-24) / (0.1*2.9979e10*6.65245e-25) ) * ( ((3.085678e21/0.678) /1.e5) * (mass5[0]) ) )
 acc2_15=acc5[0]*(2.93363e+43/4.55115e+16) #g/s
 acc3_15=acc2_15*5.03018108651911e-34/3.1688765e-8 #msun/year
 power15=(0.1*acc2_15*cl*cl) #/1e20
 Lmech15=(0.1*0.5*acc2_15*cl*cl*0.1*0.1) #/1e20
 xc=pos5[:,0]
 yc=pos5[:,1]
 zc=pos5[:,2]

 red=f['Header'].attrs['Redshift']
 ag=f['Header'].attrs['Time']
 idk = (np.abs(red - tz)).argmin()
 time=np.round(tage[idk], 2)

 mass4=np.array(f['PartType4']['Masses'])
 vel4=np.array(f['PartType4']['Velocities'])
 pos4=np.array(f['PartType4']['Coordinates'])
 pot4=np.array(f['PartType4']['Potential'])
 dens4=np.array(f['PartType4']['DensityAtParticleLocation'])* (1.473e-21 /1.6726e-24)
 sft4 = np.array(f['PartType4']['StellarFormationTime'])
 Z4=np.array(f['PartType4']['Metallicity'][:,0])

 mass0=np.array(f['PartType0']['Masses'])
 vel0=np.array(f['PartType0']['Velocities'])
 pos0=np.array(f['PartType0']['Coordinates'])
 sfr0 = np.array(f['PartType0']['StarFormationRate'])
 ID0 = np.array(f['PartType0']['ParticleIDs'])
 temp0=np.array(f['PartType0']['Temperature'])
 temp2=np.array(f['PartType0']['InternalEnergy']) * xx
 dens0=np.array(f['PartType0']['Density']) * (1.473e-21 /1.6726e-24) 
 mol0=np.array(f['PartType0']['MolecularMassFraction'])
 HII0=np.array(f['PartType0']['NeutralHydrogenAbundance'])
 Z0=np.array(f['PartType0']['Metallicity'][:,0])
 He0=np.array(f['PartType0']['Metallicity'][:,1])
 O0=np.array(f['PartType0']['Metallicity'][:,4])
 press0= dens0 * temp2 * kb
 entropy0= temp2/(dens0**gamma)

 x4=pos4[:,0]
 y4=pos4[:,1]
 z4=pos4[:,2]

 vx4=vel4[:,0]
 vy4=vel4[:,1]
 vz4=vel4[:,2]

 x=pos0[:,0]
 y=pos0[:,1]
 z=pos0[:,2]

 vx=vel0[:,0]
 vy=vel0[:,1]
 vz=vel0[:,2]

 #find potential minimum and center velocity
 b=np.where(np.sqrt((x4-xc)*(x4-xc)+(y4-yc)*(y4-yc)+(z4-zc)*(z4-zc))<=30.)[0]
 x4f=np.take(x4, b)
 y4f=np.take(y4, b)
 z4f=np.take(z4, b)
 vx4f=np.take(vx4, b)
 vy4f=np.take(vy4, b)
 vz4f=np.take(vz4, b)
 pot0=np.take(pot4, b)

 ind_cp=np.where(pot0==np.min(pot0))
 xc=np.take(x4f, ind_cp[0])
 yc=np.take(y4f, ind_cp[0])
 zc=np.take(z4f, ind_cp[0])

 ind_cpv=np.where(np.sqrt((x4-xc)*(x4-xc)+(y4-yc)*(y4-yc)+(z4-zc)*(z4-zc))<=1.)
 vxc=np.median(np.take(vx4, ind_cpv[0]))
 vyc=np.median(np.take(vy4, ind_cpv[0]))
 vzc=np.median(np.take(vz4, ind_cpv[0]))

 #center
 vx=vx-vxc
 vy=vy-vyc
 vz=vz-vzc
 x=x-xc
 y=y-yc
 z=z-zc

 vx4=vx4-vxc
 vy4=vy4-vyc
 vz4=vz4-vzc
 x4=x4-xc
 y4=y4-yc
 z4=z4-zc

 r4=np.sqrt(x4*x4+y4*y4+z4*z4)
 r=np.sqrt(x*x+y*y+z*z)

 b = np.where(r4 < rlim)[0]

 mass4=np.take(mass4, b)
 r4=np.take(r4, b)
 massinr4=np.sum(mass4)
 halfmass=(massinr4/2.)*1.e10
 dens4=np.median(np.take(dens4, b))
 Z40=np.sum(np.take(Z4, b)*mass4*1.e10)/(massinr4*1.e10)

 smass=0.
 eff_r=0.
 while smass < halfmass:
        eff_r+=0.01
        sdr=np.where(r4<=eff_r)[0]
        smass=np.sum(np.take(mass4, sdr)*1.e10)

 sft00=np.take(sft4, b)
 zgg= ((1/sft00)-1)
 birth=(cosmo.age(zgg).value)
 now=time
 nsind=(np.where(np.absolute(birth-now)<=0.03))[0]
 newstars=np.take(mass4, nsind)
 SFRavg=np.sum(newstars)*(1.e10/3.e7)


 c = np.where(r < rlim)[0]
 r=np.take(r, c)
 mass0g=np.take(mass0, c)
 massinr0=np.sum(mass0g)
 sfr0g=np.take(sfr0, c)
 ID=np.take(ID0, c)

 sfrinr0=np.sum(sfr0g)
 temp0g=np.median(np.take(temp0, c))
 temp2g=np.median(np.take(temp2, c))
 dens0g=np.median(np.take(dens0, c))
 press0=np.median(np.take(press0, c))
 entropy0=np.median(np.take(entropy0, c))
 Z00=np.sum(np.take(Z0, c)*mass0g*1.e10)/(massinr0*1.e10)
 He00=np.sum(np.take(He0, c)*mass0g*1.e10)/(massinr0*1.e10)
 O00=np.sum(np.take(O0, c)*mass0g*1.e10)/(massinr0*1.e10)
 mol00=np.sum(np.take(mol0, c)*mass0g*1.e10)/(massinr0*1.e10)
 HII00=np.sum(np.take(HII0, c)*mass0g*1.e10)/(massinr0*1.e10)
 H=((massinr0*1.e10)- np.sum(np.take(Z0, c)*mass0g*1.e10)-np.sum(np.take(He0, c)*mass0g*1.e10))*1.98847/1.6735 #number H atoms in galaxy/1e57
 O=np.sum(np.take(O0, c)*mass0g*1.e10)*1.98847/26.561 #number 0 atoms in galaxy/1e57
 OH=12+np.log10(O/H)

 jetindall=np.where(ID0==1913298393)[0]
 jetindgal=np.where(ID==1913298393)[0]
 numj_all=len(jetindall)
 numj_gal=len(jetindgal)



 print(j, np.round(red, 2), time, np.round(np.log10(massinr4*1.e10), 2), np.round(np.max(r4), 2), np.round(np.log10(massinr0*1.e10), 2), np.round(np.max(r), 2), np.round(eff_r, 2), np.round(np.log10(SFRavg), 2), np.round(np.log10(sfrinr0), 2), np.round(np.log10(dens4), 2), np.round(np.log10(Z40/0.02), 2), np.round(np.log10(temp2g), 2), np.round(np.log10(dens0g), 2), np.round(np.log10(press0), 2), np.round(np.log10(entropy0), 2), np.round(np.log10(mol00), 2), np.round(np.log10(HII00), 2), np.round(np.log10(OH), 2), np.round(np.log10(Z00/0.02), 2), np.round(np.log10(mass5[0]*1.e10),2), np.round(np.log10(pmass5[0]*1.e10),2), np.round(np.log10(fedd15),2), np.round(np.log10(acc2_15),2), np.round(np.log10(acc3_15),2), np.round(np.log10(power15*1.e20),2), np.round(np.log10(Lmech15*1.e20),2), numj_gal, numj_all,  np.round(np.log10(temp0g), 2))


