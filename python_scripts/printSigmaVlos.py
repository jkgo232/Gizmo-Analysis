import h5py
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from scipy.spatial.transform import Rotation as R
plt.rcParams["mathtext.fontset"] = "cm"
font = {'weight': 'normal','size': 28}

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

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def get_sigma(snapdir, BHdir, rlim, zlim, j):

 if j<100:
  g=h5py.File(str(BHdir)+"snapshot_0"+str(j)+".hdf5", 'r')
  f=h5py.File(str(snapdir)+"snapshot_0"+str(j)+".hdf5", 'r')
 if j>=100:
  g=h5py.File(str(BHdir)+"snapshot_"+str(j)+".hdf5", 'r')
  f=h5py.File(str(snapdir)+"snapshot_"+str(j)+".hdf5", 'r')

 pos5=np.array(g['PartType5']['Coordinates'])
 xc=pos5[:,0]
 yc=pos5[:,1]
 zc=pos5[:,2]

 red=f['Header'].attrs['Redshift']
 ag=f['Header'].attrs['Time']
 idk = (np.abs(red - tz)).argmin()
 time=np.round(tage[idk], 4)

 mass4=np.array(f['PartType4']['Masses'])
 vel4=np.array(f['PartType4']['Velocities'])
 pos4=np.array(f['PartType4']['Coordinates'])
 pot4=np.array(f['PartType4']['Potential'])

 mass0=np.array(f['PartType0']['Masses'])
 vel0=np.array(f['PartType0']['Velocities'])
 pos0=np.array(f['PartType0']['Coordinates'])
 ID0 = np.array(f['PartType0']['ParticleIDs'])

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
 pot0=np.take(pot4, b)

 ind_cp=np.where(pot0==np.min(pot0))
 xc=np.take(x4f, ind_cp[0])
 yc=np.take(y4f, ind_cp[0])
 zc=np.take(z4f, ind_cp[0])

 ind_cpv=np.where(np.sqrt((x4-xc)*(x4-xc)+(y4-yc)*(y4-yc)+(z4-zc)*(z4-zc))<=1.)
 vxc=np.median(np.take(vx4, ind_cpv[0]))
 vyc=np.median(np.take(vy4, ind_cpv[0]))
 vzc=np.median(np.take(vz4, ind_cpv[0]))

 vc=np.column_stack((vxc,vyc,vzc))
 c=np.column_stack((xc,yc,zc))

 #rotate based on angular momentum of particles inside 3kpc and center
 mask3 = np.where((np.abs(x4-xc) < 3) & (np.abs(y4-yc) < 3) & (np.abs(z4-zc) < 3))
 x0=x4[mask3]-xc
 y0=y4[mask3]-yc
 z0=z4[mask3]-zc
 vx0=vx4[mask3]-vxc
 vy0=vy4[mask3]-vyc
 vz0=vz4[mask3]-vzc
 Jx0=(y0*vz0-z0*vy0)
 Jy0=(z0*vx0-x0*vz0)
 Jz0=(x0*vy0-y0*vx0)
 Jx=np.sum(Jx0)
 Jy=np.sum(Jy0)
 Jz=np.sum(Jz0)
 J=np.sqrt(Jx*Jx+Jy*Jy+Jz*Jz)
 theta=np.arccos(Jz/J)
 b=np.sqrt(Jx*Jx+Jy*Jy)
 bx=-Jy/b
 by=Jx/b
 bz=0.
 q0=np.cos(theta/2)
 q1=bx*np.sin(theta/2)
 q2=by*np.sin(theta/2)
 q3=bz*np.sin(theta/2)
 r=R.from_quat([q1, q2, q3, q0])
 r=r.inv()

 #center
 rposc=r.apply(c)
 xc=rposc[:,0][0]
 yc=rposc[:,1][0]
 zc=rposc[:,2][0]

 rvelc=r.apply(vc)
 vxc=rvelc[:,0][0]
 vyc=rvelc[:,1][0]
 vzc=rvelc[:,2][0]

 rvel=r.apply(vel4)
 vx4=(rvel[:,0]-vxc)*np.sqrt(ag)
 vy4=(rvel[:,1]-vyc)*np.sqrt(ag)
 vz4=(rvel[:,2]-vzc)*np.sqrt(ag)

 rpos=r.apply(pos4)
 x4=(rpos[:,0]-xc)*ag/0.678
 y4=(rpos[:,1]-yc)*ag/0.678
 z4=(rpos[:,2]-zc)*ag/0.678

 r4=np.sqrt(x4*x4+y4*y4)

 b = np.where(r4 < rlim)[0]
 vx4=np.take(vx4, b)
 vy4=np.take(vy4, b)
 vz4=np.take(vz4, b)
 x4=np.take(x4, b)
 y4=np.take(y4, b)
 z4=np.take(z4, b)

 b = np.where(np.absolute(z4) < zlim)[0]
 vx4=np.take(vx4, b)
 vy4=np.take(vy4, b)
 vz4=np.take(vz4, b)
 x4=np.take(x4, b)
 y4=np.take(y4, b)
 z4=np.take(z4, b)

 pos=np.column_stack((x4,y4,z4))

 #find sigmaz for each particle
 f=3
 t = AnnoyIndex(f, "euclidean")
 for i in range(len(pos)):
     t.add_item(i, pos[i])

 t.build(1) # 1 trees
 t.save('test.ann')
 u = AnnoyIndex(f, "euclidean")
 u.load('test.ann') # super fast, will just mmap the file
   
 vsigma=np.zeros(len(pos))
 vmag=vz4  #v_z values
 for i in range(len(pos)):
   rindex=u.get_nns_by_item(i, 64)  #find 64 neighbours
   vsigma[i]=np.sqrt(np.mean( (vmag[rindex]-np.mean(vmag[rindex]) )**2.) )  #calculate sigma
   
 b2 = np.where(np.sqrt(x4*x4+y4*y4) != 0)[0]
 vx4=np.take(vx4, b2)
 vy4=np.take(vy4, b2)
 vz4=np.take(vz4, b2)
 x4=np.take(x4, b2)
 y4=np.take(y4, b2)
 z4=np.take(z4, b2)

 list=np.linspace(-rlim,rlim,40)
 sigma=[]
 vlos=[]
 rad=[]
 for i in range(len(list)-1):

  mask=np.where( (x4>list[i]) & (x4<list[i+1]) )
  x40=x4[mask]
  y40=y4[mask]
  z40=z4[mask]
  vx40=vx4[mask]
  vy40=vy4[mask]
  vz40=vz4[mask]
  
  sigsz=np.std(vz40) 
  sigma=np.append(sigma, sigsz)

  #get vlos
  vlos=np.append(vlos, np.nanmean(vy40))

  #get radius
  rad=np.append(rad, (list[i]+(list[i+1]))/2.)

 #also get sigma from NN method
 zlist=np.linspace(-zlim,zlim,40)
 sigmaNN=[]
 for i in range(len(zlist)-1):
   mask=np.where( (z4>zlist[i]) & (z4<zlist[i+1]) )
   sig=vsigma[mask]
   sigmaNN=np.append(sigmaNN, np.nanmean(sig))

 return red, time, rad, vlos, sigma, sigmaNN


def plot_vrad(snapdir, BHdir, rlim, zlim, j, label, color, line):
 red, time, rad, vlos, sigma, sigmaNN = get_sigma(snapdir, BHdir, rlim, zlim, j)
 ax.plot(smooth(rad, num), smooth(vlos/sigma, num), label=str(label), c=str(color), linewidth=3, linestyle=str(line), alpha=1.)
 return red, time, rad

def plot_vtime(snapdir, BHdir, rlim, zlim, label, color, line):
 x=[]
 y=[]
 y2=[]
 snaps=np.arange(208,2000,1)
 for j in snaps:
  red, time, rad, vlos, sigma, sigmaNN = get_sigma(snapdir, BHdir, rlim, zlim, j)
  x=np.append(x, time)
  y=np.append(y, np.nanmax(vlos)/np.nanmean(sigma))
  y2=np.append(y2, np.nanmax(vlos)/np.nanmean(sigmaNN))
  print(label, time, np.nanmax(vlos), np.nanmean(sigmaNN), np.nanmean(sigma), np.nanmax(vlos)/np.nanmean(sigmaNN), np.nanmax(vlos)/np.nanmean(sigma))
 ax.plot(smooth(x, num), smooth(y, num), label=str(label), c=str(color), linewidth=3, linestyle=str(line), alpha=1.)
 ax.plot(smooth(x, num), smooth(y2, num), label=str(label)+' NN', c=str(color), linewidth=3, linestyle=str(line), alpha=0.5)
 return x, y


def print_vtime(snapdir, BHdir, rlim, zlim, j):
 #print('finding values...')
 red, time, rad, vlos, sigma, sigmaNN = get_sigma(snapdir, BHdir, rlim, zlim, j)
 #print(snapdir, red, time, np.nanmax(vlos), np.nanmean(sigma), np.nanmean(sigmaNN))
 #print('done.')
 return red, time, np.nanmax(vlos), np.nanmean(sigma), np.nanmean(sigmaNN)


#set directories 

snapdirnoe='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992noBH/'
snapdirnoe2='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992noBHe/'
snapdir5e='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_45/'
snapdir5e2='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_45e/'
snapdir15e='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_15/'
snapdir15e2='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_15e/'
snapdir15e3='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_15T/'
snapdir50e='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_50/'
snapdir50e2='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_50e/'

snapdir5og='/home/jkgo232/gizmo/gizmo-lates/GIZMOe/out/72992_45og/'

snapdirno='/scratch/jkgo232/72992noBH/'
snapdir5='/scratch/jkgo232/72992_045/'
snapdir15='/scratch/jkgo232/72992_15/'
snapdir152='/scratch/jkgo232/72992_15_2/'
snapdir50='/scratch/jkgo232/72992_50/'

'''
#plot vs radius

fig, ax = plt.subplots(figsize=(10,8))
num=1
snap=1999
red, time, rad = plot_vrad(snapdir5, snapdir5, 23, 5, snap, '$\epsilon_\mathrm{5, LBH}$', 'black', 'solid')
red, time, rad = plot_vrad(snapdir5e, snapdir5, 23, 5, snap, '$\epsilon_\mathrm{5, EBH}$', 'red', 'dashed')
red, time, rad = plot_vrad(snapdir5og, snapdir5, 23, 5, snap, '$\epsilon_\mathrm{5, EBHtest}$', 'blue', 'dotted')
arr=np.zeros(len(rad))+1
ax.plot(rad, arr, c='gray', linestyle='dotted')
ax.tick_params(labelsize=22, length=8, width=2, right=True, direction='in')
ax.set_xlabel('Radius [kpc]', fontdict=font)
ax.set_ylabel('v$_\mathrm{los}$/$\sigma_\mathrm{z}$', fontdict=font)
ax.legend(fontsize=20, ncols=1)
fig.suptitle(r'z='+str(np.round(red,2))+r' t='+str(time), fontsize=22)
#ax.set_xlim(xlim_low, xlim_high)
#ax.set_ylim(ylim_low, ylim_high)
#plt.savefig("./"+str(name)+".png")
plt.show()
plt.close()
'''
'''
#plot vs time
fig, ax = plt.subplots(figsize=(10,8))
num=1
x, y = plot_vtime(snapdir5, snapdir5e, 23, 5, '$\epsilon_\mathrm{5, LBH}$', 'black', 'solid')
x, y = plot_vtime(snapdir5e, snapdir5, 23, 5, '$\epsilon_\mathrm{5, EBH}$', 'red', 'dashed')
x, y = plot_vtime(snapdir5og, snapdir5, 23, 5, '$\epsilon_\mathrm{5, EBHtest}$', 'blue', 'dotted')
arr=np.zeros(len(x))+1
ax.plot(x, arr, c='gray', linestyle='dotted')
#add redshift to top
ages=[0.4719, 1.1689, 3.2802, 4.2750, 5.8585, 8.5991, 10.7578, 13.76]
age=[10, 5, 2, 1.5, 1, 0.5, 0.3, 0]
ax2 = ax.twiny()
ax2.set_xticks(ages)
ax2.tick_params(labelsize=22, which='both', length=8, width=2, right=True, direction='in')
ax2.set_xticklabels(age)
ax2.set_xlabel('Redshift', fontdict=font)
ax2.set_xlim(np.min(x), np.max(x))
ax.set_xlim(np.min(x), np.max(x))
#ax2.set_xlim(np.min(0.47), 13.7)
#ax.set_xlim(np.min(0.47), 13.7)
ax.tick_params(labelsize=22, length=8, width=2, right=True, direction='in')
ax.set_xlabel('Time Since Big Bang [Gyr]', fontdict=font)
ax.set_ylabel('v$_\mathrm{los}$/$\sigma_\mathrm{z}$', fontdict=font)
ax.legend(fontsize=20, ncols=1)
#ax.set_xlim(xlim_low, xlim_high)
#ax.set_ylim(ylim_low, ylim_high)
#plt.savefig("./"+str(name)+".png")
plt.show()
plt.close()
'''
#print vs time

snaps=np.arange(1476,2000,5)
for j in snaps:
 try:
  ind=np.where(snap0==j)
  rvir=rvir0[ind]
  rlim=0.1*rvir
  time, red, vlos, sigma, sigmaNN = print_vtime(snapdirno, snapdir5, rlim, 5, j)
  time, red, vlos5, sigma5, sigmaNN5 = print_vtime(snapdir5, snapdir5, rlim, 5, j)
  time, red, vlos15, sigma15, sigmaNN15 = print_vtime(snapdir152, snapdir152, rlim, 5, j)
  time, red, vlos50, sigma50, sigmaNN50 = print_vtime(snapdir50, snapdir50, rlim, 5, j)
  time, red, vlose, sigmae, sigmaNNe = print_vtime(snapdirnoe2, snapdir5, rlim, 5, j)
  time, red, vlos5e, sigma5e, sigmaNN5e = print_vtime(snapdir5e2, snapdir5e2, rlim, 5, j)
  time, red, vlos15e, sigma15e, sigmaNN15e = print_vtime(snapdir15e3, snapdir15e3, rlim, 5, j)
  time, red, vlos50e, sigma50e, sigmaNN50e = print_vtime(snapdir50e2, snapdir50e2, rlim, 5, j)
  time, red, vlos5og, sigma5og, sigmaNN5og = print_vtime(snapdir5og, snapdir5og, rlim, 5, j)
  print(time, red, vlos, sigma, sigmaNN, vlos5, sigma5, sigmaNN5, vlos15, sigma15, sigmaNN15, vlos50, sigma50, sigmaNN50, vlose, sigmae, sigmaNNe, vlos5e, sigma5e, sigmaNN5e, vlos15e, sigma15e, sigmaNN15e, vlos50e, sigma50e, sigmaNN50e, vlos5og, sigma5og, sigmaNN5og)
 except:
  pass
