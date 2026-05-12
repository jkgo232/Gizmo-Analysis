"""
track_galaxy_properties.py
==========================
Tracks a galaxy identified by HOP group ID backwards in time using
particle IDs, computing a comprehensive set of structural, kinematic,
and chemical properties at each snapshot.

Usage
-----
python track_galaxy_properties.py \
    --hop-dir     /path/to/hop/outputs \
    --snap-dir    /path/to/snapshots \
    --snaptimes   snapshot_times.txt \
    --hop-id      0 \
    --final-snap  600 \
    --output      galaxy_properties.txt \
    [--hubble     0.678] \
    [--omega-m    0.308] \
    [--r-align    3.0]   # kpc, radius for angular momentum alignment

Dependencies
------------
    pip install numpy h5py scipy matplotlib
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import h5py
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore")


# ===========================================================================
# CONSTANTS
# ===========================================================================

SOLAR_Z       = 0.02          # Solar metallicity (mass fraction)
SOLAR_OH      = 8.69            # 12 + log(O/H) solar
O_MASS_FRAC   = 0.4295          # Oxygen mass fraction of metals (Asplund+09)
H_MASS_FRAC   = 0.76            # Hydrogen mass fraction
KPC_TO_CM     = 3.0857e21       # cm per kpc
MSUN_TO_G     = 1.989e33        # g per Msun
KM_TO_CM      = 1e5
MP            = 1.6726e-24      # proton mass, g
KB            = 1.3806e-16      # Boltzmann constant, erg/K
GAMMA         = 5.0 / 3.0


# ===========================================================================
# 1.  SNAPSHOT AND HOP FILE UTILITIES
# ===========================================================================

def load_snap_times(filepath):
    """Load snapshot_times.txt -> {snap_num: {a, z, t_gyr}}"""
    snap_map = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                snap_num = int(float(parts[0]))
                a        = float(parts[1])
                z        = float(parts[2])
                t        = float(parts[3])   # Gyr
                snap_map[snap_num] = {"a": a, "z": z, "t_gyr": t}
            except (ValueError, IndexError):
                continue
    return snap_map


def find_snap_file(snap_dir, snap_num):
    """Find snapshot file by number."""
    snap_dir = Path(snap_dir)
    candidates = list(snap_dir.glob(f"snapshot_{snap_num:03d}.hdf5")) + \
                 list(snap_dir.glob(f"snapshot_{snap_num:04d}.hdf5")) + \
                 list(snap_dir.glob(f"snapshot_{snap_num}.hdf5"))
    if not candidates:
        return None
    return candidates[0]


def find_hop_file(hop_dir, snap_num):
    """Find HOP output file for a given snapshot."""
    hop_dir = Path(hop_dir)
    candidates = list(hop_dir.glob(f"*{snap_num:03d}*.txt")) + \
                 list(hop_dir.glob(f"*{snap_num:04d}*.txt")) 
                 #list(hop_dir.glob(f"hop_{snap_num:03d}.hdf5")) + \
                 #list(hop_dir.glob(f"hop_{snap_num:04d}.hdf5"))
    if not candidates:
        return None
    return candidates[0]


# ===========================================================================
# 2.  READ HOP GROUP AND GET PARTICLE IDs
# ===========================================================================

def read_hop_group(hop_file, hop_id, snap_dir,
                   part_types=("PartType0", "PartType4")):
    """
    Read particle IDs belonging to a HOP group from a text-format HOP file,
    using particle IDs loaded directly from a GIZMO snapshot.

    Parameters
    ----------
    hop_file : str
        Path to HOP output text file.
    hop_id : int
        Desired HOP group number.
    snap_dir : str
        Path to GIZMO snapshot HDF5 file.
    part_types : tuple
        Particle types to include (default: gas + stars).

    Returns
    -------
    dict
        {part_type: set of particle IDs}
    """

    ids_by_type = {}

    # --- Load snapshot particle IDs ---
    snap_ids = {}
    with h5py.File(snap_dir, "r") as f:
        for pt in part_types:
            if pt in f:
                snap_ids[pt] = f[pt]["ParticleIDs"][:]
            else:
                snap_ids[pt] = np.array([], dtype=np.int64)

    snap_ids0 = snap_ids.get("PartType0", np.array([], dtype=np.int64))
    snap_ids4 = snap_ids.get("PartType4", np.array([], dtype=np.int64))

    n_gas = len(snap_ids0)
    n_star = len(snap_ids4)

    # --- Read HOP file once ---
    all_groups = np.loadtxt(hop_file, dtype=int)

    expected_len = n_gas + n_star
    if len(all_groups) < expected_len:
        raise ValueError(
            f"HOP file shorter than expected: {len(all_groups)} vs {expected_len}"
        )

    # Split into gas + star segments
    gas_groups = all_groups[:n_gas]
    star_groups = all_groups[n_gas:n_gas + n_star]

    # --- Find indices ---
    gas_idx = np.where(gas_groups == hop_id)[0]
    star_idx = np.where(star_groups == hop_id)[0]

    # --- Map to particle IDs ---
    if "PartType0" in part_types:
        ids_by_type["PartType0"] = set(
            snap_ids0[gas_idx].astype(np.int64)
        )

    if "PartType4" in part_types:
        ids_by_type["PartType4"] = set(
            snap_ids4[star_idx].astype(np.int64)
        )

    # --- Logging ---
    n_g = len(ids_by_type.get("PartType0", []))
    n_s = len(ids_by_type.get("PartType4", []))
    total = n_g + n_s

    print(
        f"HOP group {hop_id}: "
        f"{n_g:,} gas + {n_s:,} stars = {total:,} particles"
    )

    return ids_by_type

def get_particle_ids_from_snap(snap_file, group_ids, part_type):
    """
    Given a set of group_ids, return the indices (row numbers) in
    the snapshot's part_type array whose ParticleIDs are in group_ids.
    """
    with h5py.File(snap_file, "r") as hf:
        if part_type not in hf:
            return np.array([], dtype=int)
        snap_ids = hf[f"{part_type}/ParticleIDs"][:].astype(np.int64)
    mask = np.isin(snap_ids, list(group_ids))
    return np.where(mask)[0]


# ===========================================================================
# 3.  READ PARTICLE DATA
# ===========================================================================

def read_particles(snap_file, indices, part_type, fields):
    """
    Read specified fields for particles at given indices.
    Returns dict of arrays.
    """
    data = {}
    with h5py.File(snap_file, "r") as hf:
        if part_type not in hf:
            return {f: np.array([]) for f in fields}
        grp = hf[part_type]
        for field in fields:
            if field in grp:
                arr = grp[field][:]
                if arr.ndim == 1:
                    data[field] = arr[indices]
                else:
                    data[field] = arr[indices]
            else:
                data[field] = None
    return data


def load_all_particles(snap_file, star_ids, gas_ids, h, box_kpc):
    """
    Load all relevant particle data for stars and gas.
    Converts units:
      Positions : kpc/h comoving -> kpc physical
      Masses    : 1e10 Msun/h -> Msun
      Velocities: km/s (peculiar, already physical in GIZMO)
    Returns star_data, gas_data dicts.
    """
    with h5py.File(snap_file, "r") as hf:
        a    = float(hf["Header"].attrs["Time"])
        h_   = float(hf["Header"].attrs["HubbleParam"])

    # ---- Stars ----
    star_fields = ["Coordinates", "Masses", "Velocities",
                   "Metallicity", "StellarFormationTime",
                   "ParticleIDs"]
    with h5py.File(snap_file, "r") as hf:
        if "PartType4" in hf and len(star_ids) > 0:
            snap_ids = hf["PartType4/ParticleIDs"][:].astype(np.int64)
            idx      = np.where(np.isin(snap_ids, list(star_ids)))[0]
            sdata    = {}
            for field in star_fields:
                if field in hf["PartType4"]:
                    arr = hf[f"PartType4/{field}"][:]
                    sdata[field] = arr[idx] if arr.ndim == 1 else arr[idx]
        else:
            sdata = {f: np.array([]) for f in star_fields}

    #Z0=np.array(f['PartType0']['Metallicity'][:,0])
    #He0=np.array(f['PartType0']['Metallicity'][:,1])
    #O0=np.array(f['PartType0']['Metallicity'][:,4])
    # ---- Gas ----
    gas_fields = ["Coordinates", "Masses", "Velocities",
                  "Metallicity", "InternalEnergy",
                  "ElectronAbundance", "Density",
                  "StarFormationRate", "ParticleIDs", "MolecularMassFraction"]
    with h5py.File(snap_file, "r") as hf:
        if "PartType0" in hf and len(gas_ids) > 0:
            snap_ids = hf["PartType0/ParticleIDs"][:].astype(np.int64)
            idx      = np.where(np.isin(snap_ids, list(gas_ids)))[0]
            gdata    = {}
            for field in gas_fields:
                if field in hf["PartType0"]:
                    arr = hf[f"PartType0/{field}"][:]
                    gdata[field] = arr[idx] if arr.ndim == 1 else arr[idx]
        else:
            gdata = {f: np.array([]) for f in gas_fields}

   # ---- BH ----
    bh_fields = ['BH_Mass', 'BH_Mdot', 'BH_Specific_AngMom', 'Masses']
    with h5py.File(snap_file, "r") as hf:
        if "PartType5" in hf:
            bdata    = {}
            for field in bh_fields:
                if field in hf["PartType5"]:
                    arr = hf[f"PartType5/{field}"][:]
                    bdata[field] = arr[0] if arr.ndim == 1 else arr[0]
        else:
            bdata = {f: np.array([]) for f in bh_fields}

    # Unit conversions
    def convert_pos(pos):
        if pos is None or len(pos) == 0:
            return pos
        return pos * a / h_    # kpc/h comoving -> kpc physical

    def convert_mass(mass):
        if mass is None or len(mass) == 0:
            return mass
        return mass * 1e10 / h_   # 1e10 Msun/h -> Msun

    def convert_bhmass(mass):
        if mass is None:
            return mass
        return mass * 1e10 / h_   # 1e10 Msun/h -> Msun

    def convert_vel(vel):
        # GIZMO velocities are already km/s peculiar
        # Add Hubble flow correction: v_phys = v_pec + H(a)*r
        return vel * np.sqrt(a)   # internal -> km/s physical

    def convert_lum(mdot):
        acc2_15=mdot*(2.93363e+43/4.55115e+16) #g/s
        Lmech=(0.1*0.5*acc2_15*3*3*0.1*0.1) #erg/s/1e20
        u_internal = (3/2) * ((KB) / MP)  # erg/g; T_spawn = BAL_internal_temperature we set 1.e10K
        Ltherm = (acc2_15 * u_internal)/1.e10 #erg/s
        return Lmech + 0.1*Ltherm   # erg/s

    for d in [sdata, gdata]:
        if "Coordinates" in d and d["Coordinates"] is not None \
                and len(d["Coordinates"]) > 0:
            d["Coordinates"] = convert_pos(d["Coordinates"])
        if "Masses" in d and d["Masses"] is not None \
                and len(d["Masses"]) > 0:
            d["Masses"] = convert_mass(d["Masses"])
        if "Velocities" in d and d["Velocities"] is not None \
                and len(d["Velocities"]) > 0:
            d["Velocities"] = convert_vel(d["Velocities"])

    for d in [bdata]:
        if "BH_Mass" in d and d["BH_Mass"] is not None:
            d["BH_Mass"] = convert_bhmass(d["BH_Mass"])
        if "BH_Mdot" in d and d["BH_Mdot"] is not None:
            d["BH_Mdot"] = convert_lum(d["BH_Mdot"])

    sdata["a"] = a
    gdata["a"] = a
    bdata["a"] = a
    return sdata, gdata, bdata


# ===========================================================================
# 4.  GALAXY CENTRE AND ALIGNMENT
# ===========================================================================

def compute_centre(pos, masses, n_iter=5, frac=0.5):
    """
    Iterative shrinking-sphere centre of mass.
    Starts with all particles, shrinks by frac each iteration.
    """
    centre = np.average(pos, weights=masses, axis=0)
    r_max  = np.max(np.linalg.norm(pos - centre, axis=1))

    for _ in range(n_iter):
        r     = np.linalg.norm(pos - centre, axis=1)
        r_max *= frac
        mask  = r < r_max
        if mask.sum() < 10:
            break
        centre = np.average(pos[mask], weights=masses[mask], axis=0)

    return centre


def compute_angular_momentum(pos, vel, masses, centre, vel_centre, r_max_kpc):
    """
    Compute total angular momentum vector of stars within r_max_kpc.
    L = sum_i m_i * (r_i x v_i)
    """
    r   = pos - centre
    v   = vel - vel_centre
    dist = np.linalg.norm(r, axis=1)
    mask = dist < r_max_kpc
    if mask.sum() < 5:
        return np.array([0., 0., 1.])

    r_sel = r[mask]
    v_sel = v[mask]
    m_sel = masses[mask]

    L = np.sum(m_sel[:, None] * np.cross(r_sel, v_sel), axis=0)
    norm = np.linalg.norm(L)
    return L / norm if norm > 0 else np.array([0., 0., 1.])


def rotation_matrix_to_align(L_hat):
    """
    Returns rotation matrix R such that R @ L_hat = [0, 0, 1].
    Face-on view: L along z-axis.
    Edge-on view: rotate 90 deg around x after face-on alignment.
    """
    z = np.array([0., 0., 1.])
    v = np.cross(L_hat, z)
    s = np.linalg.norm(v)
    c = np.dot(L_hat, z)

    if s < 1e-10:
        # Already aligned
        return np.eye(3) if c > 0 else np.diag([1., 1., -1.])

    Vx = np.array([[0,    -v[2],  v[1]],
                   [v[2],  0,    -v[0]],
                   [-v[1], v[0],  0   ]])

    R = np.eye(3) + Vx + Vx @ Vx * (1 - c) / (s**2)
    return R


def rotate_particles(pos, vel, centre, vel_centre, R):
    """Apply rotation matrix to positions and velocities."""
    pos_rot = (R @ (pos - centre).T).T
    vel_rot = (R @ (vel - vel_centre).T).T
    return pos_rot, vel_rot

def rotate(spos, svel, pos, vel, centre, vel_centre):
      x4=spos[:,0]
      y4=spos[:,1]
      z4=spos[:,2]
      vx4=svel[:,0]
      vy4=svel[:,1]
      vz4=svel[:,2]

      cen_pos = centre 
      cen_vel = vel_centre
      xc=cen_pos[0]
      yc=cen_pos[1]
      zc=cen_pos[2]
      vxc=cen_vel[0]
      vyc=cen_vel[1]
      vzc=cen_vel[2]

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
      rpos=r.apply(centre)
      xc=rpos[0]
      yc=rpos[1]
      zc=rpos[2]
      rpos=r.apply(pos)
      x=rpos[:,0]-xc
      y=rpos[:,1]-yc
      z=rpos[:,2]-zc
      rpos=r.apply(vel_centre)
      vxc=rpos[0]
      vyc=rpos[1]
      vzc=rpos[2]
      rpos=r.apply(vel)
      vx=rpos[:,0]-vxc
      vy=rpos[:,1]-vyc
      vz=rpos[:,2]-vzc

      rot_vel=np.column_stack((vx,vy,vz))
      rot_pos=np.column_stack((x,y,z))

      return rot_pos, rot_vel


# ===========================================================================
# 5.  BASIC MASSES AND SFR
# ===========================================================================

def compute_stellar_mass(star_masses):
    return float(np.sum(star_masses)) if len(star_masses) > 0 else 0.0


def compute_gas_mass(gas_masses):
    return float(np.sum(gas_masses)) if len(gas_masses) > 0 else 0.0


def lookback_time_myr(a):
    """
    Lookback time from a=1 to scale factor a, in Myr.
    """
    from scipy.integrate import quad
    omega_m=0.308 
    hubble=0.678
    H0_myr  = hubble * 100 / 977.8   # H0 in 1/Myr
    omega_l = 1.0 - omega_m

    def integrand(ap):
        return 1.0 / (ap * np.sqrt(omega_m / ap**3 + omega_l))

    result, _ = quad(integrand, a, 1.0)
    return result / H0_myr


def compute_sfr(sdata, dt_myr=30.0):
    """
    SFR using a fixed physical time window of dt_myr Myr.
    """
    tdata=np.loadtxt('./outputs/lookuptable.txt')
    ta=tdata[:,0]
    tz=tdata[:,1]
    tage=tdata[:,2]

    form_a = sdata.get("StellarFormationTime")
    mass = sdata.get("Masses")
    if form_a is None or len(form_a) == 0:
        return 0.0

    a_now = float(sdata.get("a", 1.0))
    z_now = 1.0 / a_now - 1.0

    idk = (np.abs(z_now - tz)).argmin()
    time=tage[idk]
    myr30=time-30.e-3
    idk30 = (np.abs(myr30 - tage)).argmin()
    a30=ta[idk30]
    z30=tz[idk30]

    a_min = a30
    nsind=np.where(form_a>=a30)
    newstars=mass[nsind]
    m_new=np.sum(newstars)
    SFRavg=np.sum(newstars)/(3.e7)

    
    if m_new > 0:
        z_str = f"z={z_now:.3f}"
        print(f"    SFR ({z_str}): {len(newstars)} new stars in "
              f"last {dt_myr:.0f} Myr  "
              f"(a_min={a_min:.5f}, da={a_now-a_min:.5f})")

    return SFRavg


# ===========================================================================
# 6.  METALLICITIES AND OXYGEN ABUNDANCE
# ===========================================================================

def compute_metallicity(data, key="Metallicity", mass_key="Masses"):
    """
    Mass-weighted mean metallicity.
    Metallicity field in GIZMO is total metal mass fraction Z.
    Returns Z/Z_solar.
    """
    Z    = data.get(key)
    mass = data.get(mass_key)
    if Z is None or mass is None or len(Z) == 0:
        return 0.0
    if Z.ndim > 1:
        Z = Z[:, 0]   # total metallicity column
    Z_mw = float(np.average(Z, weights=mass))
    return Z_mw / SOLAR_Z


def compute_log_OH(gdata):
    """
    12 + log(O/H) from gas metallicity.

    Method:
      Z_gas = total metal mass fraction (from Metallicity field)
      Oxygen mass fraction ~ O_MASS_FRAC * Z_gas
      n_O/n_H = (Z_gas * O_MASS_FRAC / 16) / (H_MASS_FRAC / 1)
              in number ratio (atomic masses: O=16, H=1)
    Returns mass-weighted 12+log(O/H).
    """
    Z    = gdata.get("Metallicity")
    mass = gdata.get("Masses")
    if Z is None or mass is None or len(Z) == 0:
        return 0.0
    Z0 = Z[:, 0]
    He0 = Z[:, 1]
    O0 = Z[:, 4]

    # Avoid log of zero
    #Z0    = np.maximum(Z0, 1e-20)
    H=(np.sum(mass) - np.sum(Z0*mass)-np.sum(He0*mass))*1.98847/1.6735 #number H atoms in galaxy/1e57
    O=np.sum(O0*mass)*1.98847/26.561 #number 0 atoms in galaxy/1e57
    OH=12+np.log10(O/H)
    #nO_nH = (Z * O_MASS_FRAC / 16.0) / (H_MASS_FRAC / 1.0)
    #log_OH = 12.0 + np.log10(nO_nH)
    return float(np.average(OH))


# ===========================================================================
# 7.  HALF-MASS RADIUS
# ===========================================================================

def compute_half_mass_radius(pos_rot, masses):
    """
    3D half-mass radius: radius enclosing 50% of total mass,
    computed from sorted cumulative mass profile.
    """
    if len(pos_rot) == 0:
        return 0.0
    r    = np.linalg.norm(pos_rot, axis=1)
    idx  = np.argsort(r)
    m_cum = np.cumsum(masses[idx])
    m_half = m_cum[-1] / 2.0
    i_half = np.searchsorted(m_cum, m_half)
    return float(r[idx[min(i_half, len(r)-1)]])


# ===========================================================================
# 8.  SURFACE DENSITIES
# ===========================================================================

def compute_surface_density(pos_rot, masses, r_half, n_annuli=10):
    """
    Projected (face-on) surface density within the half-mass radius.
    Uses x-y plane of the rotated frame (face-on view).

    Sigma = M_enclosed / (pi * r_half^2)   [Msun / kpc^2]
    """
    if len(pos_rot) == 0 or r_half <= 0:
        return 0.0
    R    = np.sqrt(pos_rot[:, 0]**2 + pos_rot[:, 1]**2)
    mask = R < r_half
    M_enc = np.sum(masses[mask])
    area  = np.pi * r_half**2
    return float(M_enc / area) if area > 0 else 0.0


# ===========================================================================
# 9.  TEMPERATURE AND DENSITY
# ===========================================================================

def compute_temperature(gdata):
    """
    Gas temperature from internal energy and electron abundance.

    T = (gamma-1) * u * mu * mp / kb

    where mu = mean molecular weight:
      mu = 4 / (3*X + 1 + 4*X*xe)
      X  = hydrogen mass fraction = 0.76
      xe = electron abundance (electrons per hydrogen atom)
    """
    u  = gdata.get("InternalEnergy")
    xe = gdata.get("ElectronAbundance")
    if u is None or xe is None or len(u) == 0:
        return np.array([])

    mu = 4.0 / (3 * H_MASS_FRAC + 1 + 4 * H_MASS_FRAC * xe)
    T  = (GAMMA - 1) * u * KM_TO_CM**2 * mu * MP / KB
    return T


def compute_median_temperature(gdata):
    T = compute_temperature(gdata)
    return float(np.median(T)) if len(T) > 0 else 0.0


def compute_median_density(gdata, h):
    """
    Median gas density in physical units (Msun/kpc^3).
    GIZMO density is in code units: 1e10 Msun/h / (kpc/h)^3
    Convert: rho [Msun/kpc^3] = rho_code * 1e10 * h^2 / a^3
    """
    rho  = gdata.get("Density")
    a    = gdata.get("a", 1.0)
    if rho is None or len(rho) == 0:
        return 0.0
    rho_phys = rho * 1e10 * h**2 / a**3   # Msun/kpc^3
    return float(np.median(rho_phys))


# ===========================================================================
# 10. VELOCITY DISPERSIONS
# ===========================================================================

def compute_los_velocity_dispersion(vel, pos, centre):
    """
    Line-of-sight velocity (v_los).
    In the box frame, LOS = x-axis for random direction.
    v_los = max(v_x)
    """
    r=np.sqrt((pos[:,0]-centre[0])*(pos[:,0]-centre[0])+(pos[:,1]-centre[1])*(pos[:,1]-centre[1])+(pos[:,2]-centre[2])*(pos[:,2]-centre[2]))
    x=pos[:,0]
    rlim=np.max(r)
    #print(np.max(r), np.min(r), np.median(r))
    list=np.logspace(np.log10(np.min(r)),np.log10(rlim),40)
    vlos=[]

    #get mean vlos in shell
    for i in range(len(list)-1):
       mask=np.where( (r >list[i]) & (r < list[i+1]) )
       vel0=vel[:,0][mask]
       vlos=np.append(vlos, np.nanmean(np.abs(vel0)))

    return float(np.nanmax(vlos))


def compute_gas_velocity_dispersions(vel_rot):
    """
    Gas velocity dispersion in each Cartesian coordinate.
    Returns (sigma_x, sigma_y, sigma_z) in km/s.
    """
    if len(vel_rot) == 0:
        return 0.0, 0.0, 0.0
    return (float(np.std(vel_rot[:, 0])),
            float(np.std(vel_rot[:, 1])),
            float(np.std(vel_rot[:, 2])))


# ===========================================================================
# 11. DISK / BULGE DECOMPOSITION
# ===========================================================================

def decompose_disk_bulge(pos_rot, vel_rot, masses, r_half):
    """
    Simple kinematic disk/bulge decomposition based on circularity
    parameter epsilon = Jz / Jz_max(E).

    Method (Abadi+03 / Scannapieco+09):
      1. Compute specific angular momentum Jz = (r x v)_z for each star
      2. Compute specific energy E = 0.5*v^2 (no potential available,
         use kinetic energy as proxy)
      3. For each star find Jz_max at same energy (maximum Jz on
         circular orbit) — approximated by binning in energy
      4. epsilon = Jz / Jz_max
      5. Disk stars: epsilon > 0.7
         Bulge stars: epsilon < 0.7 (or counter-rotating)

    Returns disk_mass, bulge_mass, disk_r_half, bulge_r_half
    """
    if len(pos_rot) < 20:
        return 0.0, 0.0, 0.0, 0.0

    r  = pos_rot
    v  = vel_rot
    Jz = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]   # z-component of r x v
    E  = 0.5 * np.sum(v**2, axis=1)                # specific kinetic energy

    # Bin in energy and find Jz_max in each bin
    n_bins   = 50
    E_bins   = np.percentile(E, np.linspace(0, 100, n_bins+1))
    Jz_max   = np.zeros(len(E))
    for i in range(n_bins):
        mask = (E >= E_bins[i]) & (E < E_bins[i+1])
        if mask.sum() > 0:
            Jz_max[mask] = np.max(np.abs(Jz[mask]))

    # Avoid division by zero
    Jz_max   = np.where(Jz_max > 0, Jz_max, 1e-10)
    epsilon  = Jz / Jz_max

    disk_mask  = epsilon > 0.7
    bulge_mask = ~disk_mask

    disk_mass  = float(np.sum(masses[disk_mask]))
    bulge_mass = float(np.sum(masses[bulge_mask]))

    disk_r  = compute_half_mass_radius(pos_rot[disk_mask],
                                       masses[disk_mask]) \
              if disk_mask.sum() > 5 else 0.0
    bulge_r = compute_half_mass_radius(pos_rot[bulge_mask],
                                       masses[bulge_mask]) \
              if bulge_mask.sum() > 5 else 0.0

    return disk_mass, bulge_mass, disk_r, bulge_r


# ===========================================================================
# 12. BAR DETECTION
# ===========================================================================

def compute_bar_properties(pos_rot, masses, r_max=10.0, n_bins=20):
    """
    Bar radius and ellipticity from moment-of-inertia tensor of the
    projected (face-on) stellar distribution.

    Method:
      1. Compute inertia tensor I in the x-y plane within r_max
      2. Eigenvalues give principal axes a >= b
      3. Ellipticity e = 1 - b/a
      4. Bar radius estimated as the radius where ellipticity profile
         peaks before declining (bar end criterion)

    Returns bar_radius (kpc), bar_ellipticity
    """
    if len(pos_rot) < 20:
        return 0.0, 0.0

    R    = np.sqrt(pos_rot[:, 0]**2 + pos_rot[:, 1]**2)
    mask = R < r_max
    if mask.sum() < 10:
        return 0.0, 0.0

    x = pos_rot[mask, 0]
    y = pos_rot[mask, 1]
    m = masses[mask]

    r_bins  = np.linspace(0, r_max, n_bins+1)
    ellip   = []
    r_mids  = []

    for i in range(n_bins):
        bm = (R[mask] >= r_bins[i]) & (R[mask] < r_bins[i+1])
        if bm.sum() < 5:
            ellip.append(0.0)
            r_mids.append((r_bins[i] + r_bins[i+1]) / 2)
            continue

        xb = x[bm]
        yb = y[bm]
        mb_ = m[bm]

        # Inertia tensor
        Ixx = np.sum(mb_ * xb**2)
        Iyy = np.sum(mb_ * yb**2)
        Ixy = np.sum(mb_ * xb * yb)

        I   = np.array([[Ixx, Ixy], [Ixy, Iyy]])
        evals = np.linalg.eigvalsh(I)
        evals = np.sort(np.abs(evals))[::-1]   # descending

        a_ax = np.sqrt(evals[0]) if evals[0] > 0 else 1e-10
        b_ax = np.sqrt(evals[1]) if evals[1] > 0 else 1e-10
        e    = 1.0 - b_ax / a_ax
        ellip.append(e)
        r_mids.append((r_bins[i] + r_bins[i+1]) / 2)

    ellip  = np.array(ellip)
    r_mids = np.array(r_mids)

    # Bar radius: last radius where ellipticity > 0.25
    bar_mask = ellip > 0.25
    if bar_mask.sum() == 0:
        return 0.0, float(np.max(ellip))

    bar_r   = float(r_mids[bar_mask][-1])
    bar_e   = float(np.max(ellip))
    return bar_r, bar_e

# ===========================================================================
# 13. FIGURE
# ===========================================================================


def plot_galaxy_maps(snap_num, snap_info, sdata, gdata,
                     centre, vel_cen, RR, output_dir,
                     frame_kpc=50.0, npix=500, smooth_sigma=0.2):
    """
    Plot face-on and edge-on projections of stellar and gas components
    in a 2x2 figure grid and save to output_dir.

    Layout:
      [0,0] Stars  face-on    [0,1] Gas    face-on
      [1,0] Stars  edge-on    [1,1] Gas    edge-on

    Parameters
    ----------
    snap_num     : int    snapshot number (for filename)
    snap_info    : dict   {a, z, t_gyr}
    sdata        : dict   stellar particle data (Coordinates, Masses, etc.)
    gdata        : dict   gas particle data
    centre       : (3,)   galaxy centre in physical kpc
    vel_cen      : (3,)   bulk velocity in km/s
    RR            : (3,3)  face-on rotation matrix (L aligned to z)
    output_dir   : str or Path
    frame_kpc    : float  full frame width in physical kpc (default 30)
    npix         : int    image pixels per side (default 400)
    smooth_sigma : float  Gaussian smoothing in pixels (default 1.5)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec
    from scipy.ndimage import gaussian_filter
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    half  = frame_kpc / 2.0
    z     = snap_info["z"]
    t_gyr = snap_info["t_gyr"]


    # ---- Helper: make 2D mass-weighted image ----
    def make_image(x, y, weights, half, npix, sigma):
        if len(x) == 0:
            return np.zeros((npix, npix))
        img, _, _ = np.histogram2d(
            x, y,
            bins   = npix,
            range  = [[-half, half], [-half, half]],
            weights = weights,
        )
        if sigma > 0:
            img = gaussian_filter(img, sigma=sigma)
        return img.T   # transpose: x=horizontal, y=vertical

    # ---- Helper: gas temperature colourmap (2D median T per pixel) ----
    def make_temp_image(x, y, temps, half, npix, sigma):
        """Mass-weighted mean temperature per pixel."""
        if len(x) == 0:
            return np.zeros((npix, npix))
        # numerator: T * mass weighted
        T_img, _, _ = np.histogram2d(
            x, y,
            bins    = npix,
            range   = [[-half, half], [-half, half]],
            weights = temps,
        )
        cnt_img, _, _ = np.histogram2d(
            x, y,
            bins  = npix,
            range = [[-half, half], [-half, half]],
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_T = np.where(cnt_img > 0, T_img / cnt_img, 0)
        if sigma > 0:
            mean_T = gaussian_filter(mean_T, sigma=sigma)
        return mean_T.T

    # ---- Extract positions and masses ----
    s_pos  = sdata.get("Coordinates")
    s_mass = sdata.get("Masses")
    g_pos  = gdata.get("Coordinates")
    g_mass = gdata.get("Masses")
    s_vel  = sdata.get("Velocities")
    g_vel  = gdata.get("Velocities")
    s_sft  = sdata.get("StellarFormationTime")

    has_stars = s_pos is not None and len(s_pos) > 0
    has_gas   = g_pos is not None and len(g_pos) > 0

    # Gas temperature for colourmap
    T_gas = compute_temperature(gdata) if has_gas else np.array([])

    # ---- Build images ----
    # Stars face-on (x-y plane)
    if has_stars:
        #sx_fo, sy_fo = get_proj(s_pos, R,      (0, 1))
        #sx_eo, sy_eo = get_proj(s_pos, R_edge, (0, 2))
        #sx, sy, sz = rotate(s_pos, s_vel, s_pos)
        s_pos_rot, s_vel_rot = rotate(s_pos, s_vel, s_pos, s_vel, centre, vel_cen)
        img_s_fo = make_image(s_pos_rot[:,0], s_pos_rot[:,1], s_mass, half, npix, smooth_sigma)
        img_s_eo = make_image(s_pos_rot[:,0], s_pos_rot[:,2], s_mass, half, npix, smooth_sigma)
    else:
        img_s_fo = img_s_eo = np.zeros((npix, npix))

    # Gas face-on and edge-on
    if has_gas:
        #gx_fo, gy_fo = get_proj(g_pos, R,      (0, 1))
        #gx_eo, gy_eo = get_proj(g_pos, R_edge, (0, 2))
        #gx, gy, gz = rotate(s_pos, s_vel, g_pos)
        g_pos_rot, g_vel_rot = rotate(s_pos, s_vel, g_pos, g_vel, centre, vel_cen)
        img_g_fo = make_image(g_pos_rot[:,0], g_pos_rot[:,1], g_mass, half, npix, smooth_sigma)
        img_g_eo = make_image(g_pos_rot[:,0], g_pos_rot[:,2], g_mass, half, npix, smooth_sigma)

        # Temperature overlay: use temp-weighted image for gas colour
        if len(T_gas) == len(g_mass):
            img_T_fo = make_temp_image(g_pos_rot[:,0], g_pos_rot[:,1], T_gas,
                                       half, npix, smooth_sigma)
            img_T_eo = make_temp_image(g_pos_rot[:,0], g_pos_rot[:,2], T_gas,
                                       half, npix, smooth_sigma)
        else:
            img_T_fo = img_T_eo = None
    else:
        img_g_fo = img_g_eo = np.zeros((npix, npix))
        img_T_fo = img_T_eo = None

    # ---- Figure setup ----
    fig = plt.figure(figsize=(11, 10), facecolor="black")
    gs  = GridSpec(2, 2, figure=fig,
                   left=0.07, right=0.93,
                   top=0.88, bottom=0.05,
                   wspace=0.04, hspace=0.04)

    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])]

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=8,
                       direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

    extent = [-half, half, -half, half]

    # ---- Panel helper ----
    def render_panel(ax, img, cmap, label_text, xlabel, ylabel,
                     vmin=None, vmax=None, norm=None):
        if img.max() > 0:
            if norm is None:
                good = img[img > 0]
                vmin = vmin or (good.min() if len(good) else 1e-6)
                vmax = vmax or img.max()
                norm = LogNorm(vmin=vmin, vmax=vmax)
            ax.imshow(img, norm=norm, cmap=cmap, origin="lower",
                      extent=extent, interpolation="bicubic")
        else:
            ax.imshow(np.zeros((npix, npix)), cmap=cmap,
                      origin="lower", extent=extent)

        # Component / orientation label
        ax.text(0.03, 0.97, label_text,
                transform=ax.transAxes, color="white",
                fontsize=9, va="top", ha="left",
                bbox=dict(facecolor="black", alpha=0.4,
                          edgecolor="none", pad=2))

        ax.set_xlabel(xlabel, color="white", fontsize=9)
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)

    # ---- Render each panel ----
    # [0,0] Stars face-on
    D_flat  = img_s_fo
    D_min   = 1e5 #D_flat.min()
    D_max   = 5e8 #D_flat.max()
    norm_D  = LogNorm(vmin=D_min, vmax=D_max)
    render_panel(axes[0], img_s_fo, "afmhot",
                 "Stars  |  face-on",
                 "", "y  [kpc]", norm=norm_D)

    # [0,1] Gas face-on
    if img_T_fo is not None and img_T_fo.max() > 0:
        # Colour gas by temperature: viridis mapped to log T
        T_flat  = img_T_fo[img_T_fo > 0]
        T_min   = 1e3 #max(T_flat.min(), 1e2)
        T_max   = 1e5 #T_flat.max()
        norm_T  = LogNorm(vmin=T_min, vmax=T_max)
        render_panel(axes[1], img_T_fo, "viridis",
                     "Gas (T-weighted)  |  face-on",
                     "", "", norm=norm_T)
    else:
        render_panel(axes[1], img_g_fo, "viridis",
                     "Gas  |  face-on", "", "")

    # [1,0] Stars edge-on
    D_flat  = img_s_eo
    D_min   = 1e5 #D_flat.min()
    D_max   = 5e8 #D_flat.max()
    norm_D  = LogNorm(vmin=D_min, vmax=D_max)
    render_panel(axes[2], img_s_eo, "afmhot",
                 "Stars  |  edge-on",
                 "x  [kpc]", "z  [kpc]", norm=norm_D)

    # [1,1] Gas edge-on
    if img_T_eo is not None and img_T_eo.max() > 0:
        T_flat  = img_T_eo[img_T_eo > 0]
        T_min   = 1e3 #T_flat.min()
        T_max   = 1e5 #T_flat.max()
        norm_T  = LogNorm(vmin=T_min, vmax=T_max)
        render_panel(axes[3], img_T_eo, "viridis",
                     "Gas (T-weighted)  |  edge-on",
                     "x  [kpc]", "", norm=norm_T)
    else:
        render_panel(axes[3], img_g_eo, "viridis",
                     "Gas  |  edge-on", "x  [kpc]", "")

    # Remove inner tick labels to avoid clutter
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[3].set_ylabel("")

    # ---- Scale bar: 5 kpc on bottom-left panels ----
    bar_kpc  = 5.0
    bar_x0   = -half * 0.82
    bar_y0   = -half * 0.88
    for ax in [axes[2], axes[3]]:
        ax.plot([bar_x0, bar_x0 + bar_kpc], [bar_y0, bar_y0],
                color="white", lw=2, solid_capstyle="butt")
        ax.text(bar_x0 + bar_kpc / 2, bar_y0 + half * 0.05,
                f"{bar_kpc:.0f} kpc",
                color="white", fontsize=7,
                ha="center", va="bottom")

    # ---- Colourbar for gas temperature ----
    if img_T_fo is not None and img_T_fo.max() > 0:
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=LogNorm(vmin=max(img_T_fo[img_T_fo > 0].min(), 1e2),
                         vmax=img_T_fo.max()))
        sm.set_array([])
        cax  = fig.add_axes([0.94, 0.05, 0.015, 0.38])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Gas temperature  [K]", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # ---- Colourbar for stellar surface density ----
    if img_s_fo.max() > 0:
        good  = img_s_fo[img_s_fo > 0]
        sm_s  = plt.cm.ScalarMappable(
            cmap="afmhot",
            norm=LogNorm(vmin=good.min(), vmax=img_s_fo.max()))
        sm_s.set_array([])
        cax_s = fig.add_axes([0.94, 0.50, 0.015, 0.38])
        cbar_s = fig.colorbar(sm_s, cax=cax_s)
        cbar_s.set_label(r"$\Sigma_\star$  [M$_\odot$ pix$^{-1}$]",
                         color="white", fontsize=8)
        cbar_s.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar_s.ax.yaxis.get_ticklabels(), color="white")

    # ---- Title with redshift and time ----
    fig.text(0.50, 0.93,
             f"z = {z:.3f}      t = {t_gyr:.3f} Gyr      "
             f"snap = {snap_num:04d}",
             color="white", fontsize=12, ha="center", va="bottom",
             fontweight="bold")

    fig.text(0.50, 0.905,
             f"Frame: {frame_kpc:.0f} × {frame_kpc:.0f} kpc  "
             f"(physical)  |  "
             f"L aligned face-on",
             color="#aaaaaa", fontsize=8, ha="center", va="bottom")

    # ---- Save ----
    out_path = output_dir / f"galaxy_map_{snap_num:04d}_z{z:.3f}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")
    return out_path


# ===========================================================================
# 14. PARTICLE ID TRACKING BETWEEN SNAPSHOTS
# ===========================================================================

def track_ids_to_previous_snap(current_star_ids, current_gas_ids,
                                hop_file_prev, snap_file_prev,
                                hop_id_prev=None):
    """
    At the previous snapshot, identify which particles belong to our
    galaxy by matching particle IDs.

    Strategy:
      - Use the intersection of current particle IDs with those in
        each HOP group at the previous snapshot
      - Assign to the HOP group with the highest ID overlap
      - If no HOP file available, use IDs directly from snapshot

    Returns star_ids, gas_ids at previous snapshot,
            and the matched HOP group ID (or None)
    """
    current_all_ids = current_star_ids | current_gas_ids

    if hop_file_prev is None:
        # No HOP file: just use particle IDs directly
        return current_star_ids, current_gas_ids, None

    # Find which HOP group at this snapshot has the most overlap
    best_hop_id  = 0
    best_overlap = 0

    # --- Load all HOP labels once (TEXT FORMAT ASSUMED) ---
    # IMPORTANT: assumes same structure you described (gas then stars)
    with h5py.File(snap_file_prev, "r") as hf:
        snap_ids0 = hf["PartType0/ParticleIDs"][:].astype(np.int64) if "PartType0" in hf else np.array([])
        snap_ids4 = hf["PartType4/ParticleIDs"][:].astype(np.int64) if "PartType4" in hf else np.array([])

    n_gas  = len(snap_ids0)
    n_star = len(snap_ids4)

    hop_labels = np.loadtxt(hop_file_prev, dtype=int)
    gas_labels  = hop_labels[:n_gas]
    star_labels = hop_labels[n_gas:n_gas + n_star]

    unique_groups = np.unique(hop_labels)

    # --- Find best matching HOP group ---
    for gid in unique_groups[:50]:   # optional truncation for speed

        gas_ids  = set(snap_ids0[gas_labels == gid])
        star_ids = set(snap_ids4[star_labels == gid])

        group_all_ids = gas_ids | star_ids

        overlap = len(current_all_ids & group_all_ids)

        if overlap > best_overlap:
            best_overlap = overlap
            best_hop_id  = gid
            best_star_ids = star_ids
            best_gas_ids  = gas_ids

    if best_hop_id is None or best_overlap == 0:
        return current_star_ids, current_gas_ids, None

    print(f"  Matched HOP group {best_hop_id} "
          f"(overlap={best_overlap:,})")

    return best_star_ids, best_gas_ids, best_hop_id



# ===========================================================================
# 14. MAIN MEASUREMENT FUNCTION
# ===========================================================================

def measure_snapshot(snap_file, star_ids, gas_ids, snap_info,
                     hubble, r_align_kpc=3.0, snap_num=None, output_dir=None, frame_kpc=30.0):
    """
    Compute all galaxy properties at one snapshot.

    Parameters
    ----------
    snap_file   : path to GIZMO HDF5 snapshot
    star_ids    : set of stellar particle IDs belonging to this galaxy
    gas_ids     : set of gas particle IDs belonging to this galaxy
    snap_info   : dict with a, z, t_gyr
    hubble      : little h
    r_align_kpc : radius in kpc for angular momentum alignment

    Returns dict of all measured properties.
    """
    a   = snap_info["a"]
    z   = snap_info["z"]
    t   = snap_info["t_gyr"]

    props = {
        "z": z, "t_gyr": t,
        "stellar_mass": 0.0, "gas_mass": 0.0, "sfr": 0.0,
        "Z_gas": 0.0, "Z_star": 0.0, "log_OH": 0.0,
        "r_half_star": 0.0, "r_half_gas": 0.0,
        "sigma_star_los": 0.0,
        "sigma_star_x": 0.0, "sigma_star_y": 0.0, "sigma_star_z": 0.0,
        "sigma_gas_x": 0.0, "sigma_gas_y": 0.0, "sigma_gas_z": 0.0,
        "Sigma_star": 0.0, "Sigma_gas": 0.0,
        "T_gas_median": 0.0, "rho_gas_median": 0.0,
        "disk_mass": 0.0, "bulge_mass": 0.0,
        "disk_r_half": 0.0, "bulge_r_half": 0.0,
        "bar_radius": 0.0, "bar_ellipticity": 0.0,
        "n_star": 0, "n_gas": 0,
        "BH_mass": 0.0, "BH_lum": 0.0,
    }

    if len(star_ids) == 0 and len(gas_ids) == 0:
        return props

    # Load particle data
    sdata, gdata, bdata = load_all_particles(snap_file, star_ids, gas_ids,
                                      hubble, box_kpc=None)

    s_pos  = sdata.get("Coordinates")
    s_vel  = sdata.get("Velocities")
    s_mass = sdata.get("Masses")
    g_pos  = gdata.get("Coordinates")
    g_vel  = gdata.get("Velocities")
    g_mass = gdata.get("Masses")
    bh_mass = bdata.get("BH_Mass")
    bh_lum  = bdata.get("BH_Mdot")


    has_stars = s_pos is not None and len(s_pos) > 0
    has_gas   = g_pos is not None and len(g_pos) > 0

    props["n_star"] = len(s_pos) if has_stars else 0
    props["n_gas"]  = len(g_pos) if has_gas   else 0

    # Basic masses
    props["stellar_mass"] = compute_stellar_mass(s_mass) if has_stars else 0.0
    props["gas_mass"]     = compute_gas_mass(g_mass)     if has_gas   else 0.0
    props["sfr"]          = compute_sfr(sdata)           if has_gas   else 0.0

    # Metallicities
    props["Z_star"]  = compute_metallicity(sdata) if has_stars else 0.0
    props["Z_gas"]   = compute_metallicity(gdata) if has_gas   else 0.0
    props["log_OH"]  = compute_log_OH(gdata)      if has_gas   else 0.0

    # Temperature and density
    props["T_gas_median"]   = compute_median_temperature(gdata) \
                              if has_gas else 0.0
    props["rho_gas_median"] = compute_median_density(gdata, hubble) \
                              if has_gas else 0.0

    if not has_stars:
        return props

    # Centre of mass (use stars + gas if available)
    all_pos  = np.vstack([s_pos] + ([g_pos] if has_gas else []))
    all_mass = np.concatenate([s_mass] + ([g_mass] if has_gas else []))
    centre   = compute_centre(all_pos, all_mass)

    all_vel  = np.vstack([s_vel] + ([g_vel] if has_gas else []))
    vel_cen  = np.average(all_vel, weights=all_mass, axis=0)

    # Angular momentum for alignment (stars within r_align_kpc)
    L_hat = compute_angular_momentum(
        s_pos, s_vel, s_mass, centre, vel_cen, r_align_kpc)

    # Rotation matrix: aligns L to z-axis (face-on)
    R = rotation_matrix_to_align(L_hat)

    # Rotate all particles into face-on frame
    #s_pos_rot, s_vel_rot = rotate_particles(s_pos, s_vel, centre, vel_cen, R)
    s_pos_rot, s_vel_rot = rotate(s_pos, s_vel, s_pos, s_vel, centre, vel_cen)

    if has_gas:
        g_pos_rot, g_vel_rot = rotate(s_pos, s_vel, g_pos, g_vel, centre, vel_cen)

    # Half-mass radii (3D, in rotated frame)
    props["r_half_star"] = compute_half_mass_radius(s_pos_rot, s_mass)
    if has_gas:
        props["r_half_gas"] = compute_half_mass_radius(g_pos_rot, g_mass)

    # Surface densities (projected, face-on)
    props["Sigma_star"] = compute_surface_density(
        s_pos_rot, s_mass, props["r_half_star"])
    if has_gas:
        props["Sigma_gas"] = compute_surface_density(
            g_pos_rot, g_mass, props["r_half_gas"])

    # Velocity dispersions - compute_los_velocity_dispersion(vel, pos, centre)
    props["sigma_star_los"] = compute_los_velocity_dispersion(s_vel, s_pos, centre)
    if has_gas:
        sx, sy, sz = compute_gas_velocity_dispersions(g_vel_rot)
        props["sigma_gas_x"] = sx
        props["sigma_gas_y"] = sy
        props["sigma_gas_z"] = sz
    stx, sty, stz = compute_gas_velocity_dispersions(s_vel_rot)
    props["sigma_star_x"] = stx
    props["sigma_star_y"] = sty
    props["sigma_star_z"] = stz


    # Disk / bulge decomposition
    dm, bm, dr, br = decompose_disk_bulge(
        s_pos_rot, s_vel_rot, s_mass, props["r_half_star"])
    props["disk_mass"]    = dm
    props["bulge_mass"]   = bm
    props["disk_r_half"]  = dr
    props["bulge_r_half"] = br

    # Bar properties
    bar_r, bar_e = compute_bar_properties(s_pos_rot, s_mass)
    props["bar_radius"]      = bar_r
    props["bar_ellipticity"] = bar_e

    # BH properties
    props["BH_mass"] = bh_mass
    props["BH_lum"]  = bh_lum

    #plot galaxy
    if output_dir is not None:
       plot_galaxy_maps(snap_num=snap_num, snap_info = snap_info, sdata = sdata, gdata = gdata, centre = centre, vel_cen = vel_cen, RR = R, output_dir = output_dir, frame_kpc  = frame_kpc)

    return props


# ===========================================================================
# 15. WRITE OUTPUT
# ===========================================================================


def write_header(output_path, hop_id):
    """Write the header and column names to the output file."""
    col_order = [
        "snap_num", "z", "t_gyr",
        "stellar_mass", "gas_mass", "sfr",
        "Z_star", "Z_gas", "log_OH",
        "Sigma_star", "Sigma_gas",
        "r_half_star", "r_half_gas",
        "T_gas_median", "rho_gas_median",
        "disk_mass", "bulge_mass", "disk_r_half", "bulge_r_half",
        "bar_radius", "bar_ellipticity",
        "sigma_star_los",
        "sigma_star_x", "sigma_star_y", "sigma_star_z",
        "sigma_gas_x", "sigma_gas_y", "sigma_gas_z",
        "n_star", "n_gas", "BH_mass", "BH_lum",
    ]

    units = {
        "snap_num"       : "",
        "z"              : "",
        "t_gyr"          : "Gyr",
        "stellar_mass"   : "Msun",
        "gas_mass"       : "Msun",
        "sfr"            : "Msun/yr",
        "Z_star"         : "Z/Zsun",
        "Z_gas"          : "Z/Zsun",
        "log_OH"         : "12+log(O/H)",
        "Sigma_star"     : "Msun/kpc^2",
        "Sigma_gas"      : "Msun/kpc^2",
        "r_half_star"    : "kpc",
        "r_half_gas"     : "kpc",
        "T_gas_median"   : "K",
        "rho_gas_median" : "Msun/kpc^3",
        "disk_mass"      : "Msun",
        "bulge_mass"     : "Msun",
        "disk_r_half"    : "kpc",
        "bulge_r_half"   : "kpc",
        "bar_radius"     : "kpc",
        "bar_ellipticity": "",
        "sigma_star_los" : "km/s",
        "sigma_gas_x"    : "km/s",
        "sigma_gas_y"    : "km/s",
        "sigma_gas_z"    : "km/s",
        "n_star"         : "",
        "n_gas"          : "",
        "BH_mass"       : "Msun",
        "BH_lum"       : "erg/s",
    }

    with open(output_path, "w") as f:
        f.write(f"# Galaxy property track\n")
        f.write(f"# Initial HOP ID : {hop_id}\n")
        f.write(f"# Written incrementally — file is valid up to "
                f"the last completed snapshot\n")
        f.write(f"# Angular momentum alignment: stars within r_align kpc\n")
        f.write(f"# Face-on frame: L aligned to z-axis\n")
        f.write(f"#\n")
        f.write("# Units:\n")
        for col in col_order:
            if units.get(col):
                f.write(f"#   {col:25s}: {units[col]}\n")
        f.write(f"#\n")
        f.write("  " + "  ".join(f"{c:>18}" for c in col_order) + "\n")

    return col_order


def write_row(output_path, snap_num, props, col_order):
    """Append a single row to the output file."""
    props["snap_num"] = snap_num
    row = "  " + "  ".join(
        f"{int(props.get(c, 0)):>18d}"
        if c in ("snap_num", "n_star", "n_gas")
        else f"{float(props.get(c, 0.0)):>18.6e}"
        for c in col_order
    )
    with open(output_path, "a") as f:
        f.write(row + "\n")



def write_output(all_props, snap_nums, hop_id, output_path):
    col_order = [
        "snap_num", "z", "t_gyr",
        "stellar_mass", "gas_mass", "sfr",
        "Z_star", "Z_gas", "log_OH",
        "Sigma_star", "Sigma_gas",
        "r_half_star", "r_half_gas",
        "T_gas_median", "rho_gas_median",
        "disk_mass", "bulge_mass", "disk_r_half", "bulge_r_half",
        "bar_radius", "bar_ellipticity",
        "sigma_star_los",
        "sigma_star_x", "sigma_star_y", "sigma_star_z",
        "sigma_gas_x", "sigma_gas_y", "sigma_gas_z",
        "n_star", "n_gas",
    ]

    units = {
        "snap_num"       : "",
        "z"              : "",
        "t_gyr"          : "Gyr",
        "stellar_mass"   : "Msun",
        "gas_mass"       : "Msun",
        "sfr"            : "Msun/yr",
        "Z_star"         : "Z/Zsun",
        "Z_gas"          : "Z/Zsun",
        "log_OH"         : "12+log(O/H)",
        "Sigma_star"     : "Msun/kpc^2",
        "Sigma_gas"      : "Msun/kpc^2",
        "r_half_star"    : "kpc",
        "r_half_gas"     : "kpc",
        "T_gas_median"   : "K",
        "rho_gas_median" : "Msun/kpc^3",
        "disk_mass"      : "Msun",
        "bulge_mass"     : "Msun",
        "disk_r_half"    : "kpc",
        "bulge_r_half"   : "kpc",
        "bar_radius"     : "kpc",
        "bar_ellipticity": "",
        "sigma_star_los" : "km/s",
        "sigma_gas_x"    : "km/s",
        "sigma_gas_y"    : "km/s",
        "sigma_gas_z"    : "km/s",
        "n_star"         : "",
        "n_gas"          : "",
    }

    with open(output_path, "w") as f:
        f.write(f"# Galaxy property track\n")
        f.write(f"# Initial HOP ID: {hop_id}\n")
        f.write(f"# Snapshots tracked: {len(all_props)}\n")
        f.write(f"# Angular momentum alignment: stars within r_align kpc\n")
        f.write(f"# Face-on frame: L aligned to z-axis\n")
        f.write(f"#\n")
        f.write("# Units:\n")
        for col in col_order:
            if units[col]:
                f.write(f"#   {col:25s}: {units[col]}\n")
        f.write(f"#\n")

        # Header row
        f.write("  " + "  ".join(f"{c:>18}" for c in col_order) + "\n")

        for snap_num, props in zip(snap_nums, all_props):
            props["snap_num"] = snap_num
            row = "  " + "  ".join(
                f"{int(props[c]):>18d}"
                if c in ("snap_num", "n_star", "n_gas")
                else f"{props[c]:>18.6e}"
                for c in col_order
            )
            f.write(row + "\n")

    print(f"  Written {len(all_props)} rows to {output_path}")


# ===========================================================================
# 16. MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Track galaxy properties from HOP + GIZMO snapshots.")
    p.add_argument("--hop-dir",    required=True,
                   help="Directory containing HOP output files")
    p.add_argument("--snap-dir",   required=True,
                   help="Directory containing GIZMO snapshot_*.hdf5 files")
    p.add_argument("--snaptimes",  required=True,
                   help="snapshot_times.txt")
    p.add_argument("--hop-id",     type=int, required=True,
                   help="HOP group ID at the final snapshot")
    p.add_argument("--final-snap", type=int, required=True,
                   help="Final snapshot number")
    p.add_argument("--output",     default="galaxy_properties.txt")
    p.add_argument("--hubble",     type=float, default=0.678)
    p.add_argument("--r-align",    type=float, default=3.0,
                   help="Radius in kpc for angular momentum alignment "
                        "(default: 3.0)")
    p.add_argument("--first-snap", type=int, default=None,
                   help="Earliest snapshot to process (default: first "
                        "available)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading snapshot times...")
    snap_map = load_snap_times(args.snaptimes)
    all_snaps = sorted(snap_map.keys())

    # Range of snapshots to process
    first = args.first_snap or all_snaps[0]
    snaps = [s for s in all_snaps if first <= s <= args.final_snap]
    snaps = sorted(snaps, reverse=True)   # process newest -> oldest

    print(f"Processing {len(snaps)} snapshots "
          f"({snaps[-1]} -> {snaps[0]})")

    # --- Initialise particle IDs from final snapshot ---
    final_snap_file = find_snap_file(args.snap_dir, args.final_snap)
    final_hop_file  = find_hop_file(args.hop_dir,   args.final_snap)

    if final_snap_file is None:
        sys.exit(f"ERROR: snapshot {args.final_snap} not found")
    if final_hop_file is None:
        sys.exit(f"ERROR: HOP file for snapshot {args.final_snap} not found")

    print(f"\nReading initial HOP group {args.hop_id} "
          f"from snap {args.final_snap}...")
    group_ids = read_hop_group(final_hop_file, args.hop_id, final_snap_file)

    # Separate into star and gas IDs
    star_ids = set()
    gas_ids  = set()
    with h5py.File(final_snap_file, "r") as hf:
        if "PartType4" in hf:
            sids     = hf["PartType4/ParticleIDs"][:].astype(np.int64)
            star_ids = group_ids["PartType4"] & set(sids)
        if "PartType0" in hf:
            gids    = hf["PartType0/ParticleIDs"][:].astype(np.int64)
            gas_ids = group_ids["PartType0"] & set(gids)

    print(f"  Initial: {len(star_ids):,} star IDs, "
          f"{len(gas_ids):,} gas IDs")

    # --- Write header once at the start ---
    col_order = write_header(args.output, args.hop_id)
    print(f"  Output file initialised: {args.output}")

    # --- Accumulate completed rows for final reverse-order write ---
    completed_rows = []   # (snap_num, props)
    error_snaps    = []


    # --- Main loop: process snapshots from final -> first ---
    #all_props = []
    #snap_nums = []

    for i, snap_num in enumerate(snaps):
        snap_info = snap_map[snap_num]
        snap_file = find_snap_file(args.snap_dir, snap_num)

        if snap_file is None:
            print(f"  [{i+1}/{len(snaps)}] snap {snap_num}: "
                  f"file not found, skipping")
            continue

        print(f"  [{i+1:4d}/{len(snaps)}] "
              f"snap={snap_num:4d}  "
              f"z={snap_info['z']:.4f}  "
              f"stars={len(star_ids):,}  "
              f"gas={len(gas_ids):,}")

        try: 
          props = measure_snapshot(
            snap_file, star_ids, gas_ids, snap_info,
            args.hubble, r_align_kpc=args.r_align, snap_num = snap_num, 
            output_dir = Path(args.output).parent / "galaxy_mapst_2BH_1", 
            frame_kpc=50.0)

          completed_rows.append((snap_num, props))

        except Exception as e:
            print(f"  ERROR at snap {snap_num}: {type(e).__name__}: {e}")
            print(f"  Skipping this snapshot and continuing...")
            error_snaps.append((snap_num, str(e)))
            # Still try to advance the particle tracking below

        # Track particle IDs to previous snapshot
        if i < len(snaps) - 1:
            prev_snap = snaps[i + 1]
            prev_snap_file = find_snap_file(args.snap_dir, prev_snap)
            prev_hop_file  = find_hop_file(args.hop_dir,  prev_snap)

            if prev_snap_file is not None:
                try:
                  star_ids, gas_ids, _ = track_ids_to_previous_snap(
                    star_ids, gas_ids,
                    prev_hop_file, prev_snap_file)

                except Exception as e:
                    print(f"  WARNING: particle tracking failed at "
                          f"snap {snap_num} -> {prev_snap}: "
                          f"{type(e).__name__}: {e}")
                    print(f"  Continuing with current particle IDs...")
                    error_snaps.append((snap_num,
                                        f"tracking: {e}"))

    # --- Write output in chronological order (earliest first) ---
    completed_rows.sort(key=lambda x: x[0])   # sort by snap_num ascending

    # Rewrite file with sorted rows
    col_order = write_header(args.output, args.hop_id)
    for snap_num, props in completed_rows:
        write_row(args.output, snap_num, props, col_order)

    # --- Print error summary ---
    if error_snaps:
        print(f"\n  {'='*60}")
        print(f"  Completed with {len(error_snaps)} errors:")
        for snap_num, msg in error_snaps:
            print(f"    snap {snap_num}: {msg}")
        print(f"  {'='*60}")
    else:
        print(f"\n  Completed with no errors.")

    print(f"  Written {len(completed_rows)} rows to {args.output}")
    print("Done.")



if __name__ == "__main__":
    main()
