# Gizmo-Analysis
Various independent scripts used to analyze GIZMO snapshots from cosmological hydrodynamic zoom-in simulations. 


## Overview

This repository contains Python tools to extract and measure galaxy propertiesas described below for each script included.  These were created for independent personal use and are indented for review only - they require significant edits before they would be appropriate for general application to GIZMO snapshots. 

## Usage

### printGalaxyProps.py


###printSigmaVlos

 - Reads in GIZMO snapshot, Halo data from text file, and lookuptable for conversions between a->z->time

 - Centers using the minimum potential near the SMBH in the central galaxy

 - Rotates frame based on the angular momentum of stars within +/- 3kpc from galaxy center

 - Selects particles within 0.1Rvir and +/- 5kpc in the z-direction

 - Measures $\sigma_z$ for each particle based on the 64 nearest neighbors

 - Measures v_los and sigma_z in 40 linearly space bins between the galaxy center and 0.1R_vir

 - Provide option to 
   1. plot v_los/sigma_z versus radius for a single snapshot
   2. plot v_los/sigma_z versus time
   3. print v_los/sigma_z across a specified snashot range - currently prints the values for nine different models simulataneously to be saved to a text file using 'python printSigmaVlos.py > SigmaVlos.txt' 


###printGalaxyProps

 - Reads in GIZMO snapshot, Halo data from text file, and lookuptable for conversions between a->z->time

 - Centers using the minimum potential near the SMBH in the central galaxy

 - Selects particles within 0.1Rvir

 - Measures and prints for a specified range of redshifts:  
   snapshot, 
   redshift, 
   time, 
   log_stellar_mass, 
   max_stellar_radius, 
   log_gas_mass, 
   max_gas_radius, 
   half mass radius, 
   log_sfr - from gas predicted SFR, 
   log_sfrinr0 - from number of stars formed in the last 30Myr, 
   log_steller_median_density (beta- doesn't work), 
   np.log10(Z40/0.02) - mass weighted average stellar metallicity, 
   log_temp0g - median ags temperature, 
   log_dens0g - median gas density, 
   log_press0 - median gas pressure, 
   log_entropy0 - median gas entropy, 
   log_mol00 - molecular gas gass, 
   log_HII00 - HII mass, 
   log_OH - 12+log(O/H), 
   np.log10(Z00/0.02) - mass weighted average gas metallicity, 
   log(BHmass) - SMBH mass, 
   log(BHpmass) - SMBH particle mass, 
   logfedd - SMBH accretion rate in eddigton fraction, 
   logacc_gs SMBH accretion rate in g / s, 
   logacc_msyr SMBH accretion rate in Msun / yr, 
   logpower - SMBH total power in erg/s, 
   logLmech - SMBH mechanical luminosity in erg/s, 
   nump_g - number of jet particles inside 0.1Rvir, 
   nump_all - number of jet particles in the box


###track_galaxy_properties

Newer version with better comments throughout - please see the script for a detatiled description

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
 

Output
------------
For a specified range of snapshots and model outputs: 
Image of galaxy with gas face and edge on colored by gas temperature, and stars edge on and face colored by surface density along with a text file with the following:    

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
    
