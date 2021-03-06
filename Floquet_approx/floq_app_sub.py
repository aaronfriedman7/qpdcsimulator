#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:36:45 2020

@author: Aaron
"""

### this script will be in "~/QP/
### slurm scripts should go to "~/submit/"
### 

import os
# import sys
import argparse
import numpy as np
# import itertools as itt
# import glob

### in general, same format as home directory, but replace "home1" with "work" or "scratch"
mainpath = "07375/ajfriedm/FA"
workpath = "/scratch/"+mainpath
storepath = "/work/"+mainpath

### fixed parameters:
# Lcell=4
# phase_diffs = [0.0, 0.01, 0.02, 0.05, 0.1]
# Kvals = [0.2, 1.0, 5.0]
# widths = [0.02,0.05,0.1]
# Jval = 0.1
# Bval = 0.1f
 

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='QP Floquet approximate submit code', prog='main', usage='%(prog)s [options]')
    # parser.add_argument("-ndis", "--ndis", help="disorder realization", default=8,type=int, nargs='?')
    # parser.add_argument("-rstart", "--rstart", help="disorder realization offset", default=0,type=int, nargs='?')
    # parser.add_argument("-interval", "--interval", help="max integration interval", default=100.0,type=float, nargs='?')
    # parser.add_argument("-K1", "--K1", help="K lower value (0.1, 0.2, 0.5, 1.0)", default=0.1,type=float, nargs='?')
    # parser.add_argument("-K2", "--K2", help="K upper (0.1, 0.2, 0.5, 1.0)", default=0.3,type=float, nargs='?')
    # parser.add_argument("-pd", "--pd", help="phase diff (0.0, 0.01, 0.02,0.05, 0.1)", default=0.05,type=float, nargs='?')
    # parser.add_argument("-wid", "--wid", help="width of pulse (0.02,0.05,0.1)", default=0.05,type=float, nargs='?')
    parser.add_argument("-L", "--L", help="number of UNIT CELLS", default=6,type=int, nargs='?')
    # parser.add_argument("-cd", "--cd", help="counter drive strength, 0.0 = no cd, 1.0 = full", default=0.0,type=float, nargs='?')
    parser.add_argument("-rstart", "--rstart", help="disorder realization offset", default=0,type=int, nargs='?')
    parser.add_argument("-app", "--app", help="which Fibonacci 0,1,2,3,4 to do", default=10,type=int, nargs='?')
    # parser.add_argument("-numapps", "--numapps", help="number of concurrent approximates to do", default=3,type=int, nargs='?')
    parser.add_argument("-ndis", "--ndis", help="number of concurrent disorders to do ", default=24,type=int, nargs='?')
    parser.add_argument("-maxdis", "--maxdis", help="max number of disorders to do ", default=240,type=int, nargs='?')
    # parser.add_argument("-B", "--B", help="local field dist. width", default=0.1,type=float, nargs='?')
    # parser.add_argument("-J", "--J", help="local random coupling dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-wt", "--wt", help="wall time in hours", default=24,type=int, nargs='?')
    parser.add_argument("-docorr", "--docorr", help="calc inf. temp correlations??", default=False,type=bool, nargs='?')
    
    args=parser.parse_args()
    # jobs = args.jobs
    app = args.app
    Lc = args.L
    rsd1 = args.rstart
    # napps = args.numapps
    # cdstr = args.cd
    maxdis = args.maxdis
    ndis = args.ndis
    walltime = args.wt
    os.chdir(workpath)
    dc = args.docorr
    
    Ntasks = ndis
    Nnodes = int(np.ceil(float(Ntasks)/24))
    
    # cds = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.5,0.6,0.7,0.8,0.9,1.0]
    cds = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    os.chdir("/home1/07375/ajfriedm/submit/")
    
    
    for cdstr in cds:
        scriptname = "./QPFA_{0}_Slurm.sh".format(cdstr)
        scriptfile = open(scriptname, 'w')
        scriptfile.write('#!/bin/bash')
        scriptfile.write("\n#SBATCH -p normal")
        # scriptfile.write('\nml purge')
        # scriptfile.write('\nml intel/2018.3 mkl')
        # scriptfile.write('\nml python/2.7.14')
        scriptfile.write("\n#SBATCH -J QPFA{0}".format(cdstr))
        scriptfile.write("\n#SBATCH -o QPFA{0}-%j".format(cdstr))
        scriptfile.write("\n#SBATCH -N {0}".format(Nnodes))
        scriptfile.write("\n#SBATCH -n {0}".format(Ntasks))
        # if cdstr == 0.0:
            # scriptfile.write("\n#SBATCH -t {}:00:00".format(2*walltime))
        # else:
        scriptfile.write("\n#SBATCH -t {}:00:00".format(walltime))
        scriptfile.write("\n#SBATCH -A Near-Term-Quantum-Al")
        scriptfile.write("\n#SBATCH --mail-user=aaron.friedman@austin.utexas.edu")
        scriptfile.write("\n#SBATCH --mail-type=end")
        # scriptfile.write("\nexport MKL_NUM_THREADS=1")
        # scriptfile.write('\nexport OMP_NUM_THREADS={0}'.format(Ntasks)) ### for openMP, not MPI
        if dc:
            scriptfile.write("\nibrun python3 /scratch/"+mainpath+"/floquet_approx.py -docorr=True -L={} -app={} -cd={} -rstart={} -ndis={} -maxdis={}".format(Lc,app,cdstr,rsd1,ndis,maxdis))
            # if cdstr == 0.0:
                # scriptfile.write("\nibrun python3 /scratch/"+mainpath+"/floquet_approx.py -docorr=True -L={} -app={} -cd={} -rstart={} -ndis={} -maxdis={} -pbc=True".format(Lc,app1,cdstr,rsd1,ndis,napps,maxdis))
        else:
            scriptfile.write("\nibrun python3 /scratch/"+mainpath+"/floquet_approx.py -L={} -app={} -cd={} -rstart={} -ndis={} -maxdis={}".format(Lc,app,cdstr,rsd1,ndis,maxdis))
            # if cdstr == 0.0:
                # scriptfile.write("\nibrun python3 /scratch/"+mainpath+"/floquet_approx.py -L={} -app={} -cd={} -rstart={} -ndis={}  -maxdis={} -pbc=True".format(Lc,app,cdstr,rsd1,ndis,maxdis))
        scriptfile.write('\necho "job done"')
        scriptfile.close()
        os.system("chmod 700 "+scriptname)
        os.system("sbatch "+scriptname)
        # submitted +=1