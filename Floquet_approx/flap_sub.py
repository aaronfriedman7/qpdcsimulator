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
    parser.add_argument("-L", "--L", help="number of UNIT CELLS", default=4,type=int, nargs='?')
    parser.add_argument("-Nts", "--Nts", help="number of periods to simulate", default=500,type=int, nargs='?')
    # parser.add_argument("-cd", "--cd", help="counter drive strength, 0.0 = no cd, 1.0 = full", default=0.0,type=float, nargs='?')
    # parser.add_argument("-rstart", "--rstart", help="disorder realization offset", default=0,type=int, nargs='?')
    # parser.add_argument("-appstart", "--appstart", help="which Fibonacci 0,1,2,3,4 -> 1,1,2,3,5 to start with", default=8,type=int, nargs='?')
    # parser.add_argument("-numapps", "--numapps", help="number of concurrent approximates to do", default=4,type=int, nargs='?')
    # parser.add_argument("-ndis", "--ndis", help="number of concurrent disorders to do ", default=6,type=int, nargs='?')
    # parser.add_argument("-maxdis", "--maxdis", help="max number of disorders to do ", default=18,type=int, nargs='?')
    # parser.add_argument("-B", "--B", help="local field dist. width", default=0.1,type=float, nargs='?')
    # parser.add_argument("-J", "--J", help="local random coupling dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-wt", "--wt", help="wall time in hours", default=6,type=int, nargs='?')
    
    args=parser.parse_args()
    # jobs = args.jobs
    # app1 = args.appstart
    Lc = args.L
    # rsd1 = args.rstart
    # napps = args.numapps
    # cdstr = args.cd
    # maxdis = args.maxdis
    # ndis = args.ndis
    Nts = args.Nts
    walltime = args.wt
    os.chdir(workpath)
    
    
    Ntasks = 24
    Nnodes = 1

    
    #os.system("cp /work/"+mainpath+"/qp_tevo.py /scratch/"+mainpath+"/qp_tevo.py ")
    os.chdir("/home1/07375/ajfriedm/submit/")
    #for (Kval,pd,wid,rsd) in param_iterator:
    # for rsd in range(min(ndis,jobs)):
    scriptname = "./QPFA_strob_Slurm.sh"
    scriptfile = open(scriptname, 'w')
    scriptfile.write('#!/bin/bash')
    scriptfile.write("\n#SBATCH -p normal")
    # scriptfile.write('\nml purge')
    # scriptfile.write('\nml intel/2018.3 mkl')
    # scriptfile.write('\nml python/2.7.14')
    scriptfile.write("\n#SBATCH -J QPFA_strob")
    scriptfile.write("\n#SBATCH -o QPFA_strob-%j")
    scriptfile.write("\n#SBATCH -N {0}".format(Nnodes))
    scriptfile.write("\n#SBATCH -n {0}".format(Ntasks))
    scriptfile.write("\n#SBATCH -t {}:00:00".format(walltime))
    scriptfile.write("\n#SBATCH -A Near-Term-Quantum-Al")
    scriptfile.write("\n#SBATCH --mail-user=aaron.friedman@austin.utexas.edu")
    scriptfile.write("\n#SBATCH --mail-type=end")
    # scriptfile.write("\nexport MKL_NUM_THREADS=1")
    # scriptfile.write('\nexport OMP_NUM_THREADS={0}'.format(Ntasks)) ### for openMP, not MPI
    scriptfile.write("\nibrun python3 /scratch/"+mainpath+"/flap_strob.py {} {}".format(Lc,Nts))
    scriptfile.write('\necho "job done"')
    scriptfile.close()
    os.system("chmod 700 "+scriptname)
    os.system("sbatch "+scriptname)
    # submitted +=1