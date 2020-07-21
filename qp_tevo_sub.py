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
import glob

### in general, same format as home directory, but replace "home1" with "work" or "scratch"
mainpath = "07375/ajfriedm/QP_tevo"
workpath = "/scratch/"+mainpath
storepath = "/work/"+mainpath

### fixed parameters:
# Lcell=4
# phase_diffs = [0.0, 0.01, 0.02, 0.05, 0.1]
# Kvals = [0.2, 1.0, 5.0]
# widths = [0.02,0.05,0.1]
# Jval = 0.1
# Bval = 0.1


def jobname(params,counter):
    if counter:
        return "cd_L{0}_pd{1}_wid{2}_K1{3}_K2{4}_J{5}_B{6}".format(params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B)
    else:
        return "L{0}_pd{1}_wid{2}_K1{3}_K2{4}_J{5}_B{6}".format(params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B)
 

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='xxz random unitary spectral form factor', prog='main', usage='%(prog)s [options]')
    parser.add_argument("-ndis", "--ndis", help="disorder realization", default=8,type=int, nargs='?')
    # parser.add_argument("-rstart", "--rstart", help="disorder realization offset", default=0,type=int, nargs='?')
    parser.add_argument("-tmax", "--tmax", help="max time", default=250,type=int, nargs='?')
    parser.add_argument("-interval", "--interval", help="max integration interval", default=100.0,type=float, nargs='?')
    parser.add_argument("-K1", "--K1", help="K lower value (0.1, 0.2, 0.5, 1.0)", default=0.1,type=float, nargs='?')
    parser.add_argument("-K2", "--K2", help="K upper (0.1, 0.2, 0.5, 1.0)", default=0.3,type=float, nargs='?')
    parser.add_argument("-pd", "--pd", help="phase diff (0.0, 0.01, 0.02,0.05, 0.1)", default=0.05,type=float, nargs='?')
    parser.add_argument("-wid", "--wid", help="width of pulse (0.02,0.05,0.1)", default=0.05,type=float, nargs='?')
    parser.add_argument("-L", "--L", help="number of UNIT CELLS", default=7,type=int, nargs='?')
    parser.add_argument("-cd", "--cd", help="counter drive strength, 0.0 = no cd, 1.0 = full", default=0.0,type=float, nargs='?')
    parser.add_argument("-B", "--B", help="local field dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-J", "--J", help="local random coupling dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-wt", "--wt", help="wall time in hours", default=2,type=int, nargs='?')
    
    args=parser.parse_args()
    # jobs = args.jobs
    L = args.L
    ndis = args.ndis
    tmax = args.tmax
    maxint = args.interval
    Kval1 = args.K1
    Kval2 = args.K2
    pd = args.pd
    wid = args.wid
    Lcell = args.L
    Jval = args.J
    Bval = args.B
    cd_strength = args.cd
    walltime = args.wt
    os.chdir(workpath)
    if cd_strength != 0.0:
        counter = True
        data_names = "corrs_r_cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd*_st*.npy".format(cd_strength,Lcell,pd,wid,Kval1,Kval2,Jval,Bval)
    else:
        counter = False
        data_names = "corrs_r_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd*_st*.npy".format(Lcell,pd,wid,Kval1,Kval2,Jval,Bval)
    
    flist = glob.glob(data_names)
    print("found {} files".format(len(flist)))
    rsds = []
    if len(flist) > 0:
        for fname in flist:
            rsd = int(fname[fname.find("_rsd")+4:fname.find("_st")])
            rsds.append(rsd)
    else:
        rstart = 0
    if rsds == []:
        rstart = 0
    else:
        rstart = max(rsds) + 1
    
    print("doing counter drive = {}".format(counter))
    print()
    
    Ntasks = 3*ndis
    Nnodes = int(np.ceil(float(ndis)/8.0))
    
    # param_iterator = itt.product(Kvals,phase_diffs,widths,range(ndis),range(nstates))
    # param_iterator = itt.product(Kvals,phase_diffs,widths,range(ndis))
    ### this will be iterated starting from the last index, which is nice.
    
    #os.system("cp /work/"+mainpath+"/qp_tevo.py /scratch/"+mainpath+"/qp_tevo.py ")
    os.chdir("/home1/07375/ajfriedm/submit/")
    #for (Kval,pd,wid,rsd) in param_iterator:
    # for rsd in range(min(ndis,jobs)):
    scriptname = "./QP_{0}_Slurm.sh".format(jobname(args,counter))
    scriptfile = open(scriptname, 'w')
    scriptfile.write('#!/bin/bash')
    scriptfile.write("\n#SBATCH -p normal")
    # scriptfile.write('\nml purge')
    # scriptfile.write('\nml intel/2018.3 mkl')
    # scriptfile.write('\nml python/2.7.14')
    scriptfile.write("\n#SBATCH -J {0}".format(jobname(args,counter)))
    scriptfile.write("\n#SBATCH -o {0}-%j".format(jobname(args,counter)))
    scriptfile.write("\n#SBATCH -N {0}".format(Nnodes))
    scriptfile.write("\n#SBATCH -n {0}".format(Ntasks))
    scriptfile.write("\n#SBATCH -t {}:00:00".format(walltime))
    scriptfile.write("\n#SBATCH -A Near-Term-Quantum-Al")
    scriptfile.write("\n#SBATCH --mail-user=aaron.friedman@austin.utexas.edu")
    scriptfile.write("\n#SBATCH --mail-type=end")
    scriptfile.write("\nexport MKL_NUM_THREADS=1")
    # scriptfile.write('\nexport OMP_NUM_THREADS={0}'.format(Ntasks)) ### for openMP, not MPI
    if counter:
        for foo in range(10):
            scriptfile.write("\nibrun python3 /scratch/"+mainpath+"/qp_tevo_cd.py -L={0} -K1={1} -K2={2} -wid={3} -pd={4}  -B={5} -J={6} -maxint={7} -tf={8} -rstart={9} -cd={10}".format(Lcell,Kval1,Kval2,wid,pd,Bval,Jval,maxint,tmax,rstart+foo*ndis,cd_strength))
    else:
        scriptfile.write("\nibrun python3 /scratch/"+mainpath+"/qp_tevo.py -L={0} -K1={1} -K2={2} -wid={3} -pd={4}  -B={5} -J={6} -maxint={7} -tf={8} -rstart={9}".format(Lcell,Kval1,Kval2,wid,pd,Bval,Jval,maxint,tmax,rstart))
    scriptfile.write('\necho "job done"')
    scriptfile.close()
    os.system("chmod 700 "+scriptname)
    os.system("sbatch "+scriptname)
    # submitted +=1