#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:40:19 2020

@author: Aaron
"""

import glob
# import argparse
import os
import sys
# from scipy import linalg as lin

# need to set MKL num threads? i.e. to Ncore ... note this has to be before numpy is imported. Will probably be set as part of submission?
import numpy as np
import scipy.sparse as ss

# import matplotlib.pyplot as plt
# os.environ["MKL_NUM_THREADS"] = "1" #### for non MPI jobs.

mainpath = "07375/ajfriedm/FA/"
workpath = "/scratch/"+mainpath
os.chdir(workpath)
# workpath = "./"

## the two functions below are inverses.
def spinstr(spinint, N):
    return bin(int(spinint))[2:].zfill(N)

def getint(N,config):
    return int(config.zfill(N),base=2)

### confirmed, save as array
def Zall(Lcell):
    try:
        op = np.load(workpath+"Z_all_L{0}.npy".format(Lcell))
    except:
        Size = 4**Lcell
        op = np.zeros((2*Lcell,Size),dtype=float)
        for state in range(Size):
            scfg = spinstr(state,2*Lcell)
            for j in range(2*Lcell):
                op[j,state] += 2.0*int(scfg[j])-1.0
        np.save(workpath+"Z_all_L{0}.npy".format(Lcell),op)
    return op

### confirmed, save as array
def ZZall(Lcell):
    Ns = 2*Lcell
    try:
        op = np.load(workpath+"ZZ_all_L{0}.npy".format(Lcell))
    except:
        Size = 2**Ns
        op = np.zeros((Ns,Size),dtype=float)
        for state in range(Size):
            scfg = spinstr(state,Ns)
            for j in range(Ns):
                j2 = (j+1)%Ns
                op[j,state] += (2.0*int(scfg[j])-1.0)*(2.0*int(scfg[j2])-1.0)
        np.save(workpath+"ZZ_all_L{0}.npy".format(Lcell),op)
    return op


### confirmed save as array
def Xop(Lcell,j=0):
    try:
        op = ss.load_npz(workpath+"X_L{0}_j{1}.npz".format(Lcell,j))
    except:
        Size = 4**Lcell
        op = ss.dok_matrix((Size,Size),dtype=float)
        for state in range(Size):
            newstate = state^(2**(2*Lcell-1-j))
            op[newstate,state] += 1
        op = ss.csr_matrix(op)
        ss.save_npz(workpath+"X_L{0}_j{1}.npz".format(Lcell,j),op)
    return op

### confirmed save as array
def XXop(Lcell,j=0):
    Ns = 2*Lcell
    try:
        op = ss.load_npz(workpath+"XX_L{0}_j{1}.npz".format(Lcell,j))
    except:
        Size = 2**Ns
        op = ss.dok_matrix((Size,Size),dtype=float)
        for state in range(Size):
            j2 = (j+1)%(Ns)
            newstate = state^(2**(Ns-1-j) + 2**(Ns-1-j2))
            op[newstate,state] += 1
        op = ss.csr_matrix(op)
        ss.save_npz(workpath+"XX_L{0}_j{1}.npz".format(Lcell,j),op)
    return op


pd=0.05
wid=0.05
K1=0.1
K2=0.3
J=0.1
B=0.1

Lcell = int(sys.argv[1]) # number of unit cells
Ns = 2*Lcell

Nts = int(sys.argv[2]) # number of periods

cds = [0.0,0.0,0.05,0.1,0.15,0.2,0.25,0.5,0.75,1.0]
Cn = len(cds)

apps = list(range(8,16))

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)

data = np.zeros((4,Ns,Nts))

Xguy = None

if rank == 0:
    
    Zguy = Zall(Lcell)
    ZZguy = ZZall(Lcell)
else:
    Zguy = np.empty((Ns,2**Ns))
    ZZguy = np.empty((Ns,2**Ns))

comm.Bcast(Zguy,root=0)
comm.Bcast(ZZguy,root=0)

for foo1,cd in enumerate(cds):
    for foo2,app in enumerate(apps):
        tasknum = (Cn*foo1 + 8*foo2)%24
        if rank == tasknum:
            if foo1 == 0:
                pbc = True
                bdystr = "_pbc_"
            else:
                pbc = False
                bdystr = "_"
            eignames = workpath+"Feigs_app{}".format(app)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd*.npy".format(cd,Lcell,pd,wid,K1,K2,J,B)
            vecnames = workpath+"Fvecs_app{}".format(app)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd*.npy".format(cd,Lcell,pd,wid,K1,K2,J,B)
            eigfs = list(glob.glob(eignames))
            vecfs = list(glob.glob(vecnames))
            if len(eigfs) == 0 or len(vecfs) == 0:
                continue
            ersds = []
            vrsds = []
            for ename in eigfs:
                rsd = int(ename[ename.find("_rsd")+4:ename.find(".npy")])
                ersds.append(rsd)
            for vname in vecfs:
                rsd = int(vname[vname.find("_rsd")+4:vname.find(".npy")])
                vrsds.append(rsd)
            rsds = [x for x in ersds if x in vrsds] # common realizations
            dis = np.random.choice(rsds)
            
            fname =  workpath+"Feigs_app{}".format(app)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd{}.npy".format(cd,Lcell,pd,wid,K1,K2,J,B,dis)
            eigs = np.load(fname)
            fname = workpath+"Fvecs_app{}".format(app)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd{}.npy".format(cd,Lcell,pd,wid,K1,K2,J,B,dis)
            vecs = np.load(fname)
            
            vecsH = np.transpose(vecs.conj())
            # unitary = np.matmul(vecs,np.matmul(np.diag(eigs),vecsH))
            for site in range(Ns):
                Obs = np.diag(Zguy[site])
                Obs = np.matmul(vecsH,np.matmul(Obs,vecs))
                vals = np.array([np.average(np.diag(np.matmul(np.diag(eigs**(-t)),np.matmul(Obs,np.matmul(np.diag(eigs**(t)),Obs))))).real for t in range(Nts)])
                data[0,site,:] = vals[:]
                
                Obs = np.diag(ZZguy[site])
                Obs = np.matmul(vecsH,np.matmul(Obs,vecs))
                vals = np.array([np.average(np.diag(np.matmul(np.diag(eigs**(-t)),np.matmul(Obs,np.matmul(np.diag(eigs**(t)),Obs))))).real for t in range(Nts)])
                data[1,site,:] = vals[:]
                
                Xguy = Xop(Lcell,site)
                Obs = np.matmul(vecsH,Xguy.dot(vecs))
                vals = np.array([np.average(np.diag(np.matmul(np.diag(eigs**(-t)),np.matmul(Obs,np.matmul(np.diag(eigs**(t)),Obs))))).real for t in range(Nts)])
                data[2,site,:] = vals[:]
                
                Xguy = XXop(Lcell,site)
                Obs = np.matmul(vecsH,Xguy.dot(vecs))
                vals = np.array([np.average(np.diag(np.matmul(np.diag(eigs**(-t)),np.matmul(Obs,np.matmul(np.diag(eigs**(t)),Obs))))).real for t in range(Nts)])
                data[3,site,:] = vals[:]
            
            dname = workpath+"data_app{}".format(app)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd{}_t{}.npy".format(cd,Lcell,pd,wid,K1,K2,J,B,dis,Nts)
            np.save(dname,data)
            
            
            
        
