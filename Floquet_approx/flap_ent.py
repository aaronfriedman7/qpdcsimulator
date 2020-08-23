#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:20:58 2020

@author: Aaron
"""


import glob
import argparse
import os
import sys
# from scipy import linalg as lin

# need to set MKL num threads? i.e. to Ncore ... note this has to be before numpy is imported. Will probably be set as part of submission?
import numpy as np

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

def EntEnt(Lcell, vec, cut):
    Ns = 2*Lcell
    lower = cut
    upper = Ns - cut
    Psi = np.reshape(vec,(2**(lower),2**(upper)))
    lambdas = np.linalg.svd(Psi, compute_uv = False)
    EntEnt = 0.0
    for foo in range(len(lambdas)):
        value = (lambdas[foo])**2
        if value > 0.0:
            EntEnt -= np.math.log(value)*value
    return EntEnt


pd=0.05
wid=0.05
K1=0.1
K2=0.3
J=0.1
B=0.1

# cds = [0.0,0.0,0.05,0.1,0.15,0.2,0.25,0.5,0.75,1.0]
# Cn = len(cds)

# apps = list(range(8,16))
# Napp = len(apps)
app = 10

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='xxz random unitary spectral form factor', prog='main', usage='%(prog)s [options]')
    parser.add_argument("-L", "--L", help="number of UNIT CELLS", default=4,type=int, nargs='?')
    # parser.add_argument("-pbc", "--pbc", help="periodic boundaries?", default=False,type=bool, nargs='?')
    parser.add_argument("-cd", "--cd", help="counter-drive factor", default=0.0,type=float, nargs='?')
    
    params=parser.parse_args()
    # pbc = params.pbc
    
    Ns = 2*params.L
    eigents = np.zeros(2**Ns)
    
    # for foo1,app in enumerate(apps):
        # rootnum = 3*foo1
        # if rank == rootnum:
    if rank == 0:
        ents = []
        # if pbc:
            # bdystr = "_pbc_"
        # else:
    bdystr = "_"
    vecfiles = list(glob.glob(workpath+"Fvecs_app{}".format(app)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd*.npy".format(params.cd,params.L,pd,wid,K1,K2,J,B)))
    Nf = len(vecfiles)
    for foo2,vecname in enumerate(vecfiles):
        # tasknum = rootnum + (foo2%3)
        tasknum = foo2%24
        if rank != tasknum:
            continue
        rsd = int(vecname[vecname.find("_rsd")+4:vecname.find(".npy")])
        vecs = np.load(vecname)
        
        # todo = np.random.choice(range(2**Ns),50,replace=False)
        for vind in range(2**Ns):
            # thisone = todo[dum]
            evec = vecs[:,vind]
            ent = EntEnt(params.L,evec,params.L)
            eigents[vind] = ent
        entval = np.average(eigents)
        # if rank > rootnum and rank < rootnum +3:
        if rank > 0:
            # comm.send(entval,dest=rootnum)
            comm.send(entval,dest=0)
        else:
            ents.append(entval)
            # entval = comm.recv(source=rootnum+1)#
            for other in range(1,24):
                entval = comm.recv(source=other)
                ents.append(entval)
            # entval = comm.recv(source=rootnum+2)
            # ents.append(entval)
        if rank == 0:
            entname = "ent_app{}".format(app)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(params.cd,params.L,pd,wid,K1,K2,J,B)
            np.save(entname,np.array(ents))
            
            
            
            
        
