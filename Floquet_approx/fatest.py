#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:55:36 2020

@author: Aaron
"""

import glob
import argparse
# import os
# import sys
from scipy import linalg as lin
import time

# need to set MKL num threads? i.e. to Ncore ... note this has to be before numpy is imported. Will probably be set as part of submission?
import numpy as np
import scipy.integrate as sint
import scipy.sparse as ss

# import matplotlib.pyplot as plt
# os.environ["MKL_NUM_THREADS"] = "1" #### for non MPI jobs.

workpath = "./"

# In general, all file loading try statements should only accept particular exceptions. But these have been tested.
# golden = 0.5*(1.0+np.sqrt(5.0))


# convenient file label string
def fhead(params,rsd):
    return "L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd{7}".format(params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B,rsd)

## givds F_n, F_n+1 where F_0 = F_1 = 1.
def Fibnums(n):
    a,b = 0,1
    m = 0
    while m <= n:
        a,b = b, a+b
        m+=1
    return (a,b)

def floatmod(numb,div):
    guess = int(round(numb/div))
    rem = numb - guess*div
    if rem < 0:
        while rem <0:
            rem += div
    elif rem >= div:
        while rem >= div:
            rem -= div
    return rem

# PHYSICS 
    
## the two functions below are inverses.
def spinstr(spinint, N):
    return bin(int(spinint))[2:].zfill(N)

def getint(N,config):
    return int(config.zfill(N),base=2)

### confirmed, save as array
def Zop(params,j=0):
    Lcell = params.L
    try:
        op = np.load(workpath+"Z_L{0}_j{1}.npy".format(Lcell,j))
    except:
        Size = 4**Lcell
        op = np.zeros((Size),dtype=float)
        for state in range(Size):
            scfg = spinstr(state,2*Lcell)
            op[state] += 2.0*int(scfg[j])-1.0
        np.save(workpath+"Z_L{0}_j{1}.npy".format(Lcell,j),op)
    return op

### confirmed, save as array
def Zall(params):
    Lcell = params.L
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
def ZZall(params):
    Ns = 2*params.L
    try:
        op = np.load(workpath+"ZZ_all_L{0}.npy".format(params.L))
    except:
        Size = 2**Ns
        op = np.zeros((Ns,Size),dtype=float)
        for state in range(Size):
            scfg = spinstr(state,Ns)
            for j in range(Ns):
                j2 = (j+1)%Ns
                op[j,state] += (2.0*int(scfg[j])-1.0)*(2.0*int(scfg[j2])-1.0)
        np.save(workpath+"ZZ_all_L{0}.npy".format(params.L),op)
    return op


### confirmed save as array
def Xop(params,j=0):
    Lcell = params.L
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
def XXop(params,j=0):
    Ns = 2*params.L
    try:
        op = ss.load_npz(workpath+"XX_L{0}_j{1}.npz".format(params.L,j))
    except:
        Size = 2**Ns
        op = ss.dok_matrix((Size,Size),dtype=float)
        for state in range(Size):
            j2 = (j+1)%(Ns)
            newstate = state^(2**(Ns-1-j) + 2**(Ns-1-j2))
            op[newstate,state] += 1
        op = ss.csr_matrix(op)
        ss.save_npz(workpath+"XX_L{0}_j{1}.npz".format(params.L,j),op)
    return op

# PROBLEM SPECIFIC

# Gaussian pulse, a la Brayden
def pulse_func(time,omega,stddev,accuracy=10.0**(-8.0),giveup=1000,mintry=25):
    f = 0.5*omega
    m = 1
    goodjob = 0
    while m <= giveup:
        newterm = omega*np.exp(-0.5*((omega*stddev*m)**2))*np.cos(m*omega*time+m*np.pi)
        f += newterm
        m += 1
        if m < mintry:
            continue
        else:
            if np.abs(newterm) > accuracy:
                continue
            else:
                goodjob +=1
                if goodjob < 5:
                    continue
                else:
                    break
    return f

# all static terms, saves to work path to avoid delay
def static_ham(params,rsd):
    Lcell = params.L
    pbc = params.pbc
    if pbc:
        maxsite = Lcell
        Hname = "Hstat_pbc_L{0}_K1_{1}_K2_{2}_J{3}_B{4}_rsd{5}.npz".format(Lcell,params.K1,params.K2,params.J,params.B,rsd)
    else:
        maxsite = Lcell-1
        Hname = "Hstat_L{0}_K1_{1}_K2_{2}_J{3}_B{4}_rsd{5}.npz".format(Lcell,params.K1,params.K2,params.J,params.B,rsd)
    try:
        H = ss.load_npz(workpath+Hname)
    except:
        Size = 4**Lcell
        H = ss.dok_matrix((Size,Size),dtype=complex) ## or 'c'
        try:
            Kvalues = np.load(workpath+"Kvals_L{0}_K1_{1}_K2_{2}_rsd{3}.npy".format(Lcell,params.K1,params.K2,rsd))
        except:
            Kvalues = np.random.uniform(params.K1,params.K2,(2,maxsite))*np.random.choice(np.array([-1.0,1.0]),(2,maxsite))
            np.save(workpath+"Kvals_L{0}_K1_{1}_K2_{2}_rsd{3}.npy".format(Lcell,params.K1,params.K2,rsd),Kvalues)
        try:
            Jvalues = np.load(workpath+"Jvals_L{0}_J{1}_rsd{2}.npy".format(Lcell,params.J,rsd))
        except:
            Jvalues = np.random.uniform(-params.J,params.J,(2,Lcell))
            np.save(workpath+"Jvals_L{0}_J{1}_rsd{2}.npy".format(Lcell,params.J,rsd),Jvalues)
        try:
            fields = np.load(workpath+"Bvals_L{0}_B{1}_rsd{2}.npy".format(Lcell,params.B,rsd))
        except:
            fields = np.random.uniform(-params.B,params.B,(3,2*Lcell))
            np.save(workpath+"Bvals_L{0}_B{1}_rsd{2}.npy".format(Lcell,params.B,rsd),fields)
        # constpiece = 0.5*np.pi + 0.25*np.pi/(1.0+np.sqrt(5.0)) # optional
        for state in range(Size):
            scfg = spinstr(state,2*Lcell)
            # H[state,state] -= constpiece # optional
            for siteA in range(2*Lcell):
                spin = (2*int(scfg[siteA])-1)
                # fields
                H[state,state] += fields[2,siteA]*spin
                H[state^(2**(2*Lcell-siteA-1)),state] += fields[0,siteA] + spin*1j*fields[1,siteA]
            for cell in range(Lcell):
                site1 = 2*cell
                site2 = 2*cell+1
                if scfg[site1] == scfg[site2]:
                    H[state,state] += Jvalues[1,cell]
                else:
                    H[state,state] -= Jvalues[1,cell]
                H[state^( 2**(2*Lcell-site1-1) + 2**(2*Lcell-site2-1) ),state] += Jvalues[0,cell]
            for cell in range(maxsite):
                site1 = 2*cell+1
                site2 = (1+site1)%(2*Lcell)
                if scfg[site1] == scfg[site2]:
                    H[state,state] += Kvalues[1,cell]
                else:
                    H[state,state] -= Kvalues[1,cell]
                H[state^(2**(2*Lcell-site1-1)+2**(2*Lcell-site2-1)),state] += Kvalues[0,cell]
        H = ss.csr_matrix(H)
        ss.save_npz(workpath+Hname,H)
    return H


# confirmed: diagonal elements, np array
def pulse_ham_z(params):
    Lcell = params.L
    try:
        Hdiag = np.load(workpath+"pulse_z_L{0}.npy".format(Lcell))
    except:
        Hdiag = np.zeros((4**Lcell),dtype=float)
        for state in range(4**Lcell):
            scfg = spinstr(state,2*Lcell)
            for cell in range(Lcell):
                if scfg[2*cell] == scfg[2*cell+1]:
                    Hdiag[state] += 0.5
                else:
                    Hdiag[state] -= 0.5
        np.save(workpath+"pulse_z_L{0}.npy".format(Lcell),Hdiag)
    return Hdiag


# confirmed
def pulse_ham_x(params):
    Lcell = params.L
    try:
        H = ss.load_npz(workpath+"pulse_x_L{0}.npz".format(Lcell))
    except:
        Size = 4**Lcell
        H = ss.dok_matrix((Size,Size),dtype=float)
        for state in range(Size):
            for cell in range(Lcell):
                newstate = state^(2**(2*Lcell-2*cell-1)+2**(2*Lcell-2*cell-2))
                H[newstate,state] += 0.5
        H = ss.csr_matrix(H)
        ss.save_npz(workpath+"pulse_x_L{0}.npz".format(Lcell),H)
    return H


# MAIN RHS FUNCTION
# NOTE: for some reason, it's essential to have the extra "psimem" for performance
# Zp = ss.csc_matrix(np.diag(Zdiagarray)))
## sparse.dot(nparray) = nparray by default
def evo_U_RHS(t,y,params,omegaZ,Ns,Xp,Zp,Hs,Umem,Uact):
    Uact = np.reshape(y,(2**Ns,2**Ns))
    Umem = -1j*np.reshape(Hs.dot(Uact),4**Ns)
    Umem -= 1j*(1.0-params.pd)*pulse_func(t,2.0*np.pi,params.wid)*np.reshape( Xp.dot(Uact),(4**Ns))
    Umem -= 1j*(1.0-params.pd)*pulse_func(t,omegaZ,params.wid)*np.reshape(Zp.dot(Uact),(4**Ns))
    return Umem

def LevStats(evlist):
    vals = len(evlist)
    delts = [(evlist[i+1]-evlist[i]) for i in range(0,vals-1)]
    rlist =[]
    for j in range(0,len(delts)-1):
        r = min(delts[j],delts[j+1])/max(delts[j],delts[j+1])
        rlist.append(r)
    return np.average(rlist)

def Inf_corrX(obs,evecs):
    obs1 = obs.dot(evecs)
    evecsH = np.transpose(evecs.conj())
    obs1 = np.matmul(evecsH,obs1)
    main = np.diag(obs1)
    main1 = main*main.conj()
    return np.average(main1.real)

def Inf_corrZ(obs,evecs):
    obs1 = np.diag(obs)
    obs1 = np.matmul(obs1,evecs)
    evecsH = np.transpose(evecs.conj())
    obs1 = np.matmul(evecsH,obs1)
    main = np.diag(obs1)
    main1 =  main*main.conj()
    return np.average(main1.real)

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

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# print(rank)
# Ncore = comm.Get_size()

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='xxz random unitary spectral form factor', prog='main', usage='%(prog)s [options]')
    parser.add_argument("-L", "--L", help="number of UNIT CELLS", default=4,type=int, nargs='?')
    parser.add_argument("-B", "--B", help="local field dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-K2", "--K2", help="intercell coupling dist. width", default=0.3,type=float, nargs='?')
    parser.add_argument("-K1", "--K1", help="intercell coupling dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-J", "--J", help="local random coupling dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-pd", "--pd", help="phase difference, same as Brayden", default=0.05,type=float, nargs='?')
    parser.add_argument("-wid", "--wid", help="std dev for pulses (width)", default=0.05,type=float, nargs='?')
    parser.add_argument("-maxint", "--maxint", help="maximum time interval", default=100.0,type=float, nargs='?')
    parser.add_argument("-pbc", "--pbc", help="periodic boundaries?", default=False,type=bool, nargs='?')
    parser.add_argument("-docorr", "--docorr", help="calc inf. temp correlations??", default=False,type=bool, nargs='?')
    parser.add_argument("-cd", "--cd", help="counter-drive factor", default=0.0,type=float, nargs='?')
    parser.add_argument("-app", "--app", help="which Fibonacci", default=11,type=int, nargs='?')
    # parser.add_argument("-appsN", "--appsN", help="how many apps to do", default=4,type=int, nargs='?')
    parser.add_argument("-rsd", "--rsd", help="disorder realization offset", default=0,type=int, nargs='?')
    # parser.add_argument("-ndis", "--ndis", help="num concurrent tasks", default=6,type=int, nargs='?')
    # parser.add_argument("-maxdis", "--maxdis", help="max num realizations", default=18,type=int, nargs='?')
    
    ### do napps concurrently
    ### do ndis concurrently
    ### ntasks is number of concurrent disorder realizations
    
    
    # rank = params.ntasks*tasknum + rsdbase
    
    # load arguments
    params=parser.parse_args()
    #ti = float(params.maxint)/float(params.div) ## just using ti as a placeholder
    starttime = time.time()
    
    appnum = params.app
    rsd = params.rsd
    
    Ns = 2*params.L
    Zpulse = pulse_ham_z(params) - params.cd*(Zop(params,0) + Zop(params,2*params.L-1))
    Zp = ss.csc_matrix(np.diag(Zpulse))
    Xpulse = pulse_ham_x(params) - params.cd*(Xop(params,0) + Xop(params,2*params.L-1))
    periodic = params.pbc
    if periodic:
        bdystr = "_pbc_"
    else:
        bdystr = "_"
    
    if params.docorr:
        corrdata = np.zeros((4,Ns))
        Zguy = Zall(params)
        ZZguy = ZZall(params)
    
    rsdname = "rsds_app{}".format(appnum)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(params.cd,params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B)
    rname = "rs_app{}".format(appnum)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(params.cd,params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B)
    if params.docorr:
        corrname = "corr_app{}".format(appnum)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(params.cd,params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B)
        try:
            corrs = (np.load(corrname)).tolist()
        except:
            corrs = []
    entname = "ent_app{}".format(appnum)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(params.cd,params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B)
    Fibtup = Fibnums(appnum)
    try:
        rsds = (np.load(rsdname)).tolist()
    except:
        rsds = []
    try:
        rs = (np.load(rname)).tolist()
        ents = (np.load(entname)).tolist()
    except:
        rs = []
        ents = []
    
    OmegaZ = 2.0*np.pi*Fibtup[1]/Fibtup[0]
    Tmax = Fibtup[0]
    print("{}th Fibonacci = {}, next one is {}, Tmax (period) = {}".format(appnum,Fibtup[0],Fibtup[1],Tmax))
    print()
            
    
    eig_name = workpath+"Feigs_app{}".format(appnum)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd{}.npy".format(params.cd,params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B,rsd)
    vec_name = workpath+"Fvecs_app{}".format(appnum)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd{}.npy".format(params.cd,params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B,rsd)
    Elist = glob.glob(eig_name)
    Vlist = glob.glob(vec_name)
    if len(Elist) == 0 or len(Vlist) == 0:
        np.random.seed(rsd)
        # print("Doing state {}".format(st))
        Hstatic = static_ham(params,rsd)
        
        data_name = workpath+"U_app{}".format(appnum)+bdystr+"cd{}_{}.npy".format(params.cd,fhead(params,rsd))

        Uact = np.eye(2**Ns,dtype=complex)
        # print()
        # print("U actual has shape ",np.shape(Uact))
        initial = np.reshape(Uact.copy(),(4**Ns))
        Umem = initial.copy()
        
        # print("initial state has shape {}, Uactual has shape {}, Umem has shape {}".format(np.shape(initial),np.shape(Uact),np.shape(Umem)))
        
        num_runs = int(np.ceil((Tmax)/float(params.maxint)))        
            
        for run in range(num_runs):
            tstart = float(params.maxint*run)
            tstop = min(float(params.maxint*(run+1)),Tmax)
            # print("run {} from time {} to {}".format(run,tstart,tstop))
            # print()
            
            if tstart > 0.0:
                initial = np.load(workpath+"Uvec_app{}".format(appnum)+bdystr+"cd{}_{}_time{}.npy".format(params.cd,fhead(params,rsd),int(tstart)))
                
            teval = np.array([tstart+0.5*(tstop-tstart),tstop])
            Usol = sint.solve_ivp(evo_U_RHS,(tstart,tstop),initial,t_eval=teval,args=(params,OmegaZ,2*params.L,Xpulse,Zp,Hstatic,Umem,Uact),max_step=0.5*params.wid,rtol=1e-6,atol=1e-8)
            ThisU = Usol.y[:,-1]
            print("Solution has shape ",np.shape(ThisU))
            np.save(workpath+"Uvec_app{}".format(appnum)+bdystr+"cd{}_{}_time{}.npy".format(params.cd,fhead(params,rsd),int(tstop)),ThisU)
            
        print("stop time = {}".format(tstop))
        print()
            
        Floq = np.reshape(ThisU,(2**Ns,2**Ns))
        
        Fvals,Fvecs = lin.eig(Floq,overwrite_a=True)
            
        np.save(eig_name,Fvals)
        np.save(vec_name,Fvecs)
    else:
        Fvals = np.load(eig_name)
        Fvecs = np.load(vec_name)
        
    phases = sorted(np.angle(Fvals)%(2.0*np.pi),key=float)
            
    rratio = LevStats(phases)
            
    eigents = np.zeros(2**Ns)
    # todo = np.random.choice(range(2**Ns),50,replace=False)
    for vind in range(2**Ns):
        # thisone = todo[dum]
        evec = Fvecs[:,vind]
        ent = EntEnt(params.L,evec,params.L)
        eigents[vind] = ent
    entval = np.average(eigents)
    
    if params.docorr:
        corrdata = np.zeros((4,Ns))
        Zguy = Zall(params)
        ZZguy = ZZall(params)
        for site in range(Ns):
            corrdata[0,site] = Inf_corrX(Xop(params,site),Fvecs)
            corrdata[1,site] = Inf_corrZ(Zguy[site,:],Fvecs)
            corrdata[2,site] = Inf_corrX(XXop(params,site),Fvecs)
            corrdata[3,site] = Inf_corrZ(ZZguy[site,:],Fvecs)
            
    endtime = time.time()
    print("that took {} long".format(endtime-starttime))
    
        
    ## at end:
    # np.save(workpath+"corrs_{0}.npy".format(fhead(params,rsd,st)),np.array(data_list))











