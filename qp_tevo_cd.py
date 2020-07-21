#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:55:36 2020

@author: Aaron
"""

import glob
import argparse
import os
# import sys

# need to set MKL num threads? i.e. to Ncore ... note this has to be before numpy is imported. Will probably be set as part of submission?
import numpy as np
import scipy.integrate as sint
import scipy.sparse as ss

# import matplotlib.pyplot as plt
# os.environ["MKL_NUM_THREADS"] = "1" #### for non MPI jobs.

mainpath = "07375/ajfriedm/QP_tevo/"
workpath = "/scratch/"+mainpath
os.chdir(workpath)
# workpath = "./"

# In general, all file loading try statements should only accept particular exceptions. But these have been tested.

# convenient file label string
def fhead(params,rsd,st):
    return "L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd{7}_st{8}".format(params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B,rsd,st)

def jobhead(params,rsd):
    return "L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd{7}".format(params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B,rsd)

# the 0.5 shift should compensate for phase shift (for x pulses), should we also include half-odd multiples of golden?
def timeslist(tmax,every):
    golden = 0.5*(1.0+np.sqrt(5.0))
    vals = []
    a,b = 0,1
    while a <= tmax:
        newval = max(float(b)-0.5,golden*(float(a)-0.5))
        # MODFIY BELOW to be more useful?
        vals.extend(list(set([newval-0.5,newval-0.25,newval,newval+0.25,newval+0.5,np.ceil(newval)])))
        a,b = b, a+b
    vals = sorted(vals,key=float)
    while vals[-1] > tmax:
        vals.remove(vals[-1])
    while vals[0] < 0.0:
        vals.remove(vals[0])
    vals.extend([every*foo for foo in range(int(np.ceil(float(tmax)/every))+2)])
    return sorted(list(set(vals)),key=float)

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
    try:
        H = ss.load_npz(workpath+"Hstat_L{0}_K1_{1}_K2_{2}_J{3}_B{4}_rsd{5}.npz".format(Lcell,params.K1,params.K2,params.J,params.B,rsd))
    except:
        Size = 4**Lcell
        H = ss.dok_matrix((Size,Size),dtype=complex) ## or 'c'
        try:
            Kvalues = np.load(workpath+"Kvals_L{0}_K1_{1}_K2_{2}_rsd{3}.npy".format(Lcell,params.K1,params.K2,rsd))
        except:
            Kvalues = np.random.uniform(params.K1,params.K2,(2,Lcell-1))*np.random.choice(np.array([-1.0,1.0]),(2,Lcell-1))
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
            for cell in range(Lcell-1):
                if scfg[2*cell] == scfg[2*cell + 1]:
                    H[state,state] += Jvalues[1,cell]
                else:
                    H[state,state] -= Jvalues[1,cell]
                H[state^( 2**(2*Lcell-2*cell-1) + 2**(2*Lcell-2*cell-2) ),state] += Jvalues[0,cell]
                if scfg[2*cell+1] == scfg[2*cell + 2]:
                    H[state,state] += Kvalues[1,cell]
                else:
                    H[state,state] -= Kvalues[1,cell]
                H[state^(2**(2*Lcell-2*cell-2)+2**(2*Lcell-2*cell-3)),state] += Kvalues[0,cell]
            # final cell
            if scfg[2*Lcell-2] == scfg[2*Lcell - 1]:
                H[state,state] += Jvalues[1,Lcell-1]
            else:
                H[state,state] -= Jvalues[1,Lcell-1]
            xxstate = state^3
            H[xxstate,state] += Jvalues[0,Lcell-1]
        H = ss.csr_matrix(H)
        ss.save_npz(workpath+"Hstat_L{0}_K1_{1}_K2_{2}_J{3}_B{4}_rsd{5}.npz".format(Lcell,params.K1,params.K2,params.J,params.B,rsd),H)
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
def evo_state_RHS(t,y,params,Xp,Zp,Hs,psimem):
    psimem = -1j*Hs.dot(y)
    psimem -= 1j*(1.0-params.pd)*pulse_func(t,2.0*np.pi,params.wid)*Xp.dot(y)
    psimem -= 1j*(1.0-params.pd)*pulse_func(t,4.0*np.pi/(1.0+np.sqrt(5.0)),params.wid)*(Zp*y)
    return psimem


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)
# Ncore = comm.Get_size()

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='xxz random unitary spectral form factor', prog='main', usage='%(prog)s [options]')
    parser.add_argument("-L", "--L", help="number of UNIT CELLS", default=7,type=int, nargs='?')
    parser.add_argument("-B", "--B", help="local field dist. width", default=0.0,type=float, nargs='?')
    parser.add_argument("-rstart", "--rstart", help="disorder realization offset", default=0,type=int, nargs='?')
    parser.add_argument("-K2", "--K2", help="intercell coupling dist. width", default=0.3,type=float, nargs='?')
    parser.add_argument("-K1", "--K1", help="intercell coupling dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-J", "--J", help="local random coupling dist. width", default=0.0,type=float, nargs='?')
    parser.add_argument("-pd", "--pd", help="phase difference, same as Brayden", default=0.0,type=float, nargs='?')
    parser.add_argument("-wid", "--wid", help="std dev for pulses (width)", default=0.05,type=float, nargs='?')
    parser.add_argument("-tf", "--tf", help="stop time", default=250.0,type=int, nargs='?')
    parser.add_argument("-maxint", "--maxint", help="maximum time interval", default=100.0,type=float, nargs='?')
    parser.add_argument("-ev", "--ev", help="how often to record data", default=0.5, type=float, nargs='?')
    parser.add_argument("-cd", "--cd", help="counter-drive factor", default=1.0,type=float, nargs='?')
    
    # load arguments
    params=parser.parse_args()
    #ti = float(params.maxint)/float(params.div) ## just using ti as a placeholder

    if rank == 0:
        if params.ev == 0.5:
            tname = workpath+"tarray_tf{0}.npy".format(params.tf)
        else:
            tname = workpath+"tarray_tf{0}_ev{1}.npy".format(params.tf,params.ev)
        try:
            tarray = np.load(tname)
        except (FileNotFoundError, IOError, EOFError) as err:
            print("{0} -- could not load times to do, making now".format(err))
            tarray = np.array(timeslist(params.tf,params.ev))
            np.save(tname,tarray)
        numtimes = len(tarray)
    else:
        numtimes = None

    numtimes = comm.bcast(numtimes,root=0)
    if rank != 0:
        tarray = np.empty(numtimes,dtype=float)
    comm.Bcast(tarray,root=0)
    
    if rank == 0:
        Zpulse = pulse_ham_z(params) - params.cd*(Zop(params,0) + Zop(params,2*params.L-1))
    else:
        Zpulse = np.empty(4**params.L,dtype=float)
    comm.Bcast(Zpulse,root=0)
        
    if rank == 0:
        Xpulse = pulse_ham_x(params) - params.cd*(Xop(params,0) + Xop(params,2*params.L-1))
    else:
        Xpulse = None
    Xpulse = comm.bcast(Xpulse,root=0)
    print("pulses received")
    print()
    
    tasknum = int(np.floor(rank/3))
    rsd = tasknum + params.rstart
    print("Rank is {}, task number is {}, rsd is {}".format(rank,tasknum,rsd))
    print()
    
    st = None
    Hstatic = None
    
    if rank == 3*tasknum:
        data_names = workpath+"corrs_r_cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd{}_st*.npy".format(params.cd,params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B,rsd)
        flist = glob.glob(data_names)
        if len(flist) > 0:
            fname = flist[0]
            st = int(fname[fname.find("_st")+3:fname.find(".npy")])
        else:
            np.random.seed(rsd)
            st = np.random.choice(range(5,4**(params.L)-5))
        for ind in range(3*tasknum+1,3*tasknum+3):
            comm.send(st,dest=ind)
    else:
        st = comm.recv(source=3*tasknum)
    
    print("Doing state {}".format(st))
    
    if rank == 3*tasknum:
        Hstatic = static_ham(params,rsd)
        for ind in range(3*tasknum+1,3*tasknum+3):
            comm.send(Hstatic,dest=ind)
    else:
        Hstatic = comm.recv(Hstatic,source=3*tasknum)
    print("Hamiltonian received for rank {}".format(rank))
    
    data_name = workpath+"corrs_r_cd{}_{}.npy".format(params.cd,fhead(params,rsd,st))
    
    if rank%3 == 0:
        # print("working on {0}th state with index {1} == {2}".format(tasknum,st,spinstr(st,2*params.L)))
        try:
            ts = ((np.load(data_name))[:,0]).tolist()
            ### find save state file for latest time within collected values (preferably at end)
            ti = max([float(fname[fname.find("_time")+5:fname.find(".npy")]) for fname in glob.iglob(workpath+"psi_cd{}_{}_time*.npy".format(params.cd,fhead(params,rsd,st))) if float(fname[fname.find("_time")+5:fname.find(".npy")]) <= max(ts)])
            start_data = np.load(workpath+"psi_cd{}_{}_time{}.npy".format(params.cd,fhead(params,rsd,st),int(ti)))
            newL = ts.index(ti)
            ts = ts[newL:]
            data_list = ((np.load(data_name))[:newL+1,:]).tolist()
        except (FileNotFoundError, IOError, EOFError, ValueError) as err:
            print("FNF {0} for params={1}".format(err,fhead(params,rsd,st)))
            print("starting from time 0")
            ti = 0.0
            ts = []
            data_list = []
            start_data = np.zeros((3,4**params.L),dtype=complex)
            start_data[0,st] = 1.0
            start_data[1,(st)^(2**(2*params.L-1))] = 1.0 # X edge
            start_data[2,(st)^(2**(params.L))] = 1.0 ## Xbulk,  2**(2L - 1 - (L-1)) for site j = L-1 (left side of halfway mark)
    
        # initial Z spin values, only root needs these (also data list and ts)
        Z0s = np.array([2*int((spinstr(st,2*params.L))[j])-1 for j in range(2*params.L)])
        
        #### start time evolution:
        num_runs = int(np.ceil((params.tf-ti)/float(params.maxint)))
        comm.send(num_runs,dest=3*tasknum+1)
        comm.send(num_runs,dest=3*tasknum+2)
    elif rank%3 == 1:
        num_runs = comm.recv(source=3*tasknum)
    elif rank % 3 == 2:
        num_runs = comm.recv(source=3*tasknum)
        
        
    
    for run in range(num_runs):
        if rank%3 == 0:
            tstart = ti+float(params.maxint*run)
            tstop = ti+float(params.maxint*(run+1))
            ## generate list of times to evaluate the unitary.
            teval = tarray[tarray>=tstart]
            teval = teval[teval<=tstop] 
            teval = np.array(sorted(list(set(teval.tolist())),key=float))
            # print("run {0} from t={1} to {2}, times to check = {3}".format(run,tstart,tstop,teval))
            numts = len(teval)
            data = np.zeros((numts,4+2*params.L))
        
            if tstart > 0.0:
                start_data = np.load(workpath+"psi_cd{}_{}_time{}.npy".format(params.cd,fhead(params,rsd,st),int(tstart)))
            
            init_reg = start_data[0,:]
            
            init_edge = start_data[1,:]
            comm.send(numts,dest=3*tasknum + 1)
            comm.Send(teval,dest=3*tasknum + 1)
            comm.Send(init_edge,dest=3*tasknum + 1)
            
            init_bulk = start_data[2,:]
            comm.send(numts,dest=3*tasknum + 2)
            comm.Send(teval,dest=3*tasknum + 2)
            comm.Send(init_bulk,dest=3*tasknum + 2)
            
            #### APPARENTLY Hstatic not defined
            psimem = np.zeros(4**params.L,dtype=complex)
            sol_reg = sint.solve_ivp(evo_state_RHS,(teval[0],teval[-1]),init_reg,t_eval=teval,args=(params,Xpulse,Zpulse,Hstatic,psimem),max_step=0.5*params.wid,rtol=1e-8,atol=1e-10)
            # times0 = sol_reg.t
            psis0 = sol_reg.y
        
        elif rank%3 == 1:
            numts = comm.recv(source=3*tasknum)
            teval = np.empty(numts,dtype=float)
            init_edge = np.empty(4**params.L,dtype=complex)
            edgemem = np.zeros(4**params.L,dtype=complex)
            comm.Recv(teval,source=3*tasknum)
            comm.Recv(init_edge,source=3*tasknum)
            sol_edge = sint.solve_ivp(evo_state_RHS,(teval[0],teval[-1]),init_edge,t_eval=teval,args=(params,Xpulse,Zpulse,Hstatic,edgemem),max_step=0.5*params.wid,rtol=1e-8,atol=1e-10)
            # times1 = sol_edge.t
            psis1 = sol_edge.y
            comm.Send(psis1,dest=3*tasknum)
            
        
        elif rank%3 == 2:
            numts = comm.recv(source=3*tasknum)
            teval = np.empty(numts,dtype=float)
            init_bulk = np.empty(4**params.L,dtype=complex)
            bulkmem = np.zeros(4**params.L,dtype=complex)
            comm.Recv(teval,source=3*tasknum)
            comm.Recv(init_bulk,source=3*tasknum)
            sol_bulk = sint.solve_ivp(evo_state_RHS,(teval[0],teval[-1]),init_bulk,t_eval=teval,args=(params,Xpulse,Zpulse,Hstatic,bulkmem),max_step=0.5*params.wid,rtol=1e-8,atol=1e-10)
            # times2 = sol_bulk.t
            psis2 = sol_bulk.y
            comm.Send(psis2,dest=3*tasknum)
        
        
        if rank%3 == 0:
            psis1 = np.empty((4**params.L,numts),dtype=complex)
            psis2 = psis1.copy()
            comm.Recv(psis1,source=3*tasknum + 1)
            comm.Recv(psis2,source=3*tasknum + 2)
            
            ### store final values for next run
            start_data[0,:] = psis0[:,-1]
            start_data[1,:] = psis1[:,-1]
            start_data[2,:] = psis2[:,-1]
            
            ### calculate XX,ZZ correlation stuff
            for check in range(numts):
                data[check,0] = teval[check]
                data[check,1] = np.vdot(psis0[:,check],psis0[:,check]).real
                Zops = Zall(params)
                for j in range(2*params.L):
                    Zop = Zops[j,:]
                    psimem = Zop*psis0[:,check]
                    data[check,4+j] = (Z0s[j]*np.vdot(psis0[:,check],psimem)).real

                psimem = (Xop(params,0)).dot(psis1[:,check])
                data[check,2] = (np.vdot(psis0[:,check],psimem)).real
            
                psimem = (Xop(params,params.L-1)).dot(psis2[:,check])
                data[check,3] = (np.vdot(psis0[:,check],psimem)).real
                
            data_list.extend(data.tolist())
            np.save(workpath+"psi_cd{}_{}_time{}.npy".format(params.cd,fhead(params,rsd,st),int(tstop)),start_data)
            np.save(data_name,np.array(data_list))
    if rank%3 == 0:
        for tval in range(0,params.tf,int(np.ceil(params.maxint))):
            try:
                os.system("rm "+workpath+"psi_{0}_time{1}".format(fhead(params,rsd,st),tval))
            except:
                print("could not remove file for time {}".format(tval))
        
        
    ## at end:
    # np.save(workpath+"corrs_{0}.npy".format(fhead(params,rsd,st)),np.array(data_list))











