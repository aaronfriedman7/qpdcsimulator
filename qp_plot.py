#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:56:23 2020

@author: Aaron
"""

import numpy as np
import glob
import matplotlib.pyplot as plt

## F(16) = 987, F(19) = 4181
def Fib_times(tmax,shift=-0.5):
    golden = 0.5*(1.0+np.sqrt(5.0))
    vals = []
    a,b = 0,1
    while b <= tmax:
        vals.append(max(float(b)+shift,golden*(float(a))+shift))
        a,b = b, a+b
    return vals


# pd, wid, K=1.0, J=0.1, B-0.1, rsd, st
# params = [[0.01,0.01],[0.01,0.02],[0.05, 0.01]]
### [pd,wid,Kval,Jval,Bval]
# params = [[0.0,0.01,0.5,0.05,0.02],[0.0,0.01,0.5,0.0,0.0],[0.0,0.1,0.25,0.0,0.0],[0.0,0.01,0.5,0.0,0.2],[0.0,0.02,0.2,0.1,0.1],[0.0,0.02,0.5,0.1,0.1],[0.0,0.02,1.0,0.1,0.1],[0.0,0.05,0.2,0.1,0.1],[0.0,0.05,1.0,0.1,0.1],[0.0,0.1,0.2,0.1,0.1],[0.0,0.1,0.5,0.1,0.1],[0.0,0.1,1.0,0.1,0.1],[0.0,0.05,0.5,0.1,0.1]]
# times = (np.load("./QP/corrs_L4_pd0.05_wid0.01_K1.0_J0.1_B0.1_rsd4_st43.npy"))[:,0]

# params = [[5,0.05,0.0,0.1,0.3,0.0,0.0],[5,0.05,0.0,0.1,0.3,0.01,0.0],[5,0.05,0.0,0.1,0.3,0.05,0.0],[5,0.05,0.0,0.1,0.2,0.0,0.0],[5,0.05,0.0,0.1,0.3,0.1,0.0],[5,0.05,0.0,0.2,0.5,0.0,0.0]]
params = [[7,0.05,0.0,0.0,0.3,0.0,0.0],[7,0.05,0.05,0.0,0.3,0.05,0.05],[7,0.05,0.05,0.0,0.5,0.1,0.05],[7,0.05,0.0,0.1,0.3,0.1,0.05],[7,0.05,0.0,0.1,0.3,0.1,0.1],[7,0.05,0.05,0.1,0.3,0.15,0.1],[7,0.05,0.0,0.1,0.3,0.1,0.15],[7,0.05,0.0,0.1,0.3,0.1,0.2],[7,0.05,0.0,0.1,0.3,0.1,0.25],[7,0.05,0.0,0.1,0.3,0.1,0.3]]
params.extend([[7,0.05,0.02,0.1,0.3,0.1,0.05],[7,0.05,0.05,0.1,0.3,0.1,0.05],[7,0.05,0.02,0.1,0.3,0.1,0.1],[7,0.05,0.05,0.1,0.3,0.1,0.1],[7,0.05,0.05,0.1,0.3,0.1,0.2]])
params.extend([[7,0.05,0.1,0.1,0.3,0.1,0.1],[7,0.05,0.1,0.1,0.3,0.1,0.2],[7,0.05,0.1,0.1,0.3,0.2,0.1],[7,0.05,0.05,0.1,0.3,0.2,0.2],[7,0.05,0.1,0.1,0.3,0.2,0.2]])
params.extend([[5,0.05,0.0,0.1,0.3,0.0,0.0],[5,0.05,0.0,0.1,0.3,0.05,0.0],[5,0.05,0.0,0.1,0.3,0.1,0.0],[5,0.05,0.0,0.1,0.3,0.05,0.05]])
golden = 0.5*(1.0 + np.sqrt(5.0))
# print("Fibonacci times = {0} ... {1}".format(Fibts[:10],Fibts[-1]))

for [L,wid,pd,K1,K2,B,J] in params:
    rsds = []
    sts = 0
    ts = []
    Xedge = []
    Xbulk = []
    Zedge = []
    Zbulk = []
    print("working on phase diff={}, pulse width={}, Kvals={},{}, J={}, B={}".format(pd,wid,K1,K2,J,B))
    flist = glob.glob("./QP/corrs_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
    # print(flist)
    #flist.extend(glob.glob("./QP/corrs1_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd0_st*.npy".format(L,pd,wid,K1,K2,J,B)))
    entries = []
    for fname in flist:
        state = int(fname[fname.find("_st")+3:fname.find(".npy")])
        rsd = int(fname[fname.find("_rsd")+4:fname.find("_st")])
        rsds.append(rsd)
        data = np.load(fname)
        # print(np.shape(data))
        times = data[:,0]
        if ts == []:
            ts.append(list(times))
        else:
            if not list(times) in ts:
                # print("oh fuck, times mismatch")
                ts.append(list(times))
        if times[-1] < 5000:
            continue
        entries.append(len(data[:,0]))
        Xedge.append((data[:,2]/data[:,1]).tolist())
        Xbulk.append((data[:,3]/data[:,1]).tolist())
        Zedge.append((data[:,4]/data[:,1]).tolist())
        Zbulk.append((data[:,5]/data[:,1]).tolist())
        sts += 1
    if entries != []:
        maxlen = min(entries)
    else:
        continue
    ts2 = []
    for tlist in ts:
        ts2.append(tlist[:maxlen])
        print(tlist[maxlen-1])
    print()
    if len(ts2) > 1:
        print()
        std = np.average(np.std(np.array(ts2),axis=0))
        if std != 0:
            print("times list has average stddev = {}".format(std))
            print()
            stdarray = np.std(np.array(ts2),axis=0)
            ind = np.argmax(stdarray)
            timearray = np.array(ts2)
            print(timearray[:,ind])
            print()
            print(stdarray[ind])
            print()
    rsds = list(set(rsds))
    nd = len(rsds)
    if nd <= 1:
        continue
    if entries == []:
        continue
    Xedge = [X[:maxlen] for X in Xedge]
    Xbulk = [X1[:maxlen] for X1 in Xbulk]
    Zedge = [Z[:maxlen] for Z in Zedge]
    Zbulk = [Z1[:maxlen] for Z1 in Zbulk]
    times = times[:maxlen]
    print("Found {0} good files".format(sts))
    try:
        Xcorrs_edge = np.average(np.array(Xedge),axis=0)
        Xcorrs_bulk = np.average(np.array(Xbulk),axis=0)
    except:
        print("X edge data type={0},{1}, shape={2}".format(type(Xedge),type(np.array(Xedge)),np.shape(np.array(Xedge))))
        continue
    try:
        Zcorrs_edge = np.average(np.array(Zedge),axis=0)
        Zcorrs_bulk = np.average(np.array(Zbulk),axis=0)
    except:
        print("Z edge data type={0},{1}, shape={2}".format(type(Zedge),type(np.array(Zedge)),np.shape(np.array(Zedge))))
        continue
    
    plt.figure()
    # plt.suptitle("Regular; L={0} pd={1}, Ks={2},{3} J={4} B={5}, r={6},{7}".format(2*L,pd,K1,K2,J,B,nd,sts))
    plt.suptitle("Edge vs. bulk correlation functions")
    plt.subplot(2,2,1)
    plt.plot(times,Xcorrs_edge,label="edge")
    plt.plot(times,Xcorrs_bulk,label="bulk")
    # plt.legend(loc="best")
    # plt.xlabel("$t$")
    plt.ylabel("$  X(t) X(0)$")
    plt.title("All times")
    
    plt.subplot(2,2,3)
    plt.plot(times,Zcorrs_edge,label="edge")
    plt.plot(times,Zcorrs_bulk,label="bulk")
    # plt.legend(loc="best")
    plt.xlabel("$t$")
    plt.ylabel("$  Z(t) Z(0)$")
    
    # fibtime = np.log(times[1:])/(np.log(golden))
    # plt.subplot(2,2,2)
    # plt.plot(fibtime,Xcorrs_edge[1:],label="edge")
    # plt.plot(fibtime,Xcorrs_bulk[1:],label="bulk")
    # # plt.legend(loc="best")
    # # plt.xlabel("$ {\\rm log}^{\,}_{\\varphi} \, t$")
    # # plt.ylabel("$ X(t) X(0) $")
    
    # plt.subplot(2,2,4)
    # plt.plot(fibtime,Zcorrs_edge[1:],label="edge")
    # plt.plot(fibtime,Zcorrs_bulk[1:],label="bulk")
    # # plt.title("Z(t) Z(0), width = {0} K = {1}".format(wid,Kval))
    # # plt.legend(loc="best")
    # plt.xlabel("$ {\\rm log}^{\,}_{\\varphi} \, t$")
    # # plt.ylabel("$  Z(t) Z(0) $")
    
    # plt.show()
    
    fbinds = []
    Fibts = Fib_times(int(times[-1]),0.0)
    for Fibtime in Fibts:
        inds = np.where(np.abs(times-Fibtime)<0.25)[0]
        fbinds.append(inds[np.argmin(times[inds])])
    fibinds = np.array(fbinds)
    print("compare true = {0}".format(np.array(Fibts)))
    print("with sample = {0}".format(times[fibinds]))
    
    # plt.figure()
    # plt.suptitle("Fibonacci; L={0} pd={1}, Ks={2},{3} J={4} B={5}, r={6},{7}".format(2*L,pd,K1,K2,J,B,nd,sts))
    
    # plt.subplot(2,2,1)
    # plt.plot(times[fibinds],Xcorrs_edge[fibinds],label="edge")
    # plt.plot(times[fibinds],Xcorrs_bulk[fibinds],label="bulk")
    # #plt.legend(loc="best")
    # # plt.xlabel("$t$")
    # plt.ylabel("$  X(t) X(0)$")
    
    
    # plt.subplot(2,2,3)
    # plt.plot(times[fibinds],Zcorrs_edge[fibinds],label="edge")
    # plt.plot(times[fibinds],Zcorrs_bulk[fibinds],label="bulk")
    # #plt.legend(loc="best")
    # plt.xlabel("$F^{\,}_n $")
    # plt.ylabel("$  Z(t) Z(0)$")
    
    fibnums = np.arange(len(fibinds))
    
    plt.subplot(2,2,2)
    plt.plot(fibnums,Xcorrs_edge[fibinds],label="edge")
    plt.plot(fibnums,Xcorrs_bulk[fibinds],label="bulk")
    plt.title("Fibonacci times")
    # plt.legend(loc="best")
    # plt.xlabel("$ {\\rm log}^{\,}_{\\varphi} \, t$")
    # plt.ylabel("$ X(t) X(0) $")
    
    plt.subplot(2,2,4)
    plt.plot(fibnums,Zcorrs_edge[fibinds],label="edge")
    plt.plot(fibnums,Zcorrs_bulk[fibinds],label="bulk")
    # plt.title("Z(t) Z(0), width = {0} K = {1}".format(wid,Kval))
    # plt.legend(loc="best")
    plt.xlabel("$ n $")
    # plt.ylabel("$  Z(t) Z(0) $")
    
    # plt.show()
    # plt.savefig("./Corrplot_fib_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}.pdf".format(2*L,pd,wid,K1,K2,J,B))
    plt.savefig("./Corrplot_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}.pdf".format(2*L,pd,wid,K1,K2,J,B))
    plt.close()
    
