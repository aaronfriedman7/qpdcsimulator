#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:56:23 2020

@author: Aaron
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:24 2020

@author: Aaron
"""


import numpy as np
import glob
import matplotlib.pyplot as plt

## F(16) = 987, F(19) = 4181
def Fib_times(tmax):
    golden = 0.5*(1.0+np.sqrt(5.0))
    vals = []
    a,b = 0,1
    while b <= tmax:
        vals.append(max(float(b)-0.5,golden*(float(a))-0.5))
        a,b = b, a+b
    return vals


cds = ['','0.02','0.025','0.03','0.035','0.04','0.045','0.05','0.055','0.06','0.065','0.07','0.075','0.08','0.1','0.2','0.5']
(L,wid,pd,K1,K2,B,J) = (7,0.05,0.05,0.1,0.3,0.1,0.1)
golden = 0.5*(1.0 + np.sqrt(5.0))
Fibts = Fib_times(200)
print("Fibonacci times = {0} ... {1}".format(Fibts[:10],Fibts[-1]))

for cd in cds:
    counter = 0
    counter1 = 0
    entries = []
    entries1 = []
    Xedge = []
    Xbulk = []
    Zedge = []
    Zbulk = []
    Xedge_cd = []
    Xbulk_cd = []
    Zedge_cd = []
    Zbulk_cd = []
    print("working on phase diff={0}, pulse width={1}, Kvals={2},{3} L={4}".format(pd,wid,K1,K2,2*L))
    flist = glob.glob("./QP/corrs_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
    # print(flist)
    for fname in flist:
        state = int(fname[fname.find("_st")+3:fname.find(".npy")])
        # rsd = int(fname[fname.find("_rsd")+4:fname.find("_st")])
        data = np.load(fname)
        times = data[:,0]
        if times[-1] != 200.0:
            # print(np.where(times==200.0))
            lastind = np.where(times==200.0)[0][0]
            times = times[:lastind]
        else:
            lastind = len(times) 
        entries.append(len(data[:lastind,0]))
        Xedge.append((data[:lastind,2]/data[:lastind,1]).tolist())
        Xbulk.append((data[:lastind,3]/data[:lastind,1]).tolist())
        Zedge.append((data[:lastind,4]/data[:lastind,1]).tolist())
        Zbulk.append((data[:lastind,5]/data[:lastind,1]).tolist())
        counter += 1
    
    flist1 = glob.glob("./QP/corrs_cd"+cd+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
    print(len(flist1))
    for fname1 in flist1:
        state = int(fname1[fname1.find("_st")+3:fname1.find(".npy")])
        # rsd = int(fname1[fname1.find("_rsd")+4:fname1.find("_st")])
        data = np.load(fname1)
        times = data[:,0]
        if times[-1] != 200.0:
            print(np.where(times==200.0))
            lastind = np.where(times==200.0)[0][0]
            times = times[:lastind]
        else:
            lastind = len(times)
        entries1.append(len(data[:lastind,0]))
        Xedge_cd.append((data[:lastind,2]/data[:lastind,1]).tolist())
        Xbulk_cd.append((data[:lastind,3]/data[:lastind,1]).tolist())
        Zedge_cd.append((data[:lastind,4]/data[:lastind,1]).tolist())
        Zbulk_cd.append((data[:lastind,5]/data[:lastind,1]).tolist())
        counter1 += 1
    if counter == 0 or counter1 == 0:
        print(counter,counter1)
        continue
    print("Found {0} good files".format(counter))
    maxlen = min(entries)
    maxlen1 = min(entries1)
    Xedge = [X[:maxlen] for X in Xedge]
    Xbulk = [X[:maxlen] for X in Xbulk]
    Zedge = [X[:maxlen] for X in Zedge]
    Zbulk = [X[:maxlen] for X in Zbulk]
    Xedge_cd = [X[:maxlen1] for X in Xedge_cd]
    Xbulk_cd = [X[:maxlen1] for X in Xbulk_cd]
    Zedge_cd = [X[:maxlen1] for X in Zedge_cd]
    Zbulk_cd = [X[:maxlen1] for X in Zbulk_cd]
    try:
        XCE = np.average(np.array(Xedge),axis=0)
        XCB = np.average(np.array(Xbulk),axis=0)
        XCEc = np.average(np.array(Xedge_cd),axis=0)
        XCBc = np.average(np.array(Xbulk_cd),axis=0)
    except:
        print("X edge data type={0},{1}, shape={2}".format(type(Xedge),type(np.array(Xedge)),np.shape(np.array(Xedge))))
        continue
    try:
        ZCE = np.average(np.array(Zedge),axis=0)
        ZCB = np.average(np.array(Zbulk),axis=0)
        ZCEc = np.average(np.array(Zedge_cd),axis=0)
        ZCBc = np.average(np.array(Zbulk_cd),axis=0)
    except:
        print("Z edge data type={0},{1}, shape={2}".format(type(Zedge),type(np.array(Zedge)),np.shape(np.array(Zedge))))
        continue
    
    plt.figure()
    plt.suptitle("cd="+cd+"; pd={}, width={}, Ks={},{} J={} B={}".format(pd,wid,K1,K2,J,B))
    plt.subplot(2,2,1)
    plt.plot(times[:maxlen],XCE,label="edge")
    plt.plot(times[:maxlen],XCB,label="bulk")
    # plt.legend(loc="best")
    # plt.xlabel("$t$")
    plt.ylabel("$  X(t) X(0)$")
    
    plt.subplot(2,2,3)
    plt.plot(times[:maxlen1],XCEc,label="edge")
    plt.plot(times[:maxlen1],XCBc,label="bulk")
    # plt.legend(loc="best")
    # plt.xlabel("$t$")
    plt.ylabel("CD $  X(t) X(0)$")
    
    plt.subplot(2,2,2)
    plt.plot(times[:maxlen],ZCE,label="edge")
    plt.plot(times[:maxlen],ZCB,label="bulk")
    # plt.legend(loc="best")
    plt.xlabel("$t$")
    plt.ylabel("$  Z(t) Z(0)$")
    
    plt.subplot(2,2,4)
    plt.plot(times[:maxlen1],ZCEc,label="edge")
    plt.plot(times[:maxlen1],ZCBc,label="bulk")
    # plt.legend(loc="best")
    plt.xlabel("$t$")
    plt.ylabel("CD $  Z(t) Z(0)$")
    
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
    plt.savefig("./CD_"+cd+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}.pdf".format(2*L,pd,wid,K1,K2,J,B))
    plt.close()
    
    # fbinds = []
    # for Fibtime in Fibts:
    #     inds = np.where(times >= Fibtime)[0]
    #     fbinds.append(inds[np.argmin(times[inds])])
    # fibinds = np.array(fbinds)
    # print("compare true = {0}".format(np.array(Fibts)))
    # print("with sample = {0}".format(times[fibinds]))
    
    # plt.figure()
    # plt.suptitle("Fibonacci; L={0}, width={1}, Ks={2},{3} J={4} B={5}".format(2*L,wid,K1,K2,J,B))
    
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
    
    # fibnums = np.arange(len(fibinds))
    
    # plt.subplot(2,2,2)
    # plt.plot(fibnums,Xcorrs_edge[fibinds],label="edge")
    # plt.plot(fibnums,Xcorrs_bulk[fibinds],label="bulk")
    # # plt.legend(loc="best")
    # # plt.xlabel("$ {\\rm log}^{\,}_{\\varphi} \, t$")
    # # plt.ylabel("$ X(t) X(0) $")
    
    # plt.subplot(2,2,4)
    # plt.plot(fibnums,Zcorrs_edge[fibinds],label="edge")
    # plt.plot(fibnums,Zcorrs_bulk[fibinds],label="bulk")
    # # plt.title("Z(t) Z(0), width = {0} K = {1}".format(wid,Kval))
    # # plt.legend(loc="best")
    # plt.xlabel("$ n $")
    # # plt.ylabel("$  Z(t) Z(0) $")
    
    # # plt.show()
    # plt.savefig("./Corrplot_fib_L{0}_wid{1}_K1_{2}_K2_{3}_J{4}_B{5}.pdf".format(2*L,wid,K1,K2,J,B))
    # plt.close()
    
