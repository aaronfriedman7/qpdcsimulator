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
def Fib_times(tmax,shift=-0.0):
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
#params = [[7,0.05,0.0,0.1,0.3,0.1,0.05],[7,0.05,0.0,0.1,0.3,0.1,0.1],[7,0.05,0.0,0.1,0.3,0.1,0.15],[7,0.05,0.0,0.1,0.3,0.1,0.2],[7,0.05,0.0,0.1,0.3,0.1,0.25],[7,0.05,0.0,0.1,0.3,0.1,0.3]]
#params.extend([[7,0.05,0.02,0.1,0.3,0.1,0.05],[7,0.05,0.05,0.1,0.3,0.1,0.05],[7,0.05,0.02,0.1,0.3,0.1,0.1],[7,0.05,0.05,0.1,0.3,0.1,0.1],[7,0.05,0.05,0.1,0.3,0.1,0.2]])
#params.extend([[7,0.05,0.1,0.1,0.3,0.1,0.1],[7,0.05,0.1,0.1,0.3,0.1,0.2],[7,0.05,0.1,0.1,0.3,0.2,0.1],[7,0.05,0.05,0.1,0.3,0.2,0.2],[7,0.05,0.1,0.1,0.3,0.2,0.2]])
#params.extend([[5,0.05,0.0,0.1,0.3,0.0,0.0],[5,0.05,0.0,0.1,0.3,0.05,0.0],[5,0.05,0.0,0.1,0.3,0.1,0.0],[5,0.05,0.0,0.1,0.3,0.05,0.05]])
golden = 0.5*(1.0 + np.sqrt(5.0))
# print("Fibonacci times = {0} ... {1}".format(Fibts[:10],Fibts[-1]))

params = [[7,0.05,0.05,0.1,0.3,0.1,0.1]]

for [L,wid,pd,K1,K2,B,J] in params:
    for rsd in range(7):
        print("rsd = {0}".format(rsd))
        sts = 0
        Xedge = []
        Xbulk = []
        Zedge = []
        Zbulk = []
        flist = glob.glob("./QP/corrs_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd{7}_st*.npy".format(L,pd,wid,K1,K2,J,B,rsd))
        # print(flist)
        #flist.extend(glob.glob("./QP/corrs1_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd0_st*.npy".format(L,pd,wid,K1,K2,J,B)))
        entries = []
        for fname in flist:
            state = int(fname[fname.find("_st")+3:fname.find(".npy")])
            data = np.load(fname)
            # print(np.shape(data))
            # print("data lengths = {0}, with shape {1}".format(len(data[:,0]),np.shape(data)))
            # if len(data[:,0]) != 20120:
            #     print("First, last time values are {0},{1}...{2},{3}".format(data[0,0],data[1,0],data[-2,0],data[-1,0]))
            #     continue
            times = data[:,0]
            entries.append(len(data[:,0]))
            Xedge.append((data[:,2]/data[:,1]).tolist())
            Xbulk.append((data[:,3]/data[:,1]).tolist())
            Zedge.append((data[:,4]/data[:,1]).tolist())
            Zbulk.append((data[:,5]/data[:,1]).tolist())
            
            plt.figure()
            plt.suptitle("Check L={0} pd={1}, Ks={2},{3} J={4} B={5}, st={6}".format(2*L,pd,K1,K2,J,B,state))
            
            plt.subplot(2,2,1)
            plt.plot(times,data[:,2]/data[:,1],label="edge")
            plt.plot(times,data[:,3]/data[:,1],label="bulk")
            plt.ylabel("$  X(t) X(0)$")
            
            plt.subplot(2,2,3)
            plt.plot(times,data[:,4]/data[:,1],label="edge")
            plt.plot(times,data[:,5]/data[:,1],label="bulk")
            plt.xlabel("$t$")
            plt.ylabel("$  Z(t) Z(0)$")
            
            fbinds = []
            Fibts = Fib_times(int(times[-1]),0.0)
            for Fibtime in Fibts:
                inds = np.where(np.abs(times-Fibtime)<0.25)[0]
                fbinds.append(inds[np.argmin(times[inds])])
            fibinds = np.array(fbinds)
            fibnums = np.arange(len(fibinds))
            
            plt.subplot(2,2,2)
            plt.plot(fibnums,data[fibinds,2]/data[fibinds,1],label="edge")
            plt.plot(fibnums,data[fibinds,3]/data[fibinds,1],label="bulk")
            
            plt.subplot(2,2,4)
            plt.plot(fibnums,data[fibinds,4]/data[fibinds,1],label="edge")
            plt.plot(fibnums,data[fibinds,5]/data[fibinds,1],label="bulk")
            plt.xlabel("$ n $")
            
            # plt.show()
            plt.savefig("./State_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd{7}_st{8}.pdf".format(2*L,pd,wid,K1,K2,J,B,rsd,state))
            plt.close()
            
            sts += 1
        maxlen = min(entries)
        Xedge = [X[:maxlen] for X in Xedge]
        Xbulk = [X[:maxlen] for X in Xbulk]
        Zedge = [X[:maxlen] for X in Zedge]
        Zbulk = [X[:maxlen] for X in Zbulk]
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
        plt.suptitle("Check L={0} pd={1}, Ks={2},{3} J={4} B={5}, nst={6}".format(2*L,pd,K1,K2,J,B,sts))
        
        plt.subplot(2,2,1)
        plt.plot(times,Xcorrs_edge,label="edge")
        plt.plot(times,Xcorrs_bulk,label="bulk")
        plt.ylabel("$  X(t) X(0)$")
        
        plt.subplot(2,2,3)
        plt.plot(times,Zcorrs_edge,label="edge")
        plt.plot(times,Zcorrs_bulk,label="bulk")
        plt.xlabel("$t$")
        plt.ylabel("$  Z(t) Z(0)$")
        
        fbinds = []
        Fibts = Fib_times(int(times[-1]),0.0)
        for Fibtime in Fibts:
            inds = np.where(np.abs(times-Fibtime)<0.25)[0]
            fbinds.append(inds[np.argmin(times[inds])])
        fibinds = np.array(fbinds)
        fibnums = np.arange(len(fibinds))
        
        plt.subplot(2,2,2)
        plt.plot(fibnums,Xcorrs_edge[fibinds],label="edge")
        plt.plot(fibnums,Xcorrs_bulk[fibinds],label="bulk")
        
        plt.subplot(2,2,4)
        plt.plot(fibnums,Zcorrs_edge[fibinds],label="edge")
        plt.plot(fibnums,Zcorrs_bulk[fibinds],label="bulk")
        plt.xlabel("$ n $")
        
        # plt.show()
        plt.savefig("./Realization_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd{7}.pdf".format(2*L,pd,wid,K1,K2,J,B,rsd))
        plt.close()
        
        
    
