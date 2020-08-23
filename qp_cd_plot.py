#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:56:23 2020

@author: Aaron
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.signal as sig
# from matplotlib import cm

## F(16) = 987, F(19) = 4181
def Fib_times(tmax,shift=-0.5):
    golden = 0.5*(1.0+np.sqrt(5.0))
    vals = []
    a,b = 0,1
    while b <= tmax:
        vals.append(max(float(b)+shift,golden*(float(a))+shift))
        a,b = b, a+b
    return vals

time1 = 150.0
maxax = 200+0.25*(200.0-time1)

params = [[7,0.05,0.05,0.1,0.3,0.1,0.1]]

# cds = [0.0,0.05,0.1,0.15,0.25,0.5,0.75,1.0]
cds = [0.0,0.05,0.1,0.15,0.25,0.5,0.75,1.0]

colors =  plt.cm.get_cmap('gnuplot',len(cds)+1)

realizations = [10,20,40,100,160,200,400]
golden = 0.5*(1.0 + np.sqrt(5.0))
# print("Fibonacci times = {0} ... {1}".format(Fibts[:10],Fibts[-1]))


### regular
for [L,wid,pd,K1,K2,B,J] in params:
    for ndis in realizations:
        plt.figure()
        plt.suptitle("CD Correlations, {} realizations".format(ndis))
        for foo,cd in enumerate(cds):
            print()
            print("counter drive strength = ",cd)
            print()
            done = 0
            ts = []
            Xedge = []
            Xbulk = []
            Zedge = []
            Zbulk = []
            # if cd == 0.0:
                # flist = glob.glob("./QP/corrs_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            flist = glob.glob("./QP/corrs_r_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            flist += glob.glob("./QP/corrs_r1_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            # print(flist)
            #flist.extend(glob.glob("./QP/corrs1_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd0_st*.npy".format(L,pd,wid,K1,K2,J,B)))
            entries = []
            for fname in flist:
                if done > ndis:
                    continue
                state = int(fname[fname.find("_st")+3:fname.find(".npy")])
                rsd = int(fname[fname.find("_rsd")+4:fname.find("_st")])
                data = np.load(fname)
                # print(np.shape(data))
                times = data[:,0]
                if times[-1] < 200.0:
                    continue
                done +=1
                firstind = np.where(times==time1)[0][-1]
                lastind = np.where(times==200.0)[0][0]
                data = data[firstind:lastind,:]
                times = times[firstind:lastind]
                ts.append(list(times))
                entries.append(len(times))
                Xedge.append((data[:,2]/data[:,1]).tolist())
                Xbulk.append((data[:,3]/data[:,1]).tolist())
                Zedge.append((data[:,4]/data[:,1]).tolist())
                if len(data[0,:]) > 6:
                    Zbulk.append((data[:,10]/data[:,1]).tolist())
                else:
                    Zbulk.append((data[:,5]/data[:,1]).tolist())
            if entries == []:
                continue
            if done < ndis:
                print("only found {} files".format(done))
                continue
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
        
        # plt.suptitle("Regular; L={0} pd={1}, Ks={2},{3} J={4} B={5}, r={6},{7}".format(2*L,pd,K1,K2,J,B,nd,sts))
            
            xemax = sig.find_peaks(Xcorrs_edge,height=0.)[0]
            print(type(xemax),np.shape(xemax))
            # xemin = sig.find_peaks(-Xcorrs_edge,height=0.1)[0]
            xbmax = sig.find_peaks(Xcorrs_bulk,height=0.05)[0]
            # xbmin = sig.find_peaks(-Xcorrs_bulk,height=0.05)[0]
            
            plt.subplot(2,1,1)
            plt.plot(times[xemax],Xcorrs_edge[xemax],color=colors(foo),linestyle='--')
            # plt.plot(times[xemin],Xcorrs_edge[xemin],color=colors(foo),linestyle='--')
            plt.plot(times[xbmax],Xcorrs_bulk[xbmax],color=colors(foo),linestyle='-',label=str(cd))
            # plt.plot(times[xbmin],Xcorrs_bulk[xbmin],color=colors(foo),linestyle='-')
            plt.legend(loc="center right",prop={'size': 10})
            # plt.xlabel("$t$")
            plt.xlim([time1,maxax])
            plt.ylim([0,1])
            plt.ylabel("$  X(t) X(0)$")
            # plt.title("All times")
            
            
            zemax = sig.find_peaks(Zcorrs_edge,height=0.05)[0]
            # zemin = sig.find_peaks(-Zcorrs_edge,height=0.1)[0]
            zbmax = sig.find_peaks(Zcorrs_bulk,height=0.05)[0]
            # zbmin = sig.find_peaks(-Zcorrs_bulk,height=0.05)[0]
            
            plt.subplot(2,1,2)
            plt.plot(times[zemax],Zcorrs_edge[zemax],color=colors(foo),linestyle='--')
            # plt.plot(times[zemin],Zcorrs_edge[zemin],color=colors(foo),linestyle='--')
            plt.plot(times[zbmax],Zcorrs_bulk[zbmax],color=colors(foo),linestyle='-',label=str(cd))
            # plt.plot(times[zbmin],Zcorrs_bulk[zbmin],color=colors(foo),linestyle='-')
            # plt.legend(loc="center right")
            plt.xlim([time1,maxax])
            plt.ylim([0,1])
            plt.xlabel("$t$")
            plt.ylabel("$  Z(t) Z(0)$")
        
        
        
        # plt.show()
        # plt.savefig("./Corrplot_fib_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}.pdf".format(2*L,pd,wid,K1,K2,J,B))
        plt.savefig("./CD_Plot_Window_ndis{}.pdf".format(ndis))
        plt.close()







### abs first
for [L,wid,pd,K1,K2,B,J] in params:
    for ndis in realizations:
        plt.figure()
        plt.suptitle("CD Correlations, {} realizations".format(ndis))
        for foo,cd in enumerate(cds):
            print()
            print("counter drive strength = ",cd)
            print()
            done = 0
            ts = []
            Xedge = []
            Xbulk = []
            Zedge = []
            Zbulk = []
            # if cd == 0.0:
                # flist = glob.glob("./QP/corrs_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            flist = glob.glob("./QP/corrs_r_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            flist += glob.glob("./QP/corrs_r1_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            # print(flist)
            #flist.extend(glob.glob("./QP/corrs1_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd0_st*.npy".format(L,pd,wid,K1,K2,J,B)))
            entries = []
            for fname in flist:
                if done > ndis:
                    continue
                state = int(fname[fname.find("_st")+3:fname.find(".npy")])
                rsd = int(fname[fname.find("_rsd")+4:fname.find("_st")])
                data = np.load(fname)
                # print(np.shape(data))
                times = data[:,0]
                if times[-1] < 200.0:
                    continue
                done +=1
                firstind = np.where(times==time1)[0][-1]
                lastind = np.where(times==200.0)[0][0]
                data = np.abs(data[firstind:lastind,:])
                times = times[firstind:lastind]
                ts.append(list(times))
                entries.append(len(times))
                Xedge.append((data[:,2]/data[:,1]).tolist())
                Xbulk.append((data[:,3]/data[:,1]).tolist())
                Zedge.append((data[:,4]/data[:,1]).tolist())
                if len(data[0,:]) > 6:
                    Zbulk.append((data[:,10]/data[:,1]).tolist())
                else:
                    Zbulk.append((data[:,5]/data[:,1]).tolist())
            if entries == []:
                continue
            if done < ndis:
                print("only found {} files".format(done))
                continue
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
        
        # plt.suptitle("Regular; L={0} pd={1}, Ks={2},{3} J={4} B={5}, r={6},{7}".format(2*L,pd,K1,K2,J,B,nd,sts))
            
            xemax = sig.find_peaks(Xcorrs_edge,height=0.)[0]
            print(type(xemax),np.shape(xemax))
            # xemin = sig.find_peaks(-Xcorrs_edge,height=0.1)[0]
            xbmax = sig.find_peaks(Xcorrs_bulk,height=0.05)[0]
            # xbmin = sig.find_peaks(-Xcorrs_bulk,height=0.05)[0]
            
            plt.subplot(2,1,1)
            plt.plot(times[xemax],Xcorrs_edge[xemax],color=colors(foo),linestyle='--')
            # plt.plot(times[xemin],Xcorrs_edge[xemin],color=colors(foo),linestyle='--')
            plt.plot(times[xbmax],Xcorrs_bulk[xbmax],color=colors(foo),linestyle='-',label=str(cd))
            # plt.plot(times[xbmin],Xcorrs_bulk[xbmin],color=colors(foo),linestyle='-')
            plt.legend(loc="center right",prop={'size': 10})
            # plt.xlabel("$t$")
            plt.xlim([time1,maxax])
            plt.ylim([0,1])
            plt.ylabel("$  X(t) X(0)$")
            # plt.title("All times")
            
            
            zemax = sig.find_peaks(Zcorrs_edge,height=0.05)[0]
            # zemin = sig.find_peaks(-Zcorrs_edge,height=0.1)[0]
            zbmax = sig.find_peaks(Zcorrs_bulk,height=0.05)[0]
            # zbmin = sig.find_peaks(-Zcorrs_bulk,height=0.05)[0]
            
            plt.subplot(2,1,2)
            plt.plot(times[zemax],Zcorrs_edge[zemax],color=colors(foo),linestyle='--')
            # plt.plot(times[zemin],Zcorrs_edge[zemin],color=colors(foo),linestyle='--')
            plt.plot(times[zbmax],Zcorrs_bulk[zbmax],color=colors(foo),linestyle='-',label=str(cd))
            # plt.plot(times[zbmin],Zcorrs_bulk[zbmin],color=colors(foo),linestyle='-')
            # plt.legend(loc="center right")
            plt.xlim([time1,maxax])
            plt.ylim([0,1])
            plt.xlabel("$t$")
            plt.ylabel("$  Z(t) Z(0)$")
        
        
        
        # plt.show()
        # plt.savefig("./Corrplot_fib_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}.pdf".format(2*L,pd,wid,K1,K2,J,B))
        plt.savefig("./CD_Plot_Abs_ndis{}.pdf".format(ndis))
        plt.close()





mindis = 10
maxdis = 500






## by site Z correlations
sitemax = 100.0
# mindis = 20
sitemaxax = 125.0


for [L,wid,pd,K1,K2,B,J] in params:
    for site in range(14):
        plt.figure()
        plt.title("$Z(t)Z(0)$ correlations v. CD, site={}".format(site+1))
        for foo1,cd in enumerate(cds):
            #print()
            #print("counter drive strength = ",cd)
            print()
            done = 0
            ts = []
            Zdata = []
            flist = glob.glob("./QP/corrs_r_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            flist += glob.glob("./QP/corrs_r1_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            # print(flist)
            #flist.extend(glob.glob("./QP/corrs1_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd0_st*.npy".format(L,pd,wid,K1,K2,J,B)))
            entries = []
            for fname in flist:
                if done >= maxdis:
                    break
                state = int(fname[fname.find("_st")+3:fname.find(".npy")])
                rsd = int(fname[fname.find("_rsd")+4:fname.find("_st")])
                data = np.load(fname)
                if len(data[0,:]) < 10:
                    # print("wrong data type")
                    continue
                # print(np.shape(data))
                times = data[:,0]
                if times[-1] < sitemax:
                    #  print("not enough times")
                    continue
                done +=1
                # firstind = np.where(times==time1)[0][-1]
                firstind = 0
                lastind = np.where(times==sitemax)[0][0]
                data = data[firstind:lastind,:]
                times = times[firstind:lastind]
                ts.append(list(times))
                entries.append(len(times))
                Zdata.append((data[:,4+site]/data[:,1]).tolist())
            # if done < mindis:
                # print("CD strength {}, sites, only found {} files".format(cd,done))
                # continue
            if done < mindis:
                continue
            if entries == []:
                continue
            try:
                Zcorrs = np.average(np.array(Zdata),axis=0)
            except:
                print("Z site data type={0},{1}, shape={2}".format(type(Zedge),type(np.array(Zdata)),np.shape(np.array(Zdata))))
                continue
            
            zmax = sig.find_peaks(Zcorrs,height=0.05)[0]
            # zemin = sig.find_peaks(-Zcorrs_edge,height=0.1)[0]
            
            plt.plot(times[zmax],Zcorrs[zmax],color=colors(foo1),linestyle='--',label=str(cd)+", "+str(done))
            # plt.plot(times[zemin],Zcorrs_edge[zemin],color=colors(foo),linestyle='--')
        plt.legend(loc="center right",prop={'size': 10})
        plt.xlim([0,sitemaxax])
        plt.ylim([0,1])
        plt.xlabel("$t$")
        plt.ylabel("$  Z_{} (t) Z_{} (0)$".format(site+1,site+1))
    

        plt.savefig("./Corrs_CD_Z_site{}.pdf".format(site+1))
        plt.close()
        
        
# abs
for [L,wid,pd,K1,K2,B,J] in params:
    for site in range(14):
        plt.figure()
        plt.title("$Z(t)Z(0)$ correlations v. CD, site={}".format(site+1))
        for foo1,cd in enumerate(cds):
            #print()
            #print("counter drive strength = ",cd)
            print()
            done = 0
            ts = []
            Zdata = []
            flist = glob.glob("./QP/corrs_r_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            flist += glob.glob("./QP/corrs_r1_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            # print(flist)
            #flist.extend(glob.glob("./QP/corrs1_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd0_st*.npy".format(L,pd,wid,K1,K2,J,B)))
            entries = []
            for fname in flist:
                if done >= maxdis:
                    break
                state = int(fname[fname.find("_st")+3:fname.find(".npy")])
                rsd = int(fname[fname.find("_rsd")+4:fname.find("_st")])
                data = np.abs(np.load(fname))
                if len(data[0,:]) < 10:
                    # print("wrong data type")
                    continue
                # print(np.shape(data))
                times = data[:,0]
                if times[-1] < sitemax:
                    #  print("not enough times")
                    continue
                done +=1
                # firstind = np.where(times==time1)[0][-1]
                firstind = 0
                lastind = np.where(times==sitemax)[0][0]
                data = data[firstind:lastind,:]
                times = times[firstind:lastind]
                ts.append(list(times))
                entries.append(len(times))
                Zdata.append((data[:,4+site]/data[:,1]).tolist())
            # if done < mindis:
                # print("CD strength {}, sites, only found {} files".format(cd,done))
                # continue
            if done < mindis:
                continue
            if entries == []:
                continue
            try:
                Zcorrs = np.average(np.array(Zdata),axis=0)
            except:
                print("Z site data type={0},{1}, shape={2}".format(type(Zedge),type(np.array(Zdata)),np.shape(np.array(Zdata))))
                continue
            
            zmax = sig.find_peaks(Zcorrs,height=0.05)[0]
            # zemin = sig.find_peaks(-Zcorrs_edge,height=0.1)[0]
            
            plt.plot(times[zmax],Zcorrs[zmax],color=colors(foo1),linestyle='--',label=str(cd)+", "+str(done))
            # plt.plot(times[zemin],Zcorrs_edge[zemin],color=colors(foo),linestyle='--')
        plt.legend(loc="center right",prop={'size': 10})
        plt.xlim([0,sitemaxax])
        plt.ylim([0,1])
        plt.xlabel("$t$")
        plt.ylabel("$  Z_{} (t) Z_{} (0)$".format(site+1,site+1))
    

        plt.savefig("./Corrs_CD_Z_site{}_abs.pdf".format(site+1))
        plt.close()




### ZZ correlators

for [L,wid,pd,K1,K2,B,J] in params:
    for site in range(13):
        plt.figure()
        plt.title("$Z_{} (t) Z_{} (t) Z_{} (0) Z_{} (0)$  v. CD".format(site+1,site+2,site+1,site+2))
        for foo1,cd in enumerate(cds):
            #print()
            #print("counter drive strength = ",cd)
            print()
            done = 0
            ts = []
            Zdata = []
            # flist = glob.glob("./QP/corrs_r_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            flist = glob.glob("./QP/corrs_r1_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            # print(flist)
            #flist.extend(glob.glob("./QP/corrs1_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd0_st*.npy".format(L,pd,wid,K1,K2,J,B)))
            entries = []
            for fname in flist:
                if done >= maxdis:
                    break
                state = int(fname[fname.find("_st")+3:fname.find(".npy")])
                rsd = int(fname[fname.find("_rsd")+4:fname.find("_st")])
                data = np.load(fname)
                if len(data[0,:]) < 16:
                    # print("wrong data type")
                    continue
                # print(np.shape(data))
                times = data[:,0]
                if times[-1] < sitemax:
                    #  print("not enough times")
                    continue
                done +=1
                # firstind = np.where(times==time1)[0][-1]
                firstind = 0
                lastind = np.where(times==sitemax)[0][0]
                data = data[firstind:lastind,:]
                times = times[firstind:lastind]
                ts.append(list(times))
                entries.append(len(times))
                Zdata.append((data[:,18+site]/data[:,1]).tolist())
            # if done < mindis:
                # print("CD strength {}, sites, only found {} files".format(cd,done))
                # continue
            if done < mindis:
                continue
            if entries == []:
                continue
            try:
                Zcorrs = np.average(np.array(Zdata),axis=0)
            except:
                print("Z site data type={0},{1}, shape={2}".format(type(Zedge),type(np.array(Zdata)),np.shape(np.array(Zdata))))
                continue
            
            zmax = sig.find_peaks(Zcorrs,height=0.05)[0]
            # zemin = sig.find_peaks(-Zcorrs_edge,height=0.1)[0]
            
            plt.plot(times[zmax],Zcorrs[zmax],color=colors(foo1),linestyle='--',label=str(cd)+", "+str(done))
            # plt.plot(times[zemin],Zcorrs_edge[zemin],color=colors(foo),linestyle='--')
        plt.legend(loc="center right",prop={'size': 10})
        plt.xlim([0,sitemaxax])
        plt.ylim([0,1])
        plt.xlabel("$t$")
        plt.ylabel("$Z_{} (t) Z_{} (t) Z_{} (0) Z_{} (0)$".format(site+1,site+2,site+1,site+2))
    

        plt.savefig("./Corrs_CD_ZZ_nn_site{}.pdf".format(site+1))
        plt.close()

# abs
for [L,wid,pd,K1,K2,B,J] in params:
    for site in range(13):
        plt.figure()
        plt.title("$Z_{} (t) Z_{} (t) Z_{} (0) Z_{} (0)$  v. CD".format(site+1,site+2,site+1,site+2))
        for foo1,cd in enumerate(cds):
            #print()
            #print("counter drive strength = ",cd)
            print()
            done = 0
            ts = []
            Zdata = []
            # flist = glob.glob("./QP/corrs_r_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            flist = glob.glob("./QP/corrs_r1_cd"+str(cd)+"_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd*_st*.npy".format(L,pd,wid,K1,K2,J,B))
            # print(flist)
            #flist.extend(glob.glob("./QP/corrs1_L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd0_st*.npy".format(L,pd,wid,K1,K2,J,B)))
            entries = []
            for fname in flist:
                if done >= maxdis:
                    break
                state = int(fname[fname.find("_st")+3:fname.find(".npy")])
                rsd = int(fname[fname.find("_rsd")+4:fname.find("_st")])
                data = np.abs(np.load(fname))
                if len(data[0,:]) < 16:
                    # print("wrong data type")
                    continue
                # print(np.shape(data))
                times = data[:,0]
                if times[-1] < sitemax:
                    #  print("not enough times")
                    continue
                done +=1
                # firstind = np.where(times==time1)[0][-1]
                firstind = 0
                lastind = np.where(times==sitemax)[0][0]
                data = data[firstind:lastind,:]
                times = times[firstind:lastind]
                ts.append(list(times))
                entries.append(len(times))
                Zdata.append((data[:,18+site]/data[:,1]).tolist())
            # if done < mindis:
                # print("CD strength {}, sites, only found {} files".format(cd,done))
                # continue
            if done < mindis:
                continue
            if entries == []:
                continue
            try:
                Zcorrs = np.average(np.array(Zdata),axis=0)
            except:
                print("Z site data type={0},{1}, shape={2}".format(type(Zedge),type(np.array(Zdata)),np.shape(np.array(Zdata))))
                continue
            
            zmax = sig.find_peaks(Zcorrs,height=0.05)[0]
            # zemin = sig.find_peaks(-Zcorrs_edge,height=0.1)[0]
            
            plt.plot(times[zmax],Zcorrs[zmax],color=colors(foo1),linestyle='--',label=str(cd)+", "+str(done))
            # plt.plot(times[zemin],Zcorrs_edge[zemin],color=colors(foo),linestyle='--')
        plt.legend(loc="center right",prop={'size': 10})
        plt.xlim([0,sitemaxax])
        plt.ylim([0,1])
        plt.xlabel("$t$")
        plt.ylabel("$Z_{} (t) Z_{} (t) Z_{} (0) Z_{} (0)$".format(site+1,site+2,site+1,site+2))
    

        plt.savefig("./Corrs_CD_ZZ_nn_site{}_abs.pdf".format(site+1))
        plt.close()
