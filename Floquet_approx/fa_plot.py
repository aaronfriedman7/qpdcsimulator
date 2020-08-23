#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:06:24 2020

@author: Aaron
"""


import numpy as np
import glob
import matplotlib.pyplot as plt
# import scipy.signal as sig
# from matplotlib import cm

plt.rc('font', size=8)

# cds = [0.0,0.0,0.05,0.1,0.15,0.2,0.25,0.5,0.75,1.0]
cds = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
Ls = [3,4,5]
# apps = list(range(8,16))
# apps = list(range(9,12))

colors0 = plt.cm.get_cmap('gnuplot',len(Ls)+1)
colors1 = plt.cm.get_cmap('gnuplot',len(cds)+1)
# colors2 = plt.cm.get_cmap('gnuplot',len(apps)+1)

pd=0.05
wid=0.05
K1=0.1
K2=0.3
J=0.1
B=0.1
app=10


# times = [3,20,40]
# SUs = np.zeros(3)

# for (dum,L) in enumerate(Ls):
#     print("L={}".format(L))
#     for cd in cds:
#         print("cd = {}".format(cd))
#         enames = "ents_app10_cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(cd,L,pd,wid,K1,K2,J,B)
#         flist = glob.glob("./QP/"+enames)
#         dis = []
#         for fname in flist:
#             ents = np.load(fname)
#             dis.append(len(ents))
#         if dis == []:
#             SUs[dum] += times[dum]*240
#             if cd == 0.0:
#                SUs[dum] += times[dum]*240 
#         else:
#             dis = np.sum(np.array(dis))
#             SUs[dum] += times[dum]*(240-dis)
#             if cd == 0.0:
#                SUs[dum] += times[dum]*(240-dis)

# SUs = SUs/(60*24)
# print("SUs to do = {}".format(SUs))
                

fix  = 1.0/np.math.log(2.0)



print("doing entanglement plots now")
print()



#### ENTANGLEMENT PLOTS
    
plt.figure()
mins = []
maxes = []
for foo1,cd in enumerate(cds):
    theseLs = []
    data = []
    stds = []
    dis = []
    bdystr = "_"
    lbl = str(cd)
    enames = "ent_app{}".format(app)+bdystr+"cd{}_L*_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(cd,pd,wid,K1,K2,J,B)
    flist = glob.glob("./QP/"+enames)
    for fname in flist:
        cdnum = float(fname[fname.find("_cd")+3:fname.find("_L")])
        if cdnum == 0.75:
            print("what?")
            continue
        Lnum = 2*int(fname[fname.find("_L")+2:fname.find("_pd")])
        theseLs.append(Lnum)
        ents = np.load(fname)
        dis.append(len(ents))
        # lbl+=", "+str(len(rs))
        data.append(2*fix*np.average(ents)/Lnum)
        stds.append(2*fix*np.std(ents)/(Lnum*np.sqrt(len(ents))))
    if theseLs == []:
        print("no Ls")
        print()
        continue
    maxes.append(max(theseLs))
    mins.append(min(theseLs))
    theseLs = np.array(theseLs)
    data = np.array(data)
    stds = np.array(stds)
    inds = np.argsort(theseLs)
    theseLs = theseLs[inds]
    data = data[inds]
    stds = stds[inds]
    rsds = "["+str(min(dis))+", "+str(max(dis))+"]"
    plt.errorbar(theseLs,data,xerr=None,yerr=stds,color=colors1(foo1),label=lbl+", "+rsds)
if maxes == [] or mins == []:
    plt.close()
    print("problem with min/max")
    print()
bigL = max(maxes)
smallL = min(mins)

plt.xlim([smallL,bigL+4])
plt.legend(loc="center right",prop={'size': 6})
# plt.ylim([0,1])
plt.xlabel("System size, $L$")
plt.ylabel("$S/L$")
plt.title("Entanglement entropy")
plt.savefig("./FA_app{}_ent_v_L.pdf".format(app))
plt.close()


### vs. CD
plt.figure()
mins = []
maxes = []
for foo2,Lcell in enumerate(Ls):
    Ns = 2*Lcell
    thesecds = []
    data = []
    stds = []
    dis = []
    rnames = "ent_app{}".format(app)+"_cd*_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(Lcell,pd,wid,K1,K2,J,B)
    flist = glob.glob("./QP/"+rnames)
    for fname in flist:
        if "pbc" in fname:
            continue
        cd = float(fname[fname.find("_cd")+3:fname.find("_L")])
        if cd not in cds:
            continue
        Lnum = int(fname[fname.find("_L")+2:fname.find("_pd")])
        thesecds.append(cd)
        rs = np.load(fname)
        dis.append(len(rs))
        # lbl+=", "+str(len(rs))
        data.append(fix*np.average(rs)/Lnum)
        stds.append(fix*np.std(rs)/(Lnum*np.sqrt(len(rs))))
    if thesecds == []:
        continue
    maxes.append(max(thesecds))
    mins.append(min(thesecds))
    thesecds = np.array(thesecds)
    data = np.array(data)
    stds = np.array(stds)
    inds = np.argsort(thesecds)
    thesecds = thesecds[inds]
    data = data[inds]
    stds = stds[inds]
    rsds = "["+str(min(dis))+", "+str(max(dis))+"]"
    plt.errorbar(thesecds,data,xerr=None,yerr=stds,color=colors0(foo2),label=str(Ns)+", "+rsds)

if maxes == [] or mins == []:
    plt.close()
bigcd = max(maxes)
smallcd = min(mins)

plt.xlim([smallcd,bigcd+0.3])
plt.legend(loc="center right",prop={'size': 6})
# plt.ylim([0,1])
plt.xlabel("counter drive strength")
plt.ylabel("$S/L$")
plt.title("Entanglement entropy")
plt.savefig("./FA_app{}_ent_v_cd.pdf".format(app))
plt.close()










print("level statistics plots now")
print()
print()






plt.figure()
print("approximate = {}".format(app))
print()
mins = []
maxes = []
for foo1,cd in enumerate(cds):
    theseLs = []
    data = []
    dis = []
    stds = []
    bdystr = "_"
    lbl = str(cd)
    rnames = "rs_app{}".format(app)+bdystr+"cd{}_L*_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(cd,pd,wid,K1,K2,J,B)
    flist = glob.glob("./QP/"+rnames)
    for fname in flist:
        cdnum = float(fname[fname.find("_cd")+3:fname.find("_L")])
        if cdnum == 0.75:
            print("what?")
            continue
        Lnum = 2*int(fname[fname.find("_L")+2:fname.find("_pd")])
        print("for L = {}, file = {}".format(Lnum,fname))
        theseLs.append(Lnum)
        rs = np.load(fname)
        print(rs)
        print()
        dis.append(len(rs))
        # lbl+=", "+str(len(rs))
        data.append(np.average(rs))
        stds.append(np.std(rs)/(np.sqrt(len(rs))))
    if theseLs == []:
        continue
    maxes.append(max(theseLs))
    mins.append(min(theseLs))
    theseLs = np.array(theseLs)
    data = np.array(data)
    stds = np.array(stds)
    inds = np.argsort(theseLs)
    theseLs = theseLs[inds]
    data = data[inds]
    stds = stds[inds]
    rsds = min(dis)
    plt.errorbar(theseLs,data,xerr=None,yerr=stds,color=colors1(foo1),label=lbl+", "+str(rsds))

bigL = max(maxes)
smallL = min(mins)

plt.xlim([smallL,bigL+4])
plt.legend(loc="center right",prop={'size': 6})
# plt.ylim([0,1])
plt.xlabel("System size, $L$")
plt.ylabel("$r$ ratio")
plt.title("Level statistics of pseudoenergies")
plt.savefig("./FA_app{}_rvals_v_L.pdf".format(app))
plt.close()


### vs. CD
plt.figure()
mins = []
maxes = []
for foo2,Lcell in enumerate(Ls):
    Ns = 2*Lcell
    thesecds = []
    data = []
    dis = []
    stds = []
    rnames = "rs_app{}".format(app)+"_cd*_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(Lcell,pd,wid,K1,K2,J,B)
    flist = glob.glob("./QP/"+rnames)
    for fname in flist:
        if "pbc" in fname:
            continue
        cd = float(fname[fname.find("_cd")+3:fname.find("_L")])
        if cd not in cds:
            continue
        thesecds.append(cd)
        rs = np.load(fname)
        # print(rs)
        # print()
        dis.append(len(rs))
        # lbl+=", "+str(len(rs))
        data.append(np.average(rs))
        stds.append(np.std(rs)/(np.sqrt(len(rs))))
    if thesecds == []:
        continue
    maxes.append(max(thesecds))
    mins.append(min(thesecds))
    thesecds = np.array(thesecds)
    data = np.array(data)
    stds = np.array(stds)
    inds = np.argsort(thesecds)
    thesecds = thesecds[inds]
    data = data[inds]
    stds = stds[inds]
    rsds = min(dis)
    plt.errorbar(thesecds,data,xerr=None,yerr=stds,color=colors0(foo2),label=str(Ns)+", "+str(rsds))

bigcd = max(maxes)
smallcd = min(mins)

plt.xlim([smallcd,bigcd+0.3])
plt.legend(loc="center right",prop={'size': 6})
# plt.ylim([0,1])
plt.xlabel("counter drive strength")
plt.ylabel("$r$ ratio")
plt.title("Level statistics of pseudoenergies")
plt.savefig("./FA_app{}_rvals_v_cd.pdf".format(app))
plt.close()


            
        

### correlation plots

for site in range(Ns):
    ylabs = ['$X_{} (t) X_{} (0)$'.format(site,site),'$Z_{} (t) Z_{} (0)$'.format(site,site),'$X_{} X_{} (t) X_{} X_{} (0)$'.format(site,site+1,site,site+1),'$Z_{} Z_{} (t) Z_{} Z_{} (0)$'.format(site,site+1,site,site+1)]
    plt.figure()
    for pind in range(4):
        plt.subplot(2,2,1+pind)
        mins = []
        maxes = []
        ### level statistics plot:
        for foo1,cd in enumerate(cds):
            apps = []
            data = []
            if foo1 == 0:
                bdystr = "_pbc_"
                lbl = "0.0 (pbc)"
            else:
                bdystr = "_"
                lbl = str(cd)
            rnames = "corr_app*_cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}.npy".format(cd,Lcell,pd,wid,K1,K2,J,B)
            flist = glob.glob("./QP/"+rnames)
            for fname in flist:
                if foo1 == 0:
                    if "pbc" not in fname:
                        continue
                    appnum = int(fname[fname.find("_app")+4:fname.find("_pbc")])
                else:
                    if "pbc" in fname:
                        # print("pbc there")
                        continue
                    appnum = int(fname[fname.find("_app")+4:fname.find("_cd")])
                apps.append(appnum)
                corrs = np.load(fname)
                # lbl+=", "+str(len(corrs[:,0,0]))
                corr = np.average(corrs,axis=0)[pind,site]
                corrdev = np.std(corrs,axis=0)[pind,site]
                # print("for site {} correlations avg = {} with std = {}".format(site,corr,corrdev))
                ### they appear to be comparable, near - 0 averages are real.
                data.append(corr)
            if apps == []:
                continue
            maxes.append(max(apps))
            mins.append(min(apps))
            inds = np.argsort(apps)
            apps = np.array(apps)
            data = np.array(data)
            apps = apps[inds]
            data = data[inds]
            plt.plot(apps,data,color=colors1(foo1),label=lbl)
        bigapp = max(maxes)
        smallapp = min(mins)
        plt.ylabel(ylabs[pind])
        plt.xlabel('$n$')
        if pind%2 == 1:
            plt.xlim([smallapp,bigapp])
            # plt.legend(loc="center right")
        # plt.ylim([0,1])
    plt.tight_layout(pad=2.0)
    plt.suptitle("Inf. Time / Temp. Correlations.")
    plt.savefig("./FA_corrs_site{}.pdf".format(site))
    plt.close()



