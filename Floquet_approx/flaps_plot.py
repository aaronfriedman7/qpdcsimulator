#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:40:19 2020

@author: Aaron
"""

import glob
# import argparse
# import os
import sys
# from scipy import linalg as lin

# need to set MKL num threads? i.e. to Ncore ... note this has to be before numpy is imported. Will probably be set as part of submission?
import numpy as np

# import matplotlib.pyplot as plt
# os.environ["MKL_NUM_THREADS"] = "1" #### for non MPI jobs.

import matplotlib.pyplot as plt
plt.rc('font', size=8)

pd=0.05
wid=0.05
K1=0.1
K2=0.3
J=0.1
B=0.1

Lcell = int(sys.argv[1]) # number of unit cells
Ns = 2*Lcell

cds = [0.0,0.0,0.05,0.1,0.15,0.2,0.25,0.5,0.75,1.0]
Cn = len(cds)

apps = list(range(8,16))

colors1 =  plt.cm.get_cmap('gnuplot',len(cds)+1)
colors2 = plt.cm.get_cmap('gnuplot',len(apps)+1)
tmax = 500
bigt = 625
### Fixed CD plots
for site in range(1):
    ylabs = ['$X_{} (t) X_{} (0)$'.format(site,site),'$Z_{} (t) Z_{} (0)$'.format(site,site),'$X_{} X_{} (t) X_{} X_{} (0)$'.format(site,site+1,site,site+1),'$Z_{} Z_{} (t) Z_{} Z_{} (0)$'.format(site,site+1,site,site+1)]
    for foo1,cd in enumerate(cds):
        if foo1 == 0:
            pbc = True
            bdystr = "_pbc_"
        else:
            pbc = False
            bdystr = "_"
        plt.figure()
        for pind in range(4):
            plt.subplot(2,2,1+pind) 
            
            for foo2,app in enumerate(apps):
                dname = "./QP/data_app{}".format(app)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd*_t*.npy".format(cd,Lcell,pd,wid,K1,K2,J,B)
                flist = list(glob.glob(dname))
                try:
                    data = np.load(flist[0])
                    ts = np.arange(len(data[0,site,:tmax]))
                except:
                    continue
                ys = data[pind,site,:tmax]
                plt.plot(ts,ys,color=colors2(foo2),label="$F_{"+str(app)+"}$")
                
            if pind%2 == 0:
                plt.xlim([0,bigt])
            else:
                plt.xlim([0,tmax])
            if pind == 0:
                plt.legend(loc="center right",prop={'size': 6})
            plt.ylabel(ylabs[pind])
            plt.xlabel('$T$')
        plt.suptitle("Correlations vs. periods for CD={}, site {} ".format(cd,site)+bdystr[1:-1])
        plt.tight_layout(pad=2.0)
        plt.savefig("CorrPlot_cd{}_L{}_site{}_Tmax{}.pdf".format(cd,Lcell,site,tmax))
        plt.close()
        
    for foo3,app in enumerate(apps):
        plt.figure()
        for pind in range(4):
            plt.subplot(2,2,1+pind) 
            
            for foo4,cd in enumerate(cds):
                if foo4 == 0:
                    pbc = True
                    bdystr = "_pbc_"
                else:
                    pbc = False
                    bdystr = "_"
                dname = "./QP/data_app{}".format(app)+bdystr+"cd{}_L{}_pd{}_wid{}_K1_{}_K2_{}_J{}_B{}_rsd*_t*.npy".format(cd,Lcell,pd,wid,K1,K2,J,B)
                flist = list(glob.glob(dname))
                try:
                    data = np.load(flist[0])
                    ts = np.arange(len(data[0,site,:tmax]))
                except:
                    continue
                ys = data[pind,site,:tmax]
                plt.plot(ts,ys,color=colors1(foo4),label="cd=${}$".format(cd)+" "+bdystr[1:-1])
                
            if pind%2 == 0:
                plt.xlim([0,bigt])
            else:
                plt.xlim([0,tmax])
            if pind == 0:
                plt.legend(loc="center right",prop={'size': 6})
            plt.ylabel(ylabs[pind])
            plt.xlabel('$T$')
        plt.suptitle("Correlations vs. periods for app={}, site {} ".format(app,site))
        plt.tight_layout(pad=2.0)
        plt.savefig("CorrPlot_app{}_L{}_site{}_Tmax{}.pdf".format(app,Lcell,site,tmax))
        plt.close()
                
            
                
            
        
            
            
            
        
