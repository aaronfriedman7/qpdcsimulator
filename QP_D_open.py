#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:57:37 2020

@author: Aaron
"""
import argparse
import numpy as np
from scipy import integrate as scint
import scipy.sparse as ss
from scipy import linalg as lin
# import glob
# import os

# mainpath = "07375/ajfriedm/QP_tevo/"
# workpath = "/scratch/"+mainpath
# os.chdir(workpath)
workpath = "./"

## Y = i X Z

def spinstr(spinint, N):
    return bin(int(spinint))[2:].zfill(N)


def getint(N,config):
    return int(config.zfill(N),base=2)

def Sparse_Comm(A,B):
    return ss.csr_matrix(A.dot(B) - B.dot(A))

def Sparse_Herm(A):
    return (A.conjugate()).transpose()

def jobhead(params,rsd):
    return "L{0}_pd{1}_wid{2}_K1_{3}_K2_{4}_J{5}_B{6}_rsd{7}".format(params.L,params.pd,params.wid,params.K1,params.K2,params.J,params.B,rsd)

golden = 0.5*(1.0+np.sqrt(5.0))
omegaX = 2.0*np.pi
omegaZ = 2.0*np.pi/golden

def realization(params,rsd):
    Lcell = params.L
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
    return Kvalues,Jvalues,fields

def Fmain(x,omega,stddev,accuracy=10.0**(-8.0),giveup=2000,mintry=25):
    f = x - (np.pi/2.0)
    m = 1
    goodjob = 0
    while m <= giveup:
        newterm = np.exp(-0.5*((omega*stddev*m)**2))*np.sin(2*m*x)/float(m)
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

def c1Re(x,omega,stddev,m):
    return np.cos(m*x)*np.cos(Fmain(x,omega,stddev))/(2.0*np.pi)


def c1Im(x,omega,stddev,m):
    return np.sin(m*x)*np.cos(Fmain(x,omega,stddev))/(2.0*np.pi)


def c2Re(x,omega,stddev,m):
    return np.cos(m*x)*(1+np.cos(2.0*Fmain(x,omega,stddev)))/(4.0*np.pi)


def c2Im(x,omega,stddev,m):
    return np.sin(m*x)*(1+np.cos(2.0*Fmain(x,omega,stddev)))/(4.0*np.pi)

def s1Re(x,omega,stddev,m):
    return np.cos(m*x)*np.sin(Fmain(x,omega,stddev))/(2.0*np.pi)


def s1Im(x,omega,stddev,m):
    return np.sin(m*x)*np.sin(Fmain(x,omega,stddev))/(2.0*np.pi)


def s2Re(x,omega,stddev,m):
    return np.cos(m*x)*(1-np.cos(2.0*Fmain(x,omega,stddev)))/(4.0*np.pi)


def s2Im(x,omega,stddev,m):
    return np.sin(m*x)*(1-np.cos(2.0*Fmain(x,omega,stddev)))/(4.0*np.pi)


def SothRe(x,omega,stddev,m):
    return np.cos(m*x)*np.sin(2.0*Fmain(x,omega,stddev))/(4.0*np.pi)


def SothIm(x,omega,stddev,m):
    return np.sin(m*x)*np.sin(2.0*Fmain(x,omega,stddev))/(4.0*np.pi)
#print("for m={0}, coeff = {1},{2}, error = {3},{4} (Re,Im)".format(every*m,ValRe,ValIm,errRe,errIm))
#print("magnitudes = {0} and err mag = {1}".format(np.abs(ValRe + 1j*ValIm),np.abs(errRe + 1j*errIm)))


### Purely imaginary, odd m only, odd func of m
def C1val(omega,stddev,m):
    # ValRe,errRe = scint.quad(c1Re,0.0,2.0*np.pi,args=(omega,stddev,2*m+1),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    ValIm,errIm = scint.quad(c1Im,0.0,2.0*np.pi,args=(omega,stddev,2*m+1),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    if np.abs(ValIm) < 1e-8: 
        ValIm = 0.0
    return ValIm

### Purely real, even m only, even func of m
def C2val(omega,stddev,m):
    ValRe,errRe = scint.quad(c2Re,0.0,2.0*np.pi,args=(omega,stddev,2*m),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    # ValIm,errIm = scint.quad(c2Im,0.0,2.0*np.pi,args=(omega,stddev,2*m),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    if np.abs(ValRe) < 1e-8: 
        ValRe = 0.0
    return ValRe

### Purely real, odd m only, even func of m
def S1val(omega,stddev,m):
    ValRe,errRe = scint.quad(s1Re,0.0,2.0*np.pi,args=(omega,stddev,2*m+1),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    # ValIm,errIm = scint.quad(s1Im,0.0,2.0*np.pi,args=(omega,stddev,2*m+1),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    if np.abs(ValRe) < 1e-8: 
        ValRe = 0.0
    return ValRe


def S2val(omega,stddev,m):
    ValRe,errRe = scint.quad(s2Re,0.0,2.0*np.pi,args=(omega,stddev,2*m),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    ValIm,errIm = scint.quad(SothIm,0.0,2.0*np.pi,args=(omega,stddev,2*m),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    if np.abs(ValRe) < 1e-8: 
        ValRe = 0.0
    if np.abs(ValIm) < 1e-8: 
        ValIm = 0.0
    return (ValRe,ValIm)


def Mval(omega,stddev,m):
    # ValRe,errRe = scint.quad(SothRe,0.0,2.0*np.pi,args=(omega,stddev,2*m),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    ValIm,errIm = scint.quad(SothIm,0.0,2.0*np.pi,args=(omega,stddev,2*m),epsabs=1.0e-13,epsrel=1.0e-13,limit=100)
    if np.abs(ValIm) < 1e-8: 
        ValIm = 0.0
    return ValIm


### call with maxm=0 to get m=0
    

#### Below define the individual terms that get multiplied by different Fourier factors
### all of these default to positive (any minus signs are incorporated)
### minus signs should only be used if c(-m) = - c(m), since I'll only save m>=0 entries


## odd n only, multiplied by c1z(m) or -c1z (for -m)
def X_c(params,fields):
    vals = []
    rows = []
    cols = []
    Lcell = params.L
    Size = 4**Lcell
    for state in range(Size):
        for site in range(2*Lcell):
            vals.append(fields[0,site])
            rows.append(state^(2**(2*Lcell-site-1)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size))


## odd n only, multiplied by s1z(m) for either m or -m
def X_s(params,fields):
    vals = []
    rows = []
    cols = []
    Lcell = params.L
    Size = 4**Lcell
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        for cell in range(Lcell):
            siteA = 2*cell
            spinA = (2*int(scfg[siteA])-1)
            spinB = (2*int(scfg[siteA+1])-1)
            ## Y flip siteA, measure siteB; coefficient from Y
            vals.append(-spinA*spinB*1j*fields[0,siteA])
            rows.append(state^(2**(2*Lcell-siteA-1)))
            cols.append(state)
            ## Y flip siteB, measure siteA; coefficient from Y
            vals.append(-spinA*spinB*1j*fields[0,siteA+1])
            rows.append(state^(2**(2*Lcell-siteA-2)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size))


    

### for Y fields, default is x Fourier type first, z Fourier type second
    
### odd nx and odd nz only, multiplied by c1x(mx) c1z(mz), or - this if EITHER mx or mz is negative
def Y_cc(params,fields):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    rows = []
    cols = []
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        for site in range(2*Lcell):
            spin = (2*int(scfg[site])-1)
            vals.append(1j*spin*fields[1,site])
            rows.append(state^(2**(2*Lcell-site-1)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size),dtype=complex)


### odd nx and nz only, multiplied by s1x(mx) s1z(mz):
def Y_ss(params,fields):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    rows = []
    cols = []
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        for cell in range(Lcell):
            siteA = 2*cell
            spinA = (2*int(scfg[siteA])-1)
            spinB = (2*int(scfg[siteA+1])-1)
            # Y flip site A with site B coeff
            vals.append(1j*spinA*fields[1,siteA+1])
            rows.append(state^(2**(2*Lcell-siteA-1)))
            cols.append(state)
            # Y flip site B with site A coeff
            vals.append(1j*spinB*fields[1,siteA])
            rows.append(state^(2**(2*Lcell-siteA-2)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size),dtype=complex)


### odd nx and nz only, multiplied by c1z(mz) s1x(mx), or - this if for mz <0
def Y_cs(params,fields):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    rows = []
    cols = []
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        for cell in range(Lcell):
            siteA = 2*cell
            spinA = (2*int(scfg[siteA])-1)
            spinB = (2*int(scfg[siteA+1])-1)
            # X flip site A and Z measure B (field corresponds to X)
            vals.append(spinB*fields[1,siteA])
            rows.append(state^(2**(2*Lcell-siteA-1)))
            cols.append(state)
            # X flip site B and Z measure A (field corresponds to X)
            vals.append(spinA*fields[1,siteA+1])
            rows.append(state^(2**(2*Lcell-siteA-2)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size))


### odd nx and nz only, multiplied by s1z(mz) c1x(mx), or - this if for mx <0
def Y_sc(params,fields):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    rows = []
    cols = []
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        for cell in range(Lcell):
            siteA = 2*cell
            spinA = (2*int(scfg[siteA])-1)
            spinB = (2*int(scfg[siteA+1])-1)
            # X flip site A and Z measure B (field corresponds to Z)
            vals.append(-spinB*fields[1,siteA+1])
            rows.append(state^(2**(2*Lcell-siteA-1)))
            cols.append(state)
            # X flip site B and Z measure A (field corresponds to Z)
            vals.append(-spinA*fields[1,siteA])
            rows.append(state^(2**(2*Lcell-siteA-2)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size))



### odd nx only, multiplied by c1z(mz) , or - this if for mz <0, used coo matrix for convenience, just storing vals would be more efficient, but annoying later.
def Z_c(params,fields):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        term = 0.0
        for site in range(2*Lcell):
            term += (2*int(scfg[site])-1)*fields[2,site]
        vals.append(term)
    return ss.coo_matrix((np.array(vals),(np.arange(Size),np.arange(Size))),shape=(Size,Size))


## odd nx only, multiplied by s1x(m) for either mx or -mx
def Z_s(params,fields):
    vals = []
    rows = []
    cols = []
    Lcell = params.L
    Size = 4**Lcell
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        for cell in range(Lcell):
            siteA = 2*cell
            spinA = (2*int(scfg[siteA])-1)
            ## Y flip site A, X flip site B, coeff corresponds to Y site
            vals.append(1j*spinA*fields[2,siteA])
            rows.append(state^(2**(2*Lcell-siteA-2) + 2**(2*Lcell-siteA-1)))
            cols.append(state)
            ## Y flip site B, X flip site A, coeff corresponds to Y site
            spinB = (2*int(scfg[siteA+1])-1)
            vals.append(1j*spinB*fields[0,siteA+1])
            rows.append(state^(2**(2*Lcell-siteA-2) + 2**(2*Lcell-siteA-1)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size),dtype=complex)


### even nx only, multiplied by np.exp(-0.125*(params.wid*nx*omegaX)**2)
def X_detune(params):
    vals = []
    rows = []
    cols = []
    Lcell = params.L
    Size = 4**Lcell
    factor = -(0.25*omegaX*params.pd)/(1.0+params.pd)
    for state in range(Size):
        for cell in range(Lcell):
            vals.append(factor)
            rows.append(state^(2**(2*Lcell-2*cell-1)+2**(2*Lcell-2*cell-2)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size))


### even nz only, multiplied by np.exp(-0.125*(params.wid*nz*omegaZ)**2)
def Z_detune(params):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    factor = -(0.25*omegaZ*params.pd)/(1.0+params.pd)
    for state in range(Size):
        term = 0.0
        scfg = spinstr(state,2*Lcell)
        for cell in range(Lcell):
            if scfg[2*cell] == scfg[2*cell + 1]:
                term += factor
            else:
                term -= factor
        vals.append(term)
    return ss.coo_matrix((np.array(vals),(np.arange(Size),np.arange(Size))),shape=(Size,Size))


### only appear in V0 / D0, correspond to J values on site, feed number of unit cells, Jarray
def cellterms(L,coeffs):
    vals = []
    rows = []
    cols = []
    Size = 4**L
    for state in range(Size):
        scfg = spinstr(state,2*L)
        term = 0.0
        for cell in range(L):
            vals.append(coeffs[0,cell])
            rows.append(state^(2**(2*L-2*cell-1)+2**(2*L-2*cell-2)))
            cols.append(state)
            if scfg[2*cell] == scfg[2*cell + 1]:
                term += coeffs[1,cell]
            else:
                term -= coeffs[1,cell]
        vals.append(term)
        rows.append(state)
        cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size))

        

### multiply by c2z (nz) even nz only, the nice part of the XX stabilizer terms 1XX1 across two cells
def XX_c(params,coeffs):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    rows = []
    cols = []
    for state in range(Size):
        for cell in range(Lcell-1):
            vals.append(coeffs[0,cell])
            rows.append(state^(2**(2*Lcell-2*cell-2)+2**(2*Lcell-2*cell-3)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size))


### multiply by s2z (nz) even nz only, the eh part of the XX stabilizer terms ZYYZ across two cells
def XX_s(params,coeffs):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    rows = []
    cols = []
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        for cell in range(Lcell-1):
            if (scfg[2*cell:2*cell+4]).count('1') %2 == 1:
                vals.append(coeffs[0,cell])
            else:
                vals.append(-coeffs[0,cell])
            rows.append(state^(2**(2*Lcell-2*cell-2) + 2**(2*Lcell-2*cell-3)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size))


### multiply by m2z (nz) even nz only, the shit part of the XX stabilizer terms 1XYZ + ZYX1 across two cells
def XX_m(params,coeffs):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    rows = []
    cols = []
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        ### (1XYZ + ZYX1) = i (1 X X 1) (1 1 Z Z + Z Z 1 1), all with same coeff of -
        for cell in range(Lcell-1):
            if (scfg[2*cell] == scfg[2*cell+1]) and (scfg[2*cell+2] == scfg[2*cell+3]):
                vals.append(-2.0*coeffs[0,cell]*1j)
                rows.append(state^(2**(2*Lcell-2*cell-2) + 2**(2*Lcell-2*cell-3)))
                cols.append(state)
            elif (scfg[2*cell] != scfg[2*cell+1]) and (scfg[2*cell+2] != scfg[2*cell+3]):
                vals.append(2.0*coeffs[0,cell]*1j)
                rows.append(state^(2**(2*Lcell-2*cell-2) + 2**(2*Lcell-2*cell-3)))
                cols.append(state)
            else:
                continue
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size),dtype=complex)




### multiply by c2x (nx) even nx only, the nice part of the ZZ stabilizer terms 1ZZ1 across two cells
def ZZ_c(params,coeffs):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    for state in range(Size):
        term = 0.0
        scfg = spinstr(state,2*Lcell)
        for cell in range(Lcell-1):
            if scfg[2*cell+1] == scfg[2*cell + 2]:
                term -= coeffs[1,cell]
            else:
                term += coeffs[1,cell]
        vals.append(term)
    return ss.coo_matrix((np.array(vals),(np.arange(Size),np.arange(Size))),shape=(Size,Size))


### multiply by s2x (nx) even nx only, the less nice part of the ZZ stabilizer terms XYYX across two cells
def ZZ_s(params,coeffs):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    rows = []
    cols = []
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        for cell in range(Lcell-1):
            if scfg[2*cell+1] == scfg[2*cell+2]:
                vals.append(-coeffs[1,cell])
            else:
                vals.append(coeffs[1,cell])
            rows.append(state^( 2**(2*Lcell-2*cell-1) + 2**(2*Lcell-2*cell-2) + 2**(2*Lcell-2*cell-3)+2**(2*Lcell-2*cell-4)))
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size))



### multiply by m2x (nx) even nx only, the shit part of the ZZ stabilizer terms 1ZYX + XYZ1 across two cells
def ZZ_m(params,coeffs):
    Lcell = params.L
    Size = 4**Lcell
    vals = []
    rows = []
    cols = []
    for state in range(Size):
        scfg = spinstr(state,2*Lcell)
        ### (1 Z Y X  +  X Y Z 1) = i (1 1 X X  +  X X 1 1 ) (1 Z Z 1)
        for cell in range(Lcell-1):
            if scfg[2*cell+1] == scfg[2*cell+2]:
                vals.append(coeffs[1,cell]*1j)
                vals.append(coeffs[1,cell]*1j)
            else:
                vals.append(-coeffs[1,cell]*1j)
                vals.append(-coeffs[1,cell]*1j)    
            rows.append(state^(2**(2*Lcell-2*cell-1) + 2**(2*Lcell-2*cell-2)))
            rows.append(state^(2**(2*Lcell-2*cell-3) + 2**(2*Lcell-2*cell-4)))
            cols.append(state)
            cols.append(state)
    return ss.coo_matrix((np.array(vals),(np.array(rows),np.array(cols))),shape=(Size,Size),dtype=complex)


def D0(params,Ks,Js,FourierX,FourierZ):
    L=params.L
    D0 = cellterms(L,Js)
    D0 += X_detune(params) + Z_detune(params)
    D0 += C2val(omegaZ,params.wid,0)*XX_c(params,Ks)
    D0 += (S2val(omegaZ,params.wid,0)[0])*XX_s(params,Ks)
    D0 += C2val(omegaX,params.wid,0)*ZZ_c(params,Ks)
    D0 += (S2val(omegaX,params.wid,0)[0])*ZZ_s(params,Ks)
    return D0


def D1_X_field(params,fields,FourierZ):
    numterms = len(FourierZ[0,:])
    Lcell = params.L
    Size = 4**Lcell
    mat = ss.coo_matrix((Size,Size),dtype=complex)
    for m in range(numterms):
        sym = X_s(params,fields)
        anti = X_c(params,fields)
        if len((sym.nonzero())[0]) == 0 or len((anti.nonzero())[0]) == 0:
            continue
        else:
            mat += (2.0/(omegaZ*(2*m+1)))*1j*FourierZ[0,m]*FourierZ[1,m]*Sparse_Comm(anti,sym)
    return mat


def D1_Y_field(params,fields,FourierX,FourierZ):
    numterms = len(FourierZ[0,:])
    Lcell = params.L
    Size = 4**Lcell
    mat = ss.coo_matrix((Size,Size),dtype=complex)
    for mx in range(numterms):
        for mz in range(numterms):
            sym = -FourierZ[0,mz]*FourierX[0,mx]*Y_cc(params,fields) + FourierZ[1,mz]*FourierX[1,mx]*Y_ss(params,fields) ## for mx, mz same sign
            anti = FourierZ[1,mz]*FourierX[0,mx]*Y_sc(params,fields) + FourierZ[0,mz]*FourierX[1,mx]*Y_cs(params,fields) ## mx,mz > 0 ; - this if both < 0
            if not len((sym.nonzero())[0]) == 0 or len((anti.nonzero())[0]) == 0:
                mat += (2.0/(omegaZ*(2*mz+1) + omegaX*(2*mx+1)))*1j*(Sparse_Comm(anti,sym)) 
            sym = FourierZ[0,mz]*FourierX[0,mx]*Y_cc(params,fields) + FourierZ[1,mz]*FourierX[1,mx]*Y_ss(params,fields) ## for mx mz different sign
            anti = FourierZ[1,mz]*FourierX[0,mx]*Y_sc(params,fields) - FourierZ[0,mz]*FourierX[1,mx]*Y_cs(params,fields) ## for mx > 0 , mz < 0; - this for opposite
            if not len((sym.nonzero())[0]) == 0 or len((anti.nonzero())[0]) == 0:
                mat += (2.0/(omegaX*(2*mx+1) - omegaZ*(2*mz+1)))*1j*(Sparse_Comm(anti,sym)) 
    return mat



def D1_Z_field(params,fields,FourierX):
    numterms = len(FourierX[0,:])
    Lcell = params.L
    Size = 4**Lcell
    mat = ss.coo_matrix((Size,Size),dtype=complex)
    for m in range(numterms):
        sym = Z_s(params,fields)
        anti = Z_c(params,fields)
        if len((sym.nonzero())[0]) == 0 or len((anti.nonzero())[0]) == 0:
            continue
        else:
            mat += (2.0/(omegaX*(2*m+1)))*1j*FourierX[0,m]*FourierX[1,m]*Sparse_Comm(anti,sym)
    return mat
            
         
def D1_nx_even(params,Ks,FourierX):
    numterms = len(FourierX[0,:])
    Lcell = params.L
    Size = 4**Lcell
    mat = ss.coo_matrix((Size,Size),dtype=complex)
    for nx in range(numterms):
        sym = np.exp(-0.5*(params.wid*(nx+1)*omegaX)**2)*X_detune(params)
        sym += FourierX[2,nx]*ZZ_c(params,Ks) + FourierX[3,nx]*ZZ_s(params,Ks)
        anti = FourierX[5,nx]*ZZ_m(params,Ks) + FourierX[4,nx]*ZZ_s(params,Ks)
        if len((sym.nonzero())[0]) == 0 or len((anti.nonzero())[0]) == 0:
            continue
        else:
            mat += (1.0/(omegaX*(nx+1)))*1j*Sparse_Comm(anti,sym)
    return mat
          
      
def D1_nz_even(params,Ks,FourierZ):
    numterms = len(FourierZ[0,:])
    Lcell = params.L
    Size = 4**Lcell
    mat = ss.coo_matrix((Size,Size),dtype=complex)
    for nz in range(numterms):
        sym = np.exp(-0.5*(params.wid*(nz+1)*omegaZ)**2)*Z_detune(params)
        sym += FourierZ[2,nz]*XX_c(params,Ks) + FourierZ[3,nz]*XX_s(params,Ks)
        anti = FourierZ[5,nz]*XX_m(params,Ks) + FourierZ[4,nz]*XX_s(params,Ks)
        if len((sym.nonzero())[0]) == 0 or len((anti.nonzero())[0]) == 0:
            continue
        else:
            mat += (1.0/(omegaZ*(nz+1)))*1j*Sparse_Comm(anti,sym)
    return mat                                                

                                                
def D1(params,Ks,Js,fields,FourierX,FourierZ):
    D1 = D1_X_field(params,fields,FourierZ) + D1_Y_field(params,fields,FourierX,FourierZ) + D1_Z_field(params,fields,FourierX)
    D1 += D1_nx_even(params,Ks,FourierX) + D1_nz_even(params,Ks,FourierZ)
    return D1


## thetaX = 0.5*omegaX*time + 0.5*np.pi, thetaZ = 0.5*omegaZ*time + 0.5*np.pi
def G1(params,thetaX,thetaZ,Ks,Js,Bs,FourierX,FourierZ):
    numterms = len(FourierX[0,:])
    Lcell = params.L
    Size = 4**Lcell
    mat = ss.coo_matrix((Size,Size),dtype=complex)
    for mz in range(numterms):
        mze = 2*(mz+1)
        mzo = mze-1
        mat += (2.0*FourierZ[0,mz]*np.cos(mzo*thetaZ)/(omegaZ*mzo))*X_c(params,Bs) + (2.0*FourierZ[1,mz]*np.sin(mzo*thetaZ)/(omegaZ*mzo))*X_s(params,Bs)
        mat += (2*np.sin(mze*thetaZ)*np.exp(-0.125*(params.wid*mze*omegaZ)**2)/(omegaZ*mze))*Z_detune(params)
        mat += (2*np.sin(mze*thetaZ)*FourierZ[2,mz]/(omegaZ*mze))*XX_c(params,Ks) + (2*np.sin(mze*thetaZ)*FourierZ[3,mz]/(omegaZ*mze))*XX_s(params,Ks)
        mat += (2*np.cos(mze*thetaZ)*FourierZ[4,mz]/(omegaZ*mze))*XX_s(params,Ks) + (2*np.cos(mze*thetaZ)*FourierZ[5,mz]/(omegaZ*mze))*XX_m(params,Ks)
    for mx in range(numterms):
        mxe = 2*(mx+1)
        mxo = mxe-1
        mat += (2.0*FourierX[0,mx]*np.cos(mxo*thetaX)/(omegaX*mxo))*Z_c(params,Bs) + (2.0*FourierX[1,mx]*np.sin(mxo*thetaX)/(omegaX*mxo))*Z_s(params,Bs)
        mat += (2*np.sin(mxe*thetaX)*np.exp(-0.125*(params.wid*mxe*omegaX)**2)/(omegaX*mxe))*X_detune(params)
        mat += (2*np.sin(mxe*thetaX)*FourierX[2,mx]/(omegaX*mxe))*ZZ_c(params,Ks) + (2*np.sin(mxe*thetaX)*FourierX[3,mx]/(omegaX*mxe))*ZZ_s(params,Ks)
        mat += (2*np.cos(mxe*thetaX)*FourierX[4,mx]/(omegaX*mxe))*ZZ_s(params,Ks) + (2*np.cos(mxe*thetaX)*FourierX[5,mx]/(omegaX*mxe))*ZZ_m(params,Ks)
    for m1 in range(numterms):
        mx = 2*m1+1
        for m2 in range(numterms):
            mz = 2*m2+1
            mat += (2.0*FourierX[1,m1]*FourierZ[1,m2]*np.sin(mx*thetaX+mz*thetaZ)/(omegaX*mx+omegaZ*mz))*Y_ss(params,Bs)
            mat += (2.0*FourierX[1,m1]*FourierZ[1,m2]*np.sin(mx*thetaX-mz*thetaZ)/(omegaX*mx-omegaZ*mz))*Y_ss(params,Bs)
            mat += (2.0*FourierX[0,m1]*FourierZ[0,m2]*np.sin(mx*thetaX-mz*thetaZ)/(omegaX*mx-omegaZ*mz))*Y_cc(params,Bs)
            mat -= (2.0*FourierX[0,m1]*FourierZ[0,m2]*np.sin(mx*thetaX+mz*thetaZ)/(omegaX*mx+omegaZ*mz))*Y_cc(params,Bs)
            mat += (2.0*FourierX[1,m1]*FourierZ[0,m2]*np.cos(mx*thetaX+mz*thetaZ)/(omegaX*mx+omegaZ*mz))*Y_sc(params,Bs)
            mat -= (2.0*FourierX[1,m1]*FourierZ[0,m2]*np.cos(mx*thetaX-mz*thetaZ)/(omegaX*mx-omegaZ*mz))*Y_sc(params,Bs)
            mat += (2.0*FourierX[0,m1]*FourierZ[1,m2]*np.cos(mx*thetaX+mz*thetaZ)/(omegaX*mx+omegaZ*mz))*Y_cs(params,Bs)
            mat += (2.0*FourierX[0,m1]*FourierZ[1,m2]*np.cos(mx*thetaX-mz*thetaZ)/(omegaX*mx-omegaZ*mz))*Y_cs(params,Bs)
    return mat

                                              
### confirmed, save as array
def Zop(params,j=0):
    Lcell = params.L
    try:
        op = ss.load_npz(workpath+"Z_L{0}_j{1}.npz".format(Lcell,j))
    except:
        Size = 4**Lcell
        vals = []
        for state in range(Size):
            scfg = spinstr(state,2*Lcell)
            vals.append(2.0*int(scfg[j])-1.0)
        op =  ss.coo_matrix((np.array(vals),(np.arange(Size),np.arange(Size))),shape=(Size,Size),dtype=float)
        ss.save_npz(workpath+"Z_L{0}_j{1}.npz".format(Lcell,j),op)
    return op


### confirmed save as array
def Xop(params,j=0):
    Lcell = params.L
    try:
        op = ss.load_npz(workpath+"X_L{0}_j{1}.npz".format(Lcell,j))
    except:
        Size = 4**Lcell
        vals = []
        rows = []
        op = ss.dok_matrix((Size,Size),dtype=float)
        for state in range(Size):
            rows.append(state^(2**(2*Lcell-1-j)))
            vals.append(1.0)
        op =  ss.coo_matrix((np.array(vals),(np.array(rows),np.arange(Size))),shape=(Size,Size),dtype=float)
        ss.save_npz(workpath+"X_L{0}_j{1}.npz".format(Lcell,j),op)
    return op


def Xsym(params):
    Lcell=params.L
    try:
        op = ss.load_npz(workpath+"Xsym_L{0}.npz".format(Lcell))
    except:
        Size = 4**Lcell
        Vals = []
        Rows = []
        ev = 0
        for j in range(2*Lcell):
            ev += 2**j
        for state in range(Size):
            Vals.append(1.0)
            Rows.append(state^ev)
        op = ss.coo_matrix((np.array(Vals),(np.array(Rows),np.arange(Size))),shape=(Size,Size),dtype=float)
        ss.save_npz(workpath+"Xsym_L{0}.npz".format(Lcell),op)
    return op

def Zsym(params):
    Lcell=params.L
    try:
        op = ss.load_npz(workpath+"Zsym_L{0}.npz".format(Lcell))
    except:
        Size = 4**Lcell
        vals = []
        for state in range(Size):
            scfg = spinstr(state,2*Lcell)
            term = np.product(np.array([2.0*int(scfg[j])-1.0 for j in range(2*Lcell)])) 
            vals.append(term)
        op = ss.coo_matrix((np.array(vals),(np.arange(Size),np.arange(Size))),shape=(Size,Size),dtype=float)
        ss.save_npz(workpath+"Zsym_L{0}.npz".format(Lcell),op)
    return op

def rvals(evlist):
    # evlist[evlist<1.e-8] = 0.0
    # inds = np.where(evlist != 0.0)[0]
    # evlist = evlist[inds]
    rlist =[]
    delts = [(evlist[i+1]-evlist[i]) for i in range(len(evlist)-1)]
    for j in range(len(delts)-1):
        r = min(delts[j],delts[j+1])/max(delts[j],delts[j+1])
        if not np.isnan(r):
            rlist.append(r)
        else:
            return np.zeros(1)
    if len(evlist) > len(rlist)+2:
        print('Threw away {0} values out of {1}'.format(len(evlist)-2-len(rlist),len(evlist)-2))
    return np.array(rlist)





### stored as real values for positive m
### imaginary guys are odd funcs of m, real guys are even funcs of m
### only allowed ms are stored
def FT_Coeffs(omega,width,numterms):
    mat = np.zeros((6,numterms),dtype=float)
    for m in range(numterms):
        mat[0,m] = C1val(omega,width,m) # IMAGINARY; ODD m only
        mat[1,m] = S1val(omega,width,m) # REAL; ODD m only
        mat[2,m] = C2val(omega,width,m+1) # REAL; EVEN m only
        S2tup = S2val(omega,width,m+1)
        mat[3,m] = S2tup[0] # REAL; EVEN m only
        mat[4,m] = S2tup[1] # IMAGINARY; EVEN m only
        mat[5,m] = Mval(omega,width,m+1) # IMAGINARY; EVEN m only
    return mat

def proj(params,evenZ=True,evenX=True,A=1.0,B=0.3):
    Lcell = params.L
    if evenZ:
        znum=0
    else:
        znum=1
    if evenX:
        xnum = 0
    else:
        xnum = 1
    try:
        op = ss.load_npz(workpath+"Proj_L{0}_x{1}_z{2}.npz".format(Lcell,xnum,znum))
    except:
        test = (A*Zsym(params)+B*Xsym(params)).todense()
        eigs,vecs = lin.eigh(test)
        print(list(set(list(eigs))))
        print()
        inds = np.where(np.abs(eigs-(1-2*znum)*A+(1-2*xnum)*B)<1.e-5)[0]
        if len(inds) != 4**(Lcell-1):
            print()
            print("fuck:  ",inds)
            print()
        proj = vecs[:,inds]
        print(np.shape(proj))
        op = ss.csc_matrix(proj)
        ss.save_npz(workpath+"Proj_L{0}_x{1}_z{2}.npz".format(Lcell,xnum,znum),op)
    return op


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='xxz random unitary spectral form factor', prog='main', usage='%(prog)s [options]')
    parser.add_argument("-L", "--L", help="number of UNIT CELLS", default=5,type=int, nargs='?')
    parser.add_argument("-ndis", "--ndis", help="num disorder realizations", default=10,type=int, nargs='?')
    parser.add_argument("-B", "--B", help="local field dist. width", default=0.2,type=float, nargs='?')
    parser.add_argument("-K2", "--K2", help="intercell coupling dist. width", default=0.3,type=float, nargs='?')
    parser.add_argument("-K1", "--K1", help="intercell coupling dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-J", "--J", help="local random coupling dist. width", default=0.1,type=float, nargs='?')
    parser.add_argument("-pd", "--pd", help="phase difference, same as Brayden", default=0.0,type=float, nargs='?')
    parser.add_argument("-wid", "--wid", help="std dev for pulses (width)", default=0.05,type=float, nargs='?')
    parser.add_argument("-numterms", "--numterms", help="number of Fourier components (12 ought to suffice)", default=15,type=int, nargs='?')
    # parser.add_argument("-tf", "--tf", help="stop time", default=10000.0,type=int, nargs='?')
    # parser.add_argument("-maxint", "--maxint", help="maximum time interval", default=100.0,type=float, nargs='?')
    # parser.add_argument("-ev", "--ev", help="how often to record data", default=0.5, type=float, nargs='?')
    
    ### make another argument (later) for RSD offset? that way I don't have to do all realizations in a single job.
    
    ### load arguments
    params=parser.parse_args()
    #ti = float(params.maxint)/float(params.div) ## just using ti as a placeholder                                                 
    
    
    
    ### save an array with 25 Fourier terms of each type; for the sin^2 one, store the complex and real parts separately
    ### feed these arrays to the functions below so they don't recalculate the coefficients every fucking time
    ### also, save them as npy files since they only change with params.wid. 
    
    try:
        FourierX = np.load(workpath+"FourierX_wid{0}.npy".format(params.wid))
    except:
        FourierX = FT_Coeffs(omegaX,params.wid,params.numterms)
        np.save(workpath+"FourierX_wid{0}.npy".format(params.wid),FourierX)
    
    try:
        FourierZ = np.load(workpath+"FourierZ_wid{0}.npy".format(params.wid))
    except:
        FourierZ = FT_Coeffs(omegaZ,params.wid,params.numterms)
        np.save(workpath+"FourierZ_wid{0}.npy".format(params.wid),FourierZ)
    
    rs = []
    sectors = [(True,True),(True,False),(False,True),(False,False)]
    
    for rsd in range(params.ndis):
        Ks,Js,Bs = realization(params,rsd)
        try:
            D = ss.load_npz(workpath+"D0_"+jobhead(params,rsd)+".npz")
        except:
            D = D0(params,Ks,Js,FourierX,FourierZ)
            ss.save_npz(workpath+"D0_"+jobhead(params,rsd)+".npz",D) 
        
        try:
            D += ss.load_npz(workpath+"D1_"+jobhead(params,rsd)+".npz")
        except:
            D_1 = D1(params,Ks,Js,Bs,FourierX,FourierZ)
            ss.save_npz(workpath+"D1_"+jobhead(params,rsd)+".npz",D_1)
            D += D_1
        
        alleigs = lin.eigh(D.todense(),eigvals_only=True)
        print()
        print("eigenvalues look like:")
        print(np.reshape(alleigs,(int(len(alleigs)/4),4)))
        print()
        
        sym = Xsym(params)
        check = Sparse_Comm(sym,D).todense()
        print("checking commutator with X symmetry, [D,X] ")
        print("average value is {0} with std dev {1}".format(np.average(check),np.std(check)))
        print()
        
        sym = Zsym(params)
        check = Sparse_Comm(sym,D).todense()
        print("checking commutator with Z symmetry, [D,Z] ")
        print("average value is {0} with std dev {1}".format(np.average(check),np.std(check)))
        print()
        
        sym = Xop(params,0)
        check = Sparse_Comm(sym,D).todense()
        print("checking commutator with X at left edge [D,X_1] ")
        print("average value is {0} with std dev {1}".format(np.average(check),np.std(check)))
        print()
        
        sym = Zop(params,0)
        check = Sparse_Comm(sym,D).todense()
        print("checking commutator with Z at left edge [D,Z_1] ")
        print("average value is {0} with std dev {1}".format(np.average(check),np.std(check)))
        print()
        
        sym = Xop(params,2*params.L-1)
        check = Sparse_Comm(sym,D).todense()
        print("checking commutator with X at right edge [D,X_N] ")
        print("average value is {0} with std dev {1}".format(np.average(check),np.std(check)))
        print()
        
        sym = Zop(params,2*params.L-1)
        check = Sparse_Comm(sym,D).todense()
        print("checking commutator with Z at right edge [D,Z_N] ")
        print("average value is {0} with std dev {1}".format(np.average(check),np.std(check)))
        print()
        
        
        for sector in sectors:
            print("sector = {0}".format(list(np.where(sector)[0])))
            projector = proj(params,sector[0],sector[1])
            D_1 = (Sparse_Herm(projector)).dot(D.dot(projector))
            Darr = D_1.todense()
            print(np.shape(Darr))
            eigs0 = lin.eigh(Darr,eigvals_only=True)
            r = np.average(rvals(eigs0))
            print("r ratio for D1 is {0}".format(r))
            print()
            rs.append(r)
        
    rval = np.average(np.array(rs))
    print("disorder average r = {0}".format(rval))

        
        
                                                 
    