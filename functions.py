import numpy as np
from scipy.integrate import quadrature
from astropy import constants as const
from astropy import units as u


def stable_deflection_angle(x, Rs, rm):
    # Avoid singularities by using np.where to handle zero-denominator cases
    denominator = np.sqrt(1 - x**2 - ((Rs / rm) * (1 - x**3)))
    return np.where(denominator != 0, 1 / denominator, 0)

# Function to compute the deflection angle using Gauss-Kronrod Quadrature
def exact_deflection_angle_stable_gk(rm, Rs):
    # Perform the integration using quadrature, which adapts to the function's behavior
    result, _ = quadrature(stable_deflection_angle, 0, 1, args=(Rs, rm), tol=1e-30, rtol=1e-30, maxiter=10000)
    return result


def magnification(t, t0, tE, u0):
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))

def tterm(M,DL,DS,v,t0,t):
    tE=EinsteinCrossTime(M,DL,DS,v).value
    return ((t-t0)/tE)

def shift_par(M,DL,DS,v,y0,t0,t):
    tt=tterm(M,DL,DS,v,t0,t)
    yy=np.sqrt(y0**2+tt**2)
    return(tt/(yy**2+2))

def shift_per(M,DL,DS,v,y0,t0,t):
    tt=tterm(M,DL,DS,v,t0,t)
    yy=np.sqrt(y0**2+tt**2)
    return(y0/(yy**2+2))

def EinsteinCrossTime(M,DL,DS,v):
    theta_e=theta_e_func(M,DL,DS)
    return(((theta_e.to('radian').value*DL*u.kpc).to('km')/v/u.km*u.s).to('day'))

def theta_e_func(M,DL,DS):
    mass=M*const.M_sun#.value
    G=const.G#.value
    c=c=const.c#.value
    aconv=180.0*3600.0/np.pi*u.arcsecond
    return((np.sqrt(4.0*(G*mass/c/c).to('kpc')*(DS-DL)/DL/DS/u.kpc))*aconv)
