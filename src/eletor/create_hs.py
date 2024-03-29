from scipy.special import kv
import numpy as np
import math

from eletor.helper import mactrick

try:
    from numba import njit
except ModuleNotFoundError:
    njit = lambda x: x

def White(
        *,
        m,
        sigma,
        **kwags
    ):
    """
    Create impulse function for White noise.

    $h = [\\sigma, 0, 0, ...]$
    """

    #--- Array to fill impulse response
    h = np.zeros(m)

    h[0] = 1 #sigma

    return sigma, h

@njit
def recursion_Power_Flicker_RW(m,d):
    """
    Recursion to create impulse function for Powerlay, Flicker or RW noise
    Flicker is Powerlaw with spectral density 0.5
    RandomWalk is Powerlaw with spectral density 1.0
    """

    h = np.zeros(m);h[0]=h0=1.0;

    for i in range(1,m):
        h[i] = (h0 := (d+i-1.0)/i * h0)

    return h

@njit
def recursion_GGM(
        *,
        m,
        d,
        one_minus_phi
    ):
    """
    Recursion to create impulse function for Powerlay, Flicker or RW noise
    Flicker is Powerlaw with spectral density 0.5
    RandomWalk is Powerlaw with spectral density 1.0
    """

    h = np.zeros(m);h[0]=h0=1.0;

    for i in range(1,m):
        h[i] = (h0 := (d+i-1.0)/i * h0 * (1.0-one_minus_phi))

    return h

def gauss_markov_scale_variance(
        *,
        sigma,
        spectral_density,
        units,
        dt,
    ):
    """
    Gauss Markov Models needs scaling of the variance for taking into account
    the time and period units.

    This scale applies to Powerlaw noise, Flicker Noise, Random Walk Noise and
    Generalized Gauss Markov noise.
    """

    if units=='mom':
        sigma *= math.pow(dt/365.25,0.5*spectral_density)  # Already adjust for scaling
    elif units=='msf':
        sigma *= math.pow(dt/3600.0,0.5*spectral_density)  # Already adjust for scaling
    else:
        raise ValueError('unknown scaling: {0:s}'.format(units))

    return sigma

def Powerlaw(
        *,
        m,
        kappa,
        sigma,
        units,
        dt,
        **kwargs
    ):
    d = -kappa/2.0
    gmsv = gauss_markov_scale_variance(sigma=sigma,spectral_density=d,units=units,dt=dt)
    return gmsv, recursion_Power_Flicker_RW(m,d)

def Flicker(
        *,
        m,
        sigma,
        units,
        dt,
        **kwargs
    ):
    d = 0.5
    gmsv = gauss_markov_scale_variance(sigma=sigma,spectral_density=d,units=units,dt=dt)
    return gmsv, recursion_Power_Flicker_RW(m,d)

def RandomWalk(
        *,
        m,
        sigma,
        units,
        dt,
        **kwargs):
    d = 1.0
    gmsv = gauss_markov_scale_variance(sigma=sigma,spectral_density=d,units=units,dt=dt)
    return gmsv, recursion_Power_Flicker_RW(m,d)

#def GGM(m,kappa,one_minus_phi,**kwargs):
## El asterisco es para que los parametros que se le pasan a la funcion sean nombrados
def GGM(
        *,
        m,
        kappa,
        one_minus_phi,
        sigma,
        units,
        dt,
        **kwargs
    ):
    d = -kappa/2.0
    gmsv = gauss_markov_scale_variance(sigma=sigma,spectral_density=d,units=units,dt=dt)
    r_GGM = recursion_GGM(m=m,d=d,one_minus_phi=one_minus_phi)
    return gmsv, r_GGM


@njit
def recursion_VaryingAnnual(m,phi,dt):
    t = np.zeros(m)
    t[0] = ti = 1.0/(2.0 * (1.0 - phi*phi))
    for i in range(1,m):
        t[i] = (ti := ti*phi)

    t[1:] *= np.cos(2 * np.pi * np.arange(1,m) * dt/365.25) # TODO: hardcoded year!

    return t

def VaryingAnnual(
        *,
        m,
        phi,
        units,
        dt,
        sigma,
        **kwargs
    ):
    if not units=='mom':
        raise ValueError('Please think again, ts_format is not mom!!!')

    t = recursion_VaryingAnnual(m,phi,dt)
    h = mactrick(m,t)

    return sigma, h


def Matern(
        *,
        m,
        lamba,
        kappa,
        sigma,
        **kwargs):

    threshold = 100.0
    print("-------> ", kappa)
    d = -0.5*kappa

    #--- create gamma_x
    t = np.zeros(m)

    #--- Constant
    alpha = 2.0*d
    c0 = 2.0/(math.gamma(alpha-0.5) * pow(2.0, alpha-0.5))

    #--- Use Modified Bessel Function, second order
    t[0] = 1.0 #-- check Eq. (62) and set tau=0
    i = 1

    tau = np.arange(0,m)

    trhix = int(threshold/lamba)

    # --- Fill values until Threshold with Eq. (60)
    t[1:trhix] = c0 * np.power(lamba*tau[1:trhix],alpha-0.5) * \
                     kv(alpha-0.5,lamba*tau[1:trhix])

    # --- Fill values from Threshold on with Eq. (61)
    t[trhix:] = c0*np.sqrt(math.pi/2.0) * np.power(lamba*tau[trhix:],alpha-0.5)\
                                           * np.exp(-lamba*tau[trhix:])

    h = mactrick(m,t)

    return sigma, h

@njit
def recursion_AR1(m,phi):
    #--- Create first row vector of Covariance matrix
    t = np.zeros(m)

    t0 = t[0] = 1.0/(1.0 - phi*phi)
    for i in range(1,m):
        t[i] = (t0 := t0*phi)

    return t

def AR1(
        *,
        m,
        phi,
        sigma,
        **kwargs
    ):

    t = recursion_AR1(m,phi)
    h = mactrick(m,t)

    return sigma, h

