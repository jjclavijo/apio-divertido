from scipy.special import kv
import numpy as np
import math

from eletor.helper import mactrick

try:
    from numba import njit
except ModuleNotFoundError:
    njit = lambda x: x

#@njit
#def mactrick(m,gamma_x):
#    """ Use Cholesky decomposition to get first column of C
#
#    Args:
#        m (int) : length of time series
#        gamma_x (float array) : first column of covariance matrix
#
#    Returns:
#        h : array of float with impulse response
#    """
#
#    U = np.zeros((m,2))  # C = U’*U
#    V = np.zeros((m,2))
#    h = np.zeros(m)
#
#    #--- define the generators u and v
#    U[:,0] = gamma_x/math.sqrt(gamma_x[0])
#    V[1:m,0] = U[1:m,0]
#    h[m-1] = U[m-1,0]
#
#    k_old =0;
#    k_new =1;
#    for k in range(0,m-1):
#        sin_theta = V[k+1,k_old]/U[k,k_old]
#        cos_theta = math.sqrt(1.0-pow(sin_theta,2))
#        U[k+1:m,k_new] = ( U[k:m-1,k_old] - sin_theta*V[k+1:m,k_old])/cos_theta
#        V[k+1:m,k_new] = (-sin_theta*U[k:m-1,k_old] + V[k+1:m,k_old])/cos_theta
#        h[m-1-k] = U[m-1,k_new]
#
#        k_old = 1-k_old
#        k_new = 1-k_new
#
#    return h

def create_h_white(
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

    h[0] = sigma

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

def create_h_Powerlaw(
        *,
        m,
        kappa,
        sigma, 
        spectral_density,
        units,
        dt,
        **kwargs
    ):
    d = -kappa/2.0
    gmsv = gauss_markov_scale_variance(sigma=sigma,spectral_density=d,units=units,dt=dt)
    return gmsv, recursion_Power_Flicker_RW(m,d)

def create_h_Flicker(
        *,
        m,
        sigma,
        spectral_density,
        units,
        dt,
        **kwargs
    ):
    d = 0.5
    gmsv = gauss_markov_scale_variance(sigma=sigma,spectral_density=d,units=units,dt=dt)
    return recursion_Power_Flicker_RW(m,d)

def create_h_RandomWalk(
        *,
        m,
        sigma,
        spectral_density,
        units,
        dt,
        **kwargs):
    d = 1.0
    gmsv = gauss_markov_scale_variance(sigma=sigma,spectral_density=d,units=units,dt=dt)
    return recursion_Power_Flicker_RW(m,d)

#def create_h_GGM(m,kappa,one_minus_phi,**kwargs):
## El asterisco es para que los parametros que se le pasan a la funcion sean nombrados
def create_h_GGM(
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

def create_h_VaryingAnnual(
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


def create_h_Matern(
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

def create_h_AR1(
        *,
        m,
        phi,
        sigma,
        **kwargs
    ):

    t = recursion_AR1(m,phi)
    h = mactrick(m,t)

    return sigma, h

