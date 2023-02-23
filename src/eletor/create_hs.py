from scipy.special import kv
import numpy as np
import math

try:
    from numba import njit
except ModuleNotFoundError:
    njit = lambda x: x

@njit
def mactrick(m,gamma_x):
    """ Use Cholesky decomposition to get first column of C

    Args:
        m (int) : length of time series
        gamma_x (float array) : first column of covariance matrix

    Returns:
        h : array of float with impulse response
    """

    U = np.zeros((m,2))  # C = U’*U
    V = np.zeros((m,2))
    h = np.zeros(m)

    #--- define the generators u and v
    U[:,0] = gamma_x/math.sqrt(gamma_x[0])
    V[1:m,0] = U[1:m,0]
    h[m-1] = U[m-1,0]

    k_old =0;
    k_new =1;
    for k in range(0,m-1):
        sin_theta = V[k+1,k_old]/U[k,k_old]
        cos_theta = math.sqrt(1.0-pow(sin_theta,2))
        U[k+1:m,k_new] = ( U[k:m-1,k_old] - sin_theta*V[k+1:m,k_old])/cos_theta
        V[k+1:m,k_new] = (-sin_theta*U[k:m-1,k_old] + V[k+1:m,k_old])/cos_theta
        h[m-1-k] = U[m-1,k_new]

        k_old = 1-k_old
        k_new = 1-k_new

    return h

def preprocess_params(plist, mname):
    def preprocessor(*args, **kwargs):
        args = list(args)
        try:
            for i in plist:
                if i not in kwargs: kwargs[i] = args.pop(0)
        except (IndexError, KeyError):
            raise TypeError(f"{mname} noise model requires arguments {plist}")
        return kwargs
    return preprocessor

_paramlist = {'White':('m','sigma'),
              'Powerlaw':('m','sigma','kappa','units','dt'),
              'Flicker':('m','sigma','units','dt'),
              'RandomWalk':('m','sigma','units','dt'),
              'GGM':('m','sigma','kappa','one_minus_phi','units','dt'),
              'VaryingAnnual':('m','sigma','phi','units','dt'),
              'Matern':('m','sigma','lamba','kappa'),
              'AR1':('m','sigma','phi')}

def create_h(noisemodel,*args,**kwargs):
    """ Create impulse function
    Args:
        noisemodel : model name, next arguments depends on this parameter:

        Model dependent parameters:
            White':('m','sigma'),
            Powerlaw':('m','sigma','kappa','units','dt'),
            Flicker':('m','sigma','units','dt'),
            RandomWalk':('m','sigma','units','dt'),
            GGM':('m','sigma','kappa','one_minus_phi','units','dt'),
            VaryingAnnual':('m','sigma','phi','units','dt'),
            Matern':('m','sigma','lamba','kappa'),
            AR1':('m','sigma','phi')

        m (int) : length of time series
        sigma (float) : noise variance
        kappa (float) : spectral index
        one_minus_phi (float) : (1-phi) for gauss markogv models
        phi (float) : phi for periodic models
        lamba (float) : lambda for mattern noise
        units (string) : 'mom' or 'msf' (for years or hours)
        dt (float) : sampling period in days
    Returns:
        sigma, h : noise amplitude + array of float with impulse response
    """

    try:
        preprocessor = preprocess_params(_paramlist[noisemodel], noisemodel)
    except KeyError:
        raise ValueError('Unknown noisemodel: {0:s}'.format(noisemodel))

    kwargs = preprocessor(*args,**kwargs)

    if noisemodel=='White':
        return kwargs.get('sigma'), \
                create_h_white(**kwargs)
    elif noisemodel == 'Powerlaw':
        kwargs['spectral_density'] = -kwargs['kappa']/2.0

        return gauss_markov_scale_variance(**kwargs),\
               create_h_Powerlaw(**kwargs)

    elif noisemodel == 'Flicker':
        kwargs['spectral_density'] = 0.5

        return gauss_markov_scale_variance(**kwargs),\
               create_h_Flicker(**kwargs)

    elif noisemodel == 'RandomWalk':
        kwargs['spectral_density'] = 1.0

        return gauss_markov_scale_variance(**kwargs),\
               create_h_RandomWalk(**kwargs)

    elif noisemodel == 'GGM':
        kwargs['spectral_density'] = -kwargs['kappa']/2.0

        return gauss_markov_scale_variance(**kwargs),\
               create_h_GGM(**kwargs)

    elif noisemodel=='VaryingAnnual':
        return kwargs.get('sigma'), \
               create_h_VaryingAnnual(**kwargs)

    elif noisemodel=='Matern':
        return kwargs.get('sigma'), \
               create_h_Matern(**kwargs)


    elif noisemodel=='AR1':
        return kwargs.get('sigma'), \
               create_h_AR1(**kwargs)

    assert False # Shouldn't get here

def create_h_white(m,sigma,**kwags):
    """
    Create impulse function for White noise.

    $h = [\\sigma, 0, 0, ...]$
    """

    #--- Array to fill impulse response
    h = np.zeros(m)

    h[0] = sigma

    return h

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
def recursion_GGM(m,d,one_minus_phi):
    """
    Recursion to create impulse function for Powerlay, Flicker or RW noise
    Flicker is Powerlaw with spectral density 0.5
    RandomWalk is Powerlaw with spectral density 1.0
    """

    h = np.zeros(m);h[0]=h0=1.0;

    for i in range(1,m):
        h[i] = (h0 := (d+i-1.0)/i * h0 * (1.0-one_minus_phi))

    return h

def gauss_markov_scale_variance(sigma,spectral_density,units,dt,**kwargs):
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

def create_h_Powerlaw(m,kappa,**kwargs):
    d = -kappa/2.0
    return recursion_Power_Flicker_RW(m,d)

def create_h_Flicker(m,**kwargs):
    d = 0.5
    return recursion_Power_Flicker_RW(m,d)

def create_h_RandomWalk(m,**kwargs):
    d = 1.0
    return recursion_Power_Flicker_RW(m,d)

def create_h_GGM(m,kappa,one_minus_phi,**kwargs):
    d = -kappa/2.0
    return recursion_GGM(m,d,one_minus_phi)

@njit
def recursion_VaryingAnnual(m,phi,dt):
    t = np.zeros(m)
    t[0] = ti = 1.0/(2.0 * (1.0 - phi*phi))
    for i in range(1,m):
        t[i] = (ti := ti*phi)

    t[1:] *= np.cos(2 * np.pi * np.arange(1,m) * dt/365.25) # TODO: hardcoded year!

    return t

def create_h_VaryingAnnual(m,phi,units,dt,**kwargs):
    if not units=='mom':
        raise ValueError('Please think again, ts_format is not mom!!!')

    t = recursion_VaryingAnnual(m,phi,dt)
    h = mactrick(m,t)

    return h


def create_h_Matern(m,lamba,kappa,**kwargs):

    threshold = 100.0

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

    return h

@njit
def recursion_AR1(m,phi):
    #--- Create first row vector of Covariance matrix
    t = np.zeros(m)

    t0 = t[0] = 1.0/(1.0 - phi*phi)
    for i in range(1,m):
        t[i] = (t0 := t0*phi)

    return t

def create_h_AR1(m,phi,**kwargs):

    t = recursion_AR1(m,phi)
    h = mactrick(m,t)

    return h

