import random
import numpy as np

import eletor.simulatenoise as hps # fork version.

# --- CONSTANTS ---

UNITS = 'mom'
DT = 1
MODELNAMES = ["Powerlaw","GGM","VaryingAnnual","Matern","AR1"]

# --- Parameter samplers.
# For each model, return functions that samples valid parameters for the
# noise distribution.
# Note that Kappa in GGM models can lie between -2 and 2, but negative
# Values produces very unlikely timeseries for geophisics.
# TODO: add published sources for the parameter limits.

models = {"Powerlaw":[
    lambda: np.random.uniform(-2,2),
    lambda:UNITS,
    lambda: DT],
          "GGM":[
    lambda: 10**np.random.uniform(-5,0),
    lambda: np.random.uniform(0,2), #Kappa < 0 diverges
    lambda:UNITS,
    lambda: DT],
          "VaryingAnnual":[
    lambda: 10**np.random.uniform(-5,0),
    lambda:UNITS,
    lambda: DT],
          "Matern":[
    lambda: np.random.uniform(1,10)*10**np.random.uniform(-3,-1),
    lambda: np.random.uniform(-2,-0.5)],
          "AR1":[
    lambda: 10**np.random.uniform(-5,0)]
          }

# default Samplers for other parameters

# Noise amplitude (combined)
sigma_sampler = lambda: random.random()*(0.005-0.03)+0.005

# Trayectory signals amplitudes (linear, annual and semiannual)
ts = lambda : np.random.normal(0,0.01)
ans = lambda : np.random.normal(0,0.01)
sas = lambda : np.random.normal(0,0.01)

# Parameters for exponential decay. Given the ammount os displacement of an
# earthquake, use a percent of that value and sample a decay
percent_sampler = lambda: np.random.uniform(0.05,0.5)* \
        np.sign(np.random.uniform(-1,1))

# Example of short decay
decay_sampler = lambda: np.random.uniform(1,10)

def sample_noise(m,sigma = sigma_sampler):
    """
    Sample noise combined form a random number of the available models.
    The parameters of each model are sampled at random too.
    """
    if callable(sigma):
        sigma = sigma()

    params = []
    nmodels = random.randrange(5)+1

    amounts = np.random.rand(nmodels)
    amounts *= 1/sum(amounts)

    sigmas = sigma * amounts

    modelorder = MODELNAMES.copy()
    random.shuffle(modelorder)

    for model,sigma in zip(modelorder[:nmodels],sigmas):
        pars = [model,m,sigma,*[f() for f in models[model]]]
        params.append(pars)

    return hps.create_noise(m,DT,0,params)

def sample_tray(m,trend = ts,annual = ans,sannual = sas):
    """
    Build the deterministic signal composed by a linear trend, annual and
    semi-annual signals.
    """

    if callable(trend):
        trend = trend()
    if callable(annual):
        annual = annual()
    if callable(sannual):
        sannual = sannual()

    time = np.arange(m)
    trend = time * trend / 365.25

    phase = np.random.uniform(-np.pi,np.pi)
    an = annual * np.cos( time * 2 * np.pi / 365.25 + phase )

    phase = np.random.uniform(-np.pi,np.pi)
    sa = sannual * np.cos( time * 4 * np.pi / 365.25 + phase )

    return trend+an+sa

def sample_post_seismic(m,m0,step_ammount,percent = percent_sampler,decay =
                        decay_sampler,prob_exp = 0.5):
    """
    Get post-seismic term.
    The model (log or exp) is selected at random with exponential taking
    prob_exp and logaritmic 1-prob_exp
    """

    if callable(percent):
        percent = percent()
    if callable(decay):
        decay = decay()

    signal = np.zeros(m)

    tau = decay
    jump = percent*step_ammount

    if np.random.rand() > prob_exp:
        #--- Post-seismic logarithmic relaxation
        signal2 = np.log(1 + np.arange(1,m-m0+1)/tau)
    else:
        #--- Post-seismic logarithmic relaxation
        signal2 = 1 - np.exp(-np.arange(1,m-m0+1)/(10*tau))

    signal[m0:] = jump * signal2

    return signal

def sample_step(m,m0,ammount):
    """
    produce the step signal.
    """
    signal = np.ones(m) * ammount
    signal[:m0] = np.zeros(m0)
    signal -= ammount/2
    signal[m0] = 0
    return signal
