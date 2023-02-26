# -*- coding: utf-8 -*-
#
# This program creates synthetic noise.
#
#  This script is part of HectorP 0.1.1
#
#  HectorP is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  any later version.
#
#  HectorP is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with HectorP. If not, see <http://www.gnu.org/licenses/>
#
# 6/2/2022 Machiel Bos, Santa Clara
#===============================================================================

import os
import math
import time
import sys
import numpy as np
import argparse
from eletor.control import Control
from eletor.observations import Observations
from scipy import signal
from pathlib import Path
from scipy.special import kv

from eletor import create_hs
from eletor.helper import mactrick
#===============================================================================
# Subroutines
#===============================================================================



#===============================================================================
# Control parameter verification functions.
#===============================================================================
"""
These are functions meant to be run just after initialization of the
Control instance.
Each function checks and ask for correction of errors in the parameter structure
"""

# The order of parameter reading form commandline in the original version is:
# TODO: Keep this order for compatibility.

class mustAsk(object):
    def __repr__(self):
        return 'MustAsk'
    def __str__(self):
        return 'MustAsk'

MustAsk = mustAsk() # A marker for need to ask for parameter.

_paramorder =( ('White',('Sigma')),
               ('Powerlaw',('Kappa','Sigma')),
               ('Flicker',('Sigma')),
               ('RandomWalk',('Sigma')),
               ('GGM',('GGM_1mphi','Kappa','Sigma')),
               ('VaryingAnnual',('Phi','Sigma')),
               ('Matern',('Lambda','Kappa','Sigma')),
               ('AR1',('Phi','Sigma')) )

def check_Stdin_params(control):
    modelorder = []
    for model,_ in _paramorder:
        modelorder.extend(\
             [ix for ix, v in enumerate(control['NoiseModels']) if v == model]\
             )
    for ix in modelorder:
        model = control['NoiseModels'][ix]
        for parameter in {i:j for i,j in _paramorder}[model]:
            print(parameter,control[parameter])
            if control[parameter] is MustAsk:
                print(f'Please Specify {parameter} for model type <{model}> : ', end='')
                parvalue = float(input())
                control[parameter] = parvalue
                continue
            try:
                if control[parameter][ix] is MustAsk:
                    print(f'Please Specify {parameter} for model <{ix}:{model}> : ', end='')
                    parvalue = float(input())
                    control[parameter][ix] = parvalue
                    continue
            except TypeError as t:
                assert 'subscriptable' in t.args[0] #only get here if control[...] is not a list

    return None


def check_MissingData(control):
    """
    if incomplete MissingData specification
    no missing data is used
    """
    # log.warning('Missing data options misspesified, ignoring')
    if 'MissingData' in control.params and\
            'PercMissingData' in control.params:
        pass
    else:
        control.params['MissingData'] = False
        control.params['PercMissingData'] = 0.0

def check_trend_bias(control):
    """
    if incomplete linear terms specification
    no linear term is used
    """
    # log.warning('Linear terms options misspesified, ignoring')
    if 'Trend' in control.params and\
            'NominalBias' in control.params:
        pass
    else:
        control.params['Trend'] = 0.0
        control.params['NominalBias'] = 0.0


def check_sigma(control):
    """
    check noise amplitude for any model
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        try:
            sigma = control.get('Sigma')[ix]
        except ValueError:
            control['Sigma'] = [None] * len(control.get('NoiseModels'))
            sigma = None
        except IndexError:
            raise ValueError(f'Sigma parameter list has wrong length, should be at least {ix}')

        if sigma is None:
            if control.unatended:
                raise ValueError(f'Should specify noise Amplitude Sigma for noise model {nm}')

            # print(f'Please Specify noise Amplitude Sigma for model <{ix}:{nm}> : ', end='')
            # sigma = float(input())
            control.get('Sigma')[ix] = MustAsk

def check_kappa(control):
    """
    check kappa parameter for GGM, Powerlaw or Matern Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm in ['Powerlaw', 'GGM', 'Matern']:
            try:
                kappa = control.get('Kappa')[ix]
            except ValueError:
                control['Kappa'] = [None] * len(control.get('NoiseModels'))
                kappa = None
            except IndexError:
                raise ValueError(f'Kappa parameter list has wrong length, should be at least {ix}')

            if nm == 'Matern':
                try:
                    kappa = control.get('kappa_fixed')
                    control.get('Kappa')[ix] = kappa
                except ValueError:
                    pass

            if kappa is None:
                if control.unatended:
                    raise ValueError(f'Should specify Spectral index kappa for noise model {nm}')

                # print(f'Please Specify Spectral index kappa for model <{ix}:{nm}> : ', end='')
                # kappa = float(input())
                control.get('Kappa')[ix] = MustAsk

            else:
                if (nm in ['Powerlaw', 'GGM']) and (kappa<-2.0-EPS or kappa>2.0+EPS):
                    raise ValueError('kappa shoud lie between -2 and 2 : {0:f}'.format(kappa))
                elif nm == 'Matern' and (kappa > -0.5):
                    # The cited paper on Matern processes limits alfa to > 0.5,
                    # implying this.
                    raise ValueError('kappa shoud be lower than -0.5 : {0:f}'.format(kappa))

def check_lambda(control):
    """
    check kappa parameter for Matern Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm == 'Matern':
            try:
                lamba = control.get('Lambda')[ix]
            except ValueError:
                control['Lambda'] = [None] * len(control.get('NoiseModels'))
                lamba = None
            except IndexError:
                raise ValueError(f'Lambda parameter list has wrong length, should be at least {ix}')

            try:
                lamba = control.get('lambda_fixed')
                control.get('Lambda')[ix] = lamba
            except ValueError:
                pass

            if lamba is None:
                if control.unatended:
                    raise ValueError(f'Should specify Lambda parameter for noise model {nm}')

                # print(f'Please Specify Lambda Parameter for model <{ix}:{nm}> : ', end='')
                # lamba = float(input())
                control.get('Lambda')[ix] = MustAsk
            elif (lamba < EPS):
                # The cited paper on Matern processes limits lambda to be
                # positive
                raise ValueError(f'Lambda should be positive : {lamba:f}')

def check_phi(control):
    """
    check Phi parameter for AR1 or VaryingAnual Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm in ['AR1','VaryingAnual']:
            try:
                phi = control.get('Phi')[ix]
            except ValueError:
                    control['Phi'] = [None] * len(control.get('NoiseModels'))
                    phi = None
            except IndexError:
                if isinstance( (phi := control.get('Phi'))
                              ,float):
                    control['Phi'] = [phi] * len(control.get('NoiseModels'))
                else:
                    raise ValueError(f'Phi parameter list has wrong length, should be at least {ix}')

            try:
                phi = control.get('phi_fixed')
                control.get('Phi')[ix] = lamba
            except ValueError:
                pass

            if phi is None:
                if control.unatended:
                    raise ValueError(f'Should specify Phi parameter for noise model {nm}')

                #print(f'Please Specify Phi Parameter for model <{ix}:{nm}> : ', end='')
                #phi = float(input())
                control.get('Phi')[ix] = MustAsk


def check_ggmphi(control):
    """
    check 1-phi parameter for GGM Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm == 'GGM':
            try:
                phi = control.get('GGM_1mphi')[ix]
            except ValueError:
                    control['GGM_1mphi'] = [None] * len(control.get('NoiseModels'))
                    phi = None
            except (IndexError, TypeError):
                if isinstance( (phi := control.get('GGM_1mphi'))
                              ,float):
                    control['GGM_1mphi'] = [phi] * len(control.get('NoiseModels'))
                else:
                    raise ValueError(f'GGM 1-phi parameter list has wrong length, should be at least {ix}')

            if phi is None:
                if control.unatended:
                    raise ValueError(f'Should specify 1-phi parameter for noise model {nm}')

                # print(f'Please Specify 1-phi Parameter for model <{ix}:{nm}> : ', end='')
                # phi = float(input())
                control.get('GGM_1mphi')[ix] = MustAsk

            elif phi<0.0 or phi>1.0+EPS:
                # The cited paper on Matern processes limits lambda to be
                # positive
                raise ValueError(f'1-phi should lie between 0 and 1: : {phi:f}')

def check_NoiseModel_islist(control):
    """
    Ensure NoiseModels is a list
    """
    if not isinstance(nm := control.get('NoiseModels'),list):
        control['NoiseModels'] = [nm]


control_defaults = {
        'SimulationDir':None,
        'SimulationLabel':None,
        'TS_format':'mom',
        'NumberOfSimulations':1,
        'NumberOfPoints':365,
        'SamplingPeriod':1,
        'TimeNoiseStart':0,
        'NoiseModels':None,
        'RepeatableNoise':False,
        'MissingData':False,
        'PercMissingData':0.0,
        'Offsets':False,
        'Trend':0,
        'NominalBias':0,
        'AnnualSignal':0}

hooks = [
        check_MissingData,
        check_trend_bias,
        check_NoiseModel_islist,
        check_lambda,
        check_kappa,
        check_sigma,
        check_ggmphi,
        check_phi,
        check_Stdin_params,
        check_lambda, # repeat boundary checks.
        check_kappa, # repeat boundary checks.
        check_sigma, # repeat boundary checks.
        check_ggmphi, # repeat boundary checks.
        check_phi # repeat boundary checks.
         ]

def simulatenoiseControl(fname,*args,**kwargs):
    if not "hooks" in kwargs:
        kwargs["hooks"] = hooks
    return Control(fname,control_defaults,args,**kwargs)

def simulate_noise(control,observations):

    #--- Some variables that define the runs
    directory     = Path(control.get('SimulationDir'))
    label         = control.get("SimulationLabel")
    n_simulations = control.get("NumberOfSimulations")
    m             = control.get("NumberOfPoints")
    dt            = control.get("SamplingPeriod")
    ms            = control.get("TimeNoiseStart")

    repeatablenoise = control.get('RepeatableNoise')

    #--- Start the clock!
    start_time = time.time()

    #--- Create random number generator
    rng = np.random.default_rng(0) if repeatablenoise else None

    #--- Does the directory exists?
    if not os.path.exists(directory):
       os.makedirs(directory)

    #--- Already create time array
    if observations.datafile=='None' and observations.ts_format=='mom':
        t0 = 51544.0
        t = np.linspace(t0,t0+m*dt,m,endpoint=False)
        # t = np.zeros(m);
        # t[0] = 51544.0 # 1 January 2000
        # for i in range(1,m):
        #     t[i] = t[i-1] + dt
    elif observations.datafile=='None' and observations.ts_format=='msf':
        t = np.linspace(0,m*dt,m,endpoint=False)
        # t = np.zeros(m);
        # for i in range(1,m):
        #     t[i] = t[i-1] + dt
    else:
        try:
            t = observations.data.index.values
            dt = observations.sampling_period
        except AttributeError:
            raise ValueError('Indexing information not provided')

    #--- Run all simulations
    for k in range(0,n_simulations):

        #--- Open file to store time-series
        datafile = label + '_' + str(k) + "." + control['TS_format']
        fname = str(directory.resolve()) + '/' + datafile

        #--- Create deterministic signal
        y = create_trend(control,t)

        #--- Create the synthetic noise
        y += create_noise_(control,rng)

        #--- convert this into Panda dataframe
        observations.create_dataframe_and_F(t,y,[],dt)

        #--- write results to file
        observations.write(fname)

    #--- Show time lapsed
    print("--- {0:8.3f} seconds ---\n".format(float(time.time() - start_time)))

#===============================================================================
# Signal Creation Functions
#===============================================================================

_paramlist = {'White':('NumberOfPoints','Sigma'),
              'Powerlaw':('NumberOfPoints','Sigma','Kappa','TS_format','SamplingPeriod'),
              'Flicker':('NumberOfPoints','Sigma','TS_format','SamplingPeriod'),
              'RandomWalk':('NumberOfPoints','Sigma','TS_format','SamplingPeriod'),
              'GGM':('NumberOfPoints','Sigma','Kappa','GGM_1mphi','TS_format','SamplingPeriod'),
              'VaryingAnnual':('NumberOfPoints','Sigma','Phi','TS_format','SamplingPeriod'),
              'Matern':('NumberOfPoints','Sigma','Lambda','Kappa'),
              'AR1':('NumberOfPoints','Sigma','Phi')}

def get_nm_parameter(
        ix,
        model,
        control,
        parameter
    ):
    if parameter == 'NumberOfPoints':
        return control['NumberOfPoints']
    if parameter == 'Sigma':
        return control['Sigma'][ix]
    if parameter == 'Kappa':
        if model == 'Matern':
            try:
                return control['kappa_fixed']
            except ValueError:
                pass
        return control['Kappa'][ix]
    if parameter == 'TS_format':
        return control['TS_format']
    if parameter == 'SamplingPeriod':
        return control['SamplingPeriod']
    if parameter == 'GGM_1mphi':
        if isinstance((ggm := control['GGM_1mphi']),list):
            return ggm[ix]
        else:
            return ggm
    if parameter == 'Phi':
        if model == 'VaryingAnnual':
            try:
                return control['phi_fixed']
            except ValueError:
                pass
        return control['Phi'][ix]
    if parameter == 'Lambda':
        if model == 'Matern':
            try:
                return control['lambda_fixed']
            except ValueError:
                pass
        return control['Lambda'][ix]

    raise ValueError(f'Incorrect parameter {parameter} for required model {model}')

def create_noise_(
        control,
        rng=None
    ):
    """ Configure and create noise from the config file
    Args:
        control: dict
            Dictionary with the control parameters, content raw
        rng: np.random.Generator or None
            ??
    Returns:
        noise: np.ndarray
            Array with the noise
    TODO: Nombre de las variables podrian ser mas decriptivos
    TODO: los aprametros generales como n_simulations, m, dt, ms, deberian ser pasados como argumentos
    TODO: los modelos y sus configuraciones deberian ser pasados como un diccionario/argumento aparte
    """

    n_simulations = control['general']["NumberOfSimulations"]
    m             = control['general']["NumberOfPoints"]
    dt            = control['general']["SamplingPeriod"]
    ms            = control['general']["TimeNoiseStart"]
    
    noiseModels   = control['NoiseModels']
    return create_noise(m,dt,ms,noiseModels,rng)

def create_noise(m,dt,ms,noiseModels,rng=None):
    """ Toma la lista de los modelos que quiere usar junto con los parametros y isntancia los nuevos modelos desde create_H
    """

    sigma = []
    h = []
    for model, params in noiseModels.items():
        print(f"--> About to run model {model} with params {params['params']}")

        ## Dinamycally get and call the function using the model name
        ## TODO capitalize/or not all function names
        model_function = getattr(create_hs, f"create_h_{model}")
        single_s, single_h = model_function(**params['params'])

        sigma.append(single_s)
        h.append(single_h)

    #--- Create random number generator
    if rng is None:
        rng = np.random.default_rng()

    #--- Create the synthetic noise
    y = np.zeros(m)

    for s,ha in zip(sigma,h):
        print(s, ha)
        w = s * rng.standard_normal(m+ms)
        y += signal.fftconvolve(ha, w)[0:m]

    return y

def create_trend(control,times,
                 dsignal_dt=365.25):
    """
    dsignal_dt: if signal units are not dt units.
                by default dt is in units of days and
                trends are expressed in units of years
                also periodic signals are expressed in
                units of years
    """
    bias = control['NominalBias']
    trend = control['Trend']
    annualAmplitude = control['AnnualSignal']

    if not isinstance(times,np.ndarray):
        times = np.array(times)

    y = bias + trend * (times - times[0])/dsignal_dt

    if abs(annualAmplitude) > 0:
        y += annualAmplitude * np.cos(2*np.pi*(times - times[0])/dsignal_dt)
        # TODO: why does the annual signal have to start alongside the noise?

    return y

#===============================================================================
# Main program
#===============================================================================

def main():

    print("\n***************************************")
    print("    simulatenoise, version 0.1.1")
    print("***************************************")

    #--- Parse command line arguments in a bit more professional way
    parser = argparse.ArgumentParser(description= 'Simulate noise time series')

    #--- List arguments that can be given
    parser.add_argument('-i', required=False, default='simulatenoise.ctl', \
                                      dest='fname', help='Name of control file')

    args = parser.parse_args()

    #--- parse command-line arguments
    fname = args.fname

    #--- Read control parameters into dictionary (singleton class)
    try:
        control = simulatenoiseControl(fname)
    except (FileNotFoundError, ValueError) as e:
        if isinstance(e,FileNotFoundError):
            print(f'Control file {fname} not found')
            return 2 # Exit with errorcode 2: file not found
        else:
            print(f'Error parsing {fname}:',e)
            return 1 # Exit with: Operation not permitted

    observations = Observations()

    simulate_noise(control,observations)

if __name__ == "__main__":
    exit(main())
