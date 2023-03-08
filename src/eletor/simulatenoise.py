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
import toml
from scipy import signal
from pathlib import Path
from scipy.special import kv

#from eletor.control import Control, parse_retro_ctl
#from eletor.observations import Observations, momwrite
from eletor.compat import momwrite

from eletor import create_hs
from eletor.helper import mactrick
#===============================================================================
# Subroutines
#===============================================================================

def simulate_noise(control):

    #--- Some variables that define the runs
    ## Unpack all variables from the control object
    directory     = Path(control['file_config'].get("SimulationDir",''))
    label         = control['file_config'].get("SimulationLabel",'')
    n_simulations = control['general'].get("NumberOfSimulations",1)
    m             = control['general'].get("NumberOfPoints")
    dt            = control['general'].get("SamplingPeriod")
    ms            = control['general'].get("TimeNoiseStart",0)
    noiseModels   = control['NoiseModels']

    repeatablenoise = control['general'].get('RepeatableNoise',False)
    deterministic_noise = control['general'].get('DeterministicNoise',False)
    time_format = control['file_config'].get('TS_format','mom')

    #--- Start the clock!
    start_time = time.time()

    #--- Create random number generator
    rng = np.random.default_rng(0) if repeatablenoise else None

    ## For testing purposes
    if deterministic_noise:
        from eletor.not_rng import rng

    #--- Does the directory exists?
    if not os.path.exists(directory):
       os.makedirs(directory)

    #--- Create time array. Default time format is mom
    # Dejo el if-else para que nos acordemos que esto tiene que
    # Cambiar.
    if time_format == 'mom':
        t0 = 51544.0
        t = np.linspace(t0,t0+m*dt,m,endpoint=False)
    else:
        raise NotImplementedError

    # En algún momento a alguien le pareció útil pensar que el índice de
    # tiempos había que tener la posibilidad de leerlo de un archivo de datos.
    # No veo que tenga mucho sentido ahora, pero lo anoto.

    #--- Run all simulations
    for k in range(0,n_simulations):

        #--- Open file to store time-series
        datafile = label + '_' + str(k) + "." + time_format
        fname = str(directory.resolve()) + '/' + datafile

        #--- Create the synthetic noise
        #y += create_noise_(control,rng)
        y += create_noise(m,dt,ms,noiseModels,rng)

        #--- TODO: not always write mom files
        momwrite(y,t,dt,fname)

    #--- Show time lapsed
    print("--- {0:8.3f} seconds ---\n".format(float(time.time() - start_time)))

#===============================================================================
# Signal Creation Functions
#===============================================================================

# TODO: probablemente esto se tenga que mudar o eliminar, creo que en algún
# lado se usa, pero queda como documentación sino.
_paramlist = {'White':('NumberOfPoints','Sigma'),
              'Powerlaw':('NumberOfPoints','Sigma','Kappa','TS_format','SamplingPeriod'),
              'Flicker':('NumberOfPoints','Sigma','TS_format','SamplingPeriod'),
              'RandomWalk':('NumberOfPoints','Sigma','TS_format','SamplingPeriod'),
              'GGM':('NumberOfPoints','Sigma','Kappa','GGM_1mphi','TS_format','SamplingPeriod'),
              'VaryingAnnual':('NumberOfPoints','Sigma','Phi','TS_format','SamplingPeriod'),
              'Matern':('NumberOfPoints','Sigma','Lambda','Kappa'),
              'AR1':('NumberOfPoints','Sigma','Phi')}

def create_noise(
        m,
        dt,
        ms,
        noiseModels,
        rng=None
    ):
    """ Toma la lista de los modelos que quiere usar junto con los parametros y
    instancia los nuevos modelos desde create_h
    """
    print("Params de create_noise: ", m, dt, ms, noiseModels, rng)
    sigma = []
    h = []
    for model, params in noiseModels.items():
        ## Dinamycally get and call the function using the model name
        model_function = getattr(create_hs, model)

        for i in params.keys():
            ## Cada modelo puede estar mas de una vez
            print(f"--> About to run model {model} with params {params[i]}")
            single_s, single_h = model_function(**params[i])
            sigma.append(single_s)
            h.append(single_h)

    #--- Create random number generator
    if rng is None:
        rng = np.random.default_rng()

    #--- Create the synthetic noise
    y = np.zeros(m)

    for s,ha in zip(sigma,h):
        w = s * rng.standard_normal(m+ms)
        y += signal.fftconvolve(ha, w)[0:m]

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
    fname = Path(args.fname)

    #--- Read control parameters into dictionary
    try:
        if fname.suffix == '.ctl':
            #control = parse_retro_ctl(fname)
            raise TypeError('.ctl files not supported: convert to toml using converion utility.')
        else:
            control = toml.load(fname)

    except FileNotFoundError:
        print(f'Control file {fname} not found')
        return 2 # Exit with errorcode 2: file not found
    except (toml.TomlDecodeError):
        print(f'Error parsing {fname}:',e)
        return 1 # Exit with: Operation not permitted
    except TypeError:
        print(f'Invalid file specification: {fname}')
        return 2 # Exit with errorcode 2: file not found

    simulate_noise(control)

if __name__ == "__main__":
    exit(main())
