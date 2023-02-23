#!/usr/bin/env python3

import io
import os

from unittest import mock
import pytest

import logging

LOGGER = logging.getLogger(__name__)

import importlib.util
import sys

import numpy as np # to avoid reloading each time
import pandas as pd

N = 10

import hectorp
import hectorp.control
import hectorp.observations
import hectorp.simulatenoise

from array import array
from itertools import cycle,islice

with open('10000Random.dat','rb') as f:
    _10000_random = array('d')
    _10000_random.fromfile(f,10000)

def rng(size):
    return np.array([*islice(cycle(_10000_random),size)])


def call_stable_main(options,stdinput,stdinput_call,writer=None):

    stdinput_call(stdinput)

    # --- prepare mocking of Control object with options dictionary
    mockControlParamsGetitem = mock.Mock(side_effect = options.__getitem__)
    mockControlParams = mock.MagicMock(**{"params.__getitem__":mockControlParamsGetitem})
    mockControl = mock.Mock(return_value = mockControlParams)

    rngMock = mock.MagicMock(side_effect=rng)
    rngMock.standard_normal.side_effect=rng
    rngMock.randn.side_effect=rng
    rngMock.default_rng.side_effect=lambda *x: rngMock

    # --- Mock control ---
    with mock.patch('hectorp.control.Control',mockControl),\
         mock.patch('hectorp.observations.Control',mockControl),\
         mock.patch('hectorp.simulatenoise.Control',mockControl),\
         mock.patch('hectorp.simulatenoise.np.random',rngMock):

        # --- Mock file writing for recording output without
        # accessing the filesystem

        # If no writer provided create own.
        writer_own = False
        if writer is None:
            writer = io.StringIO()
            writer_own = True

        # calling the mock returns itself, entering context returns the
        # same mock, not a copy, writing methods calls the StringIO
        # writing methods. "with" clauses don't close the StringIO.
        mockFile = mock.MagicMock(__call__=lambda *x,**y: x[0],
                                  __enter__=lambda x:x,
                                  write=writer.write,
                                  writelines=writer.writelines)

        # We will mock sys.open to always return the writer mock.
        mockOpen = mock.Mock(return_value=mockFile)


        mockMdir = mock.Mock()

        # Mock Filesistem Calls on observations, force CLI arguments to be
        # blank, Mock os.makedirs to do nothing.
        with mock.patch('hectorp.observations.open',mockOpen), \
             mock.patch('sys.argv',['simulatenoise']), \
             mock.patch('os.makedirs',mockMdir):

            hectorp.simulatenoise.main()

        if writer_own: writer.seek(0) # if writer is a StringIO owned
                                      # by the function, rewind.

        return writer, { 'Control.params.__getitem__':mockControlParamsGetitem,
                         'Control.params':mockControlParams,
                         'Control':mockControl,
                         'output_file_write':mockFile,
                         'observations.open':mockOpen,
                         'os.makedirs':mockMdir,
                         'np.random':rngMock}

import eletor.simulatenoise as hps

from itertools import repeat, chain

NROUND = 5

data = []

model = "Powerlaw"
sigma = np.random.uniform(0,10,N)
kappa = np.random.uniform(-2,2,N)
m = np.random.randint(10,1000,N)
dt = np.random.uniform(0.5,20,N)

sigma, kappa, dt = map(lambda x: np.round(x,NROUND),(sigma,kappa,dt))

cli_params = zip(kappa,sigma)
model_params = zip(repeat(model),m,sigma,kappa,repeat('mom'),dt)

data.append( zip(repeat(model),m,dt,cli_params,model_params) )

model = "Flicker"

cli_params = zip(sigma)
model_params = zip(repeat(model),m,sigma,repeat('mom'),dt)

data.append( zip(repeat(model),m,dt,cli_params,model_params) )

model = "RandomWalk" #Shares parameters with Flicker

cli_params = zip(sigma)
model_params = zip(repeat(model),m,sigma,repeat('mom'),dt)

data.append( zip(repeat(model),m,dt,cli_params,model_params) )

model = "GGM"

one_minus_phi = np.random.uniform(1e-5,1,N)
one_minus_phi = np.round(one_minus_phi,NROUND)

cli_params = zip(one_minus_phi,kappa,sigma)
model_params = zip(repeat(model),m,sigma,kappa,one_minus_phi,repeat('mom'),dt)

data.append( zip(repeat(model),m,dt,cli_params,model_params) )

model = "VaryingAnnual"
phi = np.random.uniform(1e-5,1,N)
phi = np.round(phi,NROUND)

cli_params = zip(phi,sigma)
model_params = zip(repeat(model),m,sigma,phi,repeat('mom'),dt)

data.append( zip(repeat(model),m,dt,cli_params,model_params) )

model = "Matern"

lamba = np.random.uniform(1,10,N) * 10 ** np.random.uniform(-3,-1,N)
kappa = np.random.uniform(-2,-0.5,N) # Kappa for Matern differs from other models

lamba, kappa = map(lambda x: np.round(x,NROUND),(lamba,kappa))

cli_params = zip(lamba,kappa,sigma)
model_params = zip(repeat(model),m,sigma,lamba,kappa)

data.append( zip(repeat(model),m,dt,cli_params,model_params) )

model = "AR1"

cli_params = zip(phi,sigma)
model_params = zip(repeat(model),m,sigma,phi)

data.append( zip(repeat(model),m,dt,cli_params,model_params) )

testdata=chain(*data)

import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

n_test = 1

@pytest.mark.parametrize("model,m,dt,cli_params,model_params",testdata)
def test_noise_compare_generic(monkeypatch,model,m,dt,cli_params,model_params):

    global n_test

    paramnames = hps._paramlist[model]
    assert len(paramnames) == len(model_params[1:])


    options = {
               'SimulationDir':'None',
               "SimulationLabel":'None',
               "NumberOfSimulations":1,
               "NumberOfPoints":m,
               "SamplingPeriod":dt,
               "TimeNoiseStart":0,
               'NoiseModels': [model],
               'RepeatableNoise': True
                }

    # --- manual imput parameters ---
    inputs = '\n'.join(map(('{:.'+f'{NROUND}'+'f}').format,cli_params)) + '\n'

    stdin_call = lambda x: monkeypatch.setattr('sys.stdin',
                                               io.StringIO(x))

    myio, mocks = call_stable_main(options,inputs,stdin_call)

    noise = pd.read_fwf(myio,comment='#',names=['noise'],index_col=0)

    myio.close()

    #LOGGER.info(noise)

    rngMock = mock.MagicMock(side_effect=rng)
    rngMock.standard_normal.side_effect=rng
    rngMock.randn.side_effect=rng
    rngMock.default_rng.side_effect=lambda *x: rngMock

    y = hps.create_noise(m,dt,0,[model_params],rng=rngMock)

    # LOGGER.info(f"nuevo: {len(y)}, viejo: {len(noise)}")
    # LOGGER.info(cm.mock_calls)
    # LOGGER.info(mparams.mock_calls)

    differ = (y - noise.noise.values)
    assert all(differ < 1e-5)

    filename = "./bmks/nada.nada"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(f'bmks/Input{n_test:04d}.json','w') as f:
        json.dump({model:{i:j for i,j in zip(paramnames,model_params[1:])}},f,cls=NpEncoder)

    ar = array('d',y)

    with open(f'bmks/Output{n_test:04d}.dat','bw') as f:
        ar.tofile(f)

    n_test += 1

    return None
