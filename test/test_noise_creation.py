#!/usr/bin/env python3

import io

from unittest import mock
import pytest

import logging

LOGGER = logging.getLogger(__name__)

import importlib.util
import sys

import numpy as np # to avoid reloading each time
import pandas as pd

N = 1

import hectorp
import hectorp.control
import hectorp.observations
import hectorp.simulatenoise
import toml

def call_stable_main(options,stdinput,stdinput_call,writer=None):

    stdinput_call(stdinput)

    # --- prepare mocking of Control object with options dictionary
    mockControlParamsGetitem = mock.Mock(side_effect = options.__getitem__)
    mockControlParams = mock.MagicMock(**{"params.__getitem__":mockControlParamsGetitem})
    mockControl = mock.Mock(return_value = mockControlParams)

    # --- Mock control ---
    with mock.patch('hectorp.control.Control',mockControl),\
         mock.patch('hectorp.observations.Control',mockControl),\
         mock.patch('hectorp.simulatenoise.Control',mockControl):

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
                         'os.makedirs':mockMdir}

import eletor.simulatenoise as hps

testdataGGM = [ (10,1,0.23,0.01,1), (20,2.3,0.51,0.01,1) ]
@pytest.mark.parametrize("m,sigma,kappa,one_minus_phi,dt",testdataGGM)
def test_h_noise_generation_GGM(monkeypatch,m,sigma,kappa,one_minus_phi,dt):

    LOGGER.info(monkeypatch)

    options = {
               'SimulationDir':'None',
               "SimulationLabel":'None',
               "NumberOfSimulations":1,
               "NumberOfPoints":m,
               "SamplingPeriod":dt,
               "TimeNoiseStart":0,
               "GGM_1mphi":one_minus_phi,
               'NoiseModels': ["GGM"]
                }

    # Kappa, Sigma
    monkeypatch.setattr('sys.stdin',
                        io.StringIO(f'{kappa:.3f}\n{sigma:.3f}\n'))

    mparams = mock.Mock(side_effect = options.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)
    mdir = mock.Mock()

    with mock.patch('hectorp.control.Control',cm),\
         mock.patch('hectorp.observations.Control',cm),\
         mock.patch('hectorp.simulatenoise.Control',cm):

        myio = io.StringIO()

        #openmock = mock.MagicMock(__call__=lambda x: x,__enter__=lambda x:mockFile)
        mockFile = mock.MagicMock(__call__=lambda *x,**y: x[0],__enter__=lambda x:x, write=myio.write, writelines=myio.writelines)
        mockOpen = mock.Mock(return_value=mockFile)

        # Mock Filesistem Calls and CLI input.
        with mock.patch('hectorp.observations.open',mockOpen), \
             mock.patch('sys.argv',['simulatenoise']), \
             mock.patch('os.makedirs',mdir):

            hectorp.simulatenoise.main()

    mockOpen.assert_called()

    #myio.seek(0)
    #noise = pd.read_fwf(myio,comment='#',names=['noise'],index_col=0)

    #LOGGER.info(noise)

    return None


def test_new_create_noise():
    toml_string = f"""
[file_config]
SimulationDir=""
SimulationLabel=""

[general]
NumberOfSimulations=1
NumberOfPoints=100
TimeNoiseStart=0
SamplingPeriod=1

[NoiseModels.GGM.params]
NumberOfSimulations=1
TimeNoiseStart=0
m=100 # Number of points
sigma=0.2
kappa=0.1
one_minus_phi=0.01 # GGN_1mphi
dt=1 # Sampling period
spectral_density=0.01
units="mom" # TS-format
"""

    control_data = toml.loads(toml_string)
    y = hps.create_noise_(control_data)

    #hps.Control.clear_all()

    assert len(y) == 100

m = np.random.randint(10,100,N)
sigma = np.random.uniform(0,10,N)
kappa = np.random.uniform(-2,2,N)
one_minus_phi = np.random.uniform(1e-6,1,N)
dt = np.random.uniform(0.5,20,N)
testdataGGM = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,kappa,one_minus_phi,dt)))
#testdataGGM = [ (20,1,0.23,0.01,1), (30,2.3,0.51,0.01,1) ]
@pytest.mark.parametrize("m,sigma,kappa,one_minus_phi,dt",testdataGGM)
def test_noise_compare_GGM(monkeypatch,m,sigma,kappa,one_minus_phi,dt):

    # Create Reference noise
    options = {
               'SimulationDir':'None',
               "SimulationLabel":'None',
               "NumberOfSimulations":1,
               "NumberOfPoints":m,
               "SamplingPeriod":dt,
               "TimeNoiseStart":0,
               "GGM_1mphi":one_minus_phi,
               'NoiseModels': ["GGM"],
               'RepeatableNoise': True
                }

    inputs = [kappa,sigma]
    inputs = '\n'.join(map('{:.3f}'.format,inputs)) + '\n'
    # Kappa, Sigma

    stdin_call = lambda x: monkeypatch.setattr('sys.stdin',
                                               io.StringIO(x))

    myio, mocks = call_stable_main(options,inputs,stdin_call)

    noise = pd.read_fwf(myio,comment='#',names=['noise'],index_col=0)
    myio.close()

    # --- Real test starts here ---
    toml_string = f"""
[file_config]
SimulationDir=""
SimulationLabel=""

[general]
NumberOfSimulations=1
NumberOfPoints={m}
TimeNoiseStart=0
SamplingPeriod=1

[NoiseModels.GGM.params]
NumberOfSimulations=1
TimeNoiseStart=0
m={m} # Number of points
sigma={sigma}
kappa={kappa}
one_minus_phi={one_minus_phi} # GGN_1mphi
dt={dt} # Sampling period
spectral_density=0.01
units="mom" # TS-format
"""

    control_data = toml.loads(toml_string)
    print("+++++++++++++++++++++++")
    print("New control structure: ")
    print(control_data)

    y = hps.create_noise_(control_data,rng=np.random.default_rng(0))
    assert 1==1

    ## LOGGER.info(f"nuevo: {len(y)}, viejo: {len(noise)}")
    ## LOGGER.info(cm.mock_calls)
    ## LOGGER.info(mparams.mock_calls)
    assert len(y) == m
    assert len(noise.noise.values) == m
    differ = (y - noise.noise.values)
    print(noise.noise.values[0], y[0])
    assert all(differ < 1e-5)

    #return None

testdataGGM = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,kappa,one_minus_phi,dt)))
#testdataGGM = [ (20,1,0.23,0.01,1), (30,2.3,0.51,0.01,1) ]
@pytest.mark.parametrize("m,sigma,kappa,one_minus_phi,dt",testdataGGM)
def test_noise_compare_GGM2(monkeypatch,m,sigma,kappa,one_minus_phi,dt):

    toml_string = f"""
[file_config]
SimulationDir=""
SimulationLabel=""

[general]
NumberOfSimulations=1
NumberOfPoints={m}
TimeNoiseStart=0
SamplingPeriod=1

[NoiseModels.GGM.params]
NumberOfSimulations=1
TimeNoiseStart=0
m={m} # Number of points
sigma={sigma}
kappa={kappa}
one_minus_phi={one_minus_phi} # GGN_1mphi
dt={dt} # Sampling period
spectral_density=0.01
units="mom" # TS-format
"""
    options = {
               'SimulationDir':'None',
               "SimulationLabel":'None',
               "NumberOfSimulations":1,
               "NumberOfPoints":m,
               "SamplingPeriod":dt,
               "TimeNoiseStart":0,
               "GGM_1mphi":one_minus_phi,
               'NoiseModels': ["GGM"],
               'RepeatableNoise': True
                }

    inputs = [kappa,sigma]
    inputs = '\n'.join(map('{:.3f}'.format,inputs)) + '\n'
    # Kappa, Sigma
    stdin_call = lambda x: monkeypatch.setattr('sys.stdin',
                                               io.StringIO(x))
    myio, mocks = call_stable_main(options,inputs,stdin_call)
    noise = pd.read_fwf(myio,comment='#',names=['noise'],index_col=0)
    myio.close()

    ##LOGGER.info(noise)

    control_data = toml.loads(toml_string)
    print("+++++++++++++++++++++++")
    print("New control structure: ")
    print(control_data)

    #y = hps.create_noise(m,dt,0,[('GGM',m,sigma,kappa,one_minus_phi,'mom',dt)],rng=np.random.default_rng(0))
    y = hps.create_noise_(control_data,rng=np.random.default_rng(0))
    # LOGGER.info(f"nuevo: {len(y)}, viejo: {len(noise)}")
    # LOGGER.info(cm.mock_calls)
    # LOGGER.info(mparams.mock_calls)

    differ = (y - noise.noise.values)
    assert all(differ < 1e-5)

    return None



from itertools import repeat, chain

NROUND = 5

data = []
model = "Powerlaw"
sigma = np.random.uniform(0,10)
kappa = np.random.uniform(-2,2)
m = np.random.randint(10,1000)
dt = np.random.uniform(0.5,20)

model_params = [(m,sigma,kappa,'mom',dt,(kappa, sigma)),]
@pytest.mark.parametrize("m,sigma,kappa,units,dt,cli_params",model_params)
def test_noise_Powerlaw(monkeypatch,m,sigma,kappa,units,dt,cli_params):

    options = {
               'SimulationDir':'None',
               "SimulationLabel":'None',
               "NumberOfSimulations":1,
               "NumberOfPoints":m,
               "SamplingPeriod":dt,
               "TimeNoiseStart":0,
               'NoiseModels': ['Powerlaw'],
               'RepeatableNoise': True
                }

    # --- manual imput parameters ---
    inputs = '\n'.join(map(('{:.'+f'{NROUND}'+'f}').format,cli_params)) + '\n'
    stdin_call = lambda x: monkeypatch.setattr('sys.stdin', io.StringIO(x))
    myio, mocks = call_stable_main(options,inputs,stdin_call)
    noise = pd.read_fwf(myio,comment='#',names=['noise'],index_col=0)
    myio.close()

    toml_string = f"""
[file_config]
SimulationDir=""
SimulationLabel=""

[general]
NumberOfSimulations=1
NumberOfPoints={m}
TimeNoiseStart=0
SamplingPeriod=1

[NoiseModels.Powerlaw.params]
NumberOfSimulations=1
TimeNoiseStart=0
m={m} # Number of points
sigma={sigma}
kappa={kappa}
dt={dt} # Sampling period
units="{units}" # TS-format
spectral_density=0.01
"""
    control_data = toml.loads(toml_string)
    y = hps.create_noise_(control_data,rng=np.random.default_rng(0))
    differ = (y - noise.noise.values)
    assert len(y) == len(noise.noise.values)
    assert all(differ < 1e-4)

model = "Flicker"
model_params = [(m,sigma,'mom',dt,(sigma,)),]
@pytest.mark.parametrize("m,sigma,units,dt,cli_params",model_params)
def test_noise_Flicker(monkeypatch,m,sigma,units,dt,cli_params):

    options = {
               'SimulationDir':'None',
               "SimulationLabel":'None',
               "NumberOfSimulations":1,
               "NumberOfPoints":m,
               "SamplingPeriod":dt,
               "TimeNoiseStart":0,
               'NoiseModels': ['Flicker'],
               'RepeatableNoise': True
                }

    # --- manual imput parameters ---
    inputs = '\n'.join(map(('{:.'+f'{NROUND}'+'f}').format,cli_params)) + '\n'
    stdin_call = lambda x: monkeypatch.setattr('sys.stdin', io.StringIO(x))
    myio, mocks = call_stable_main(options,inputs,stdin_call)
    noise = pd.read_fwf(myio,comment='#',names=['noise'],index_col=0)
    myio.close()

    toml_string = f"""
[file_config]
SimulationDir=""
SimulationLabel=""

[general]
NumberOfSimulations=1
NumberOfPoints={m}
TimeNoiseStart=0
SamplingPeriod=1

[NoiseModels.Flicker.params]
NumberOfSimulations=1
TimeNoiseStart=0
m={m} # Number of points
sigma={sigma}
dt={dt} # Sampling period
units="{units}" # TS-format
spectral_density=0.01
"""
    control_data = toml.loads(toml_string)
    y = hps.create_noise_(control_data,rng=np.random.default_rng(0))

    differ = (y - noise.noise.values)
    assert all(differ < 1e-5)
#
#model = "RandomWalk" #Shares parameters with Flicker
#
#cli_params = zip(sigma)
#model_params = zip(repeat(model),m,sigma,repeat('mom'),dt)
#
#data.append( zip(repeat(model),m,dt,cli_params,model_params) )
#
#model = "GGM"
#
#one_minus_phi = np.random.uniform(1e-5,1,N)
#one_minus_phi = np.round(one_minus_phi,NROUND)
#
#cli_params = zip(one_minus_phi,kappa,sigma)
#model_params = zip(repeat(model),m,sigma,kappa,one_minus_phi,repeat('mom'),dt)
#
#data.append( zip(repeat(model),m,dt,cli_params,model_params) )
#
#model = "VaryingAnnual"
#phi = np.random.uniform(1e-5,1,N)
#phi = np.round(phi,NROUND)
#
#cli_params = zip(phi,sigma)
#model_params = zip(repeat(model),m,sigma,phi,repeat('mom'),dt)
#
#data.append( zip(repeat(model),m,dt,cli_params,model_params) )
#
#model = "Matern"
#
#lamba = np.random.uniform(1,10,N) * 10 ** np.random.uniform(-3,-1,N)
#kappa = np.random.uniform(-2,-0.5,N) # Kappa for Matern differs from other models
#
#lamba, kappa = map(lambda x: np.round(x,NROUND),(lamba,kappa))
#
#cli_params = zip(lamba,kappa,sigma)
#model_params = zip(repeat(model),m,sigma,lamba,kappa)
#
#data.append( zip(repeat(model),m,dt,cli_params,model_params) )
#
#model = "AR1"
#
#cli_params = zip(phi,sigma)
#model_params = zip(repeat(model),m,sigma,phi)
#
#data.append( zip(repeat(model),m,dt,cli_params,model_params) )
#
#
#testdata=chain(*data)

#@pytest.mark.parametrize("model,m,dt,cli_params,model_params",testdata)
#def test_noise_compare_generic(monkeypatch,model,m,dt,cli_params,model_params):
#    print()
#    print("++++++++++++++++++++++++++++++")
#    print(f"model: {model}, m: {m}, dt: {dt}, cli_params: {cli_params}, model_params: {model_params}")
#
#    toml_string = f"""
#[file_config]
#SimulationDir=""
#SimulationLabel=""
#
#[general]
#NumberOfSimulations=1
#NumberOfPoints={m}
#TimeNoiseStart=0
#SamplingPeriod=1
#
#[NoiseModels.{model}.params]
#NumberOfSimulations=1
#TimeNoiseStart=0
#m={m} # Number of points
#sigma={sigma}
#phi=0.1
#kappa=0.1
#one_minus_phi={one_minus_phi} # GGN_1mphi
#dt={dt} # Sampling period
#spectral_density=0.01
#units="mom" # TS-format
#"""
#
#    options = {
#               'SimulationDir':'None',
#               "SimulationLabel":'None',
#               "NumberOfSimulations":1,
#               "NumberOfPoints":m,
#               "SamplingPeriod":dt,
#               "TimeNoiseStart":0,
#               'NoiseModels': [model],
#               'RepeatableNoise': True
#                }
#
#    # --- manual imput parameters ---
#    inputs = '\n'.join(map(('{:.'+f'{NROUND}'+'f}').format,cli_params)) + '\n'
#    stdin_call = lambda x: monkeypatch.setattr('sys.stdin',
#                                               io.StringIO(x))
#    myio, mocks = call_stable_main(options,inputs,stdin_call)
#    noise = pd.read_fwf(myio,comment='#',names=['noise'],index_col=0)
#    myio.close()
#
#    control_data = toml.loads(toml_string)
#    print("+++++++++++++++++++++++")
#    print("New control structure: ")
#    print(control_data)
#
#    #y = hps.create_noise(m,dt,0,[('GGM',m,sigma,kappa,one_minus_phi,'mom',dt)],rng=np.random.default_rng(0))
#    y = hps.create_noise_(control_data,rng=np.random.default_rng(0))
#    # LOGGER.info(f"nuevo: {len(y)}, viejo: {len(noise)}")
#
#    # LOGGER.info(f"nuevo: {len(y)}, viejo: {len(noise)}")
#    # LOGGER.info(cm.mock_calls)
#    # LOGGER.info(mparams.mock_calls)
#
#    differ = (y - noise.noise.values)
#    assert all(differ < 1e-5)
#
#    return None
