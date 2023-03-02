#!/usr/bin/env python3

"""
Test that create_noise function works well.
"""

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

import eletor.simulatenoise as hps
import toml

@pytest.mark.parametrize("m",[2,13,42,141])
def test_noise_length(m):
    """
    Test that the length of the generated noise is the expected.
    """

    # Generate config string. We test with only GGM model, the main difference
    # Between models is on the create_h function,
    # Returned lengths for those are tested separated.

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
sigma=0.2
kappa=0.1
one_minus_phi=0.01 # GGN_1mphi
dt=1 # Sampling period
units="mom" # TS-format
"""

    control_data = toml.loads(toml_string)
    #y = hps.create_noise_(control_data)
    y = hps.create_noise(
            control_data['general']['NumberOfPoints'],
            control_data['general']['SamplingPeriod'],
            control_data['general']['TimeNoiseStart'],
            control_data['NoiseModels'],
            None)

    assert len(y) == m

#          m  , TimeNoiseStart
params = [[2  ,1  ],
          [13 ,5  ],
          [42 ,15 ],
          [141,121],]
@pytest.mark.parametrize("m,start",params)
def test_noise_length_nonzero_start(m,start):
    """
    Test that the length of the generated noise is the expected.
    """

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
TimeNoiseStart={start}
m={m-start} # Number of points
sigma=0.2
kappa=0.1
one_minus_phi=0.01 # GGN_1mphi
dt=1 # Sampling period
units="mom" # TS-format
"""

    control_data = toml.loads(toml_string)
    #y = hps.create_noise_(control_data)
    y = hps.create_noise(
            control_data['general']['NumberOfPoints'],
            control_data['general']['SamplingPeriod'],
            control_data['general']['TimeNoiseStart'],
            control_data['NoiseModels'],
            None)

    assert len(y) == m

#          m  , TimeNoiseStart
params = [[-3 ,0 ],
          [141,200],]
@pytest.mark.parametrize("m,start",params)
def test_noise_negative_length(m,start):
    """
    Cases that should raise ValueError for asking negative dimension of noise
    """

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
TimeNoiseStart={start}
m={m-start} # Number of points
sigma=0.2
kappa=0.1
one_minus_phi=0.01 # GGN_1mphi
dt=1 # Sampling period
units="mom" # TS-format
"""

    control_data = toml.loads(toml_string)

    with pytest.raises(ValueError) as exc_info:

        y = hps.create_noise(
                control_data['general']['NumberOfPoints'],
                control_data['general']['SamplingPeriod'],
                control_data['general']['TimeNoiseStart'],
                control_data['NoiseModels'],
                None)

    exception = exc_info.value

    assert 'negative' in exception.args[0]


models = ['White','Flicker','GGM','Powerlaw','Matern','AR1','VaryingAnnual','RandomWalk']
@pytest.mark.parametrize("model",models)
def test_calls_h(model):
    """
    Test that create_noise calls the corresponding hs functions
    """
    toml_string = f"""
[file_config]
SimulationDir=""
SimulationLabel=""

[general]
NumberOfSimulations=1
NumberOfPoints=100
TimeNoiseStart=0
SamplingPeriod=1

[NoiseModels.{model}.params]
NumberOfSimulations=1
TimeNoiseStart=0
m=100 # Number of points
sigma=0.2
dt=1
units="mom"
"""
    control_data = toml.loads(toml_string)

    h_mock = mock.Mock(return_value=(1,np.ones(100)))
    with mock.patch(f'eletor.simulatenoise.create_hs.create_h_{model}',h_mock):
        y = hps.create_noise(
                control_data['general']['NumberOfPoints'],
                control_data['general']['SamplingPeriod'],
                control_data['general']['TimeNoiseStart'],
                control_data['NoiseModels'],
                None)

    h_mock.assert_called()

models = ['White','Flicker','GGM','Powerlaw','Matern','AR1','VaryingAnnual','RandomWalk']

import eletor.create_hs
import eletor.parameter_generation

## Tengo que revisar este test, en teoría si el ruido es una delta, al hacer
## la convolución resulta la identidad, pero no se como se define la delta
## para la fft discreta.
#@pytest.mark.parametrize("model",models)
#def test_convolution_identity(model):
#    """
#    Test that create_noise calls the corresponding hs functions
#    """
#    rng = lambda x: np.array([1,*np.zeros(x-1)])
#    rng.standard_normal = rng
#
#    parameter_fun = getattr(eletor.parameter_generation,
#                    f"random_{model.lower()}_parameters")
#    parameters = parameter_fun('1')
#
#    y = hps.create_noise(
#            100,
#            1,
#            0,
#            parameters['NoiseModels'],
#            rng=rng)
#
#    window_fun = getattr(eletor.create_hs,f'create_h_{model}')
#
#    parameters['NoiseModels'][model]['1']['m'] = 100
#    parameters['NoiseModels'][model]['1']['dt'] = 1
#
#    _,window = window_fun(**parameters['NoiseModels'][model]['1'])
#
#    relation = y / window
#    print(relation)
#    assert all(relation == relation[0])

