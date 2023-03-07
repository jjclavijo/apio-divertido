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
[general]
NumberOfPoints={m}
SamplingPeriod=1

[NoiseModels.GGM.params]
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
            42, # Este parametro no afecta
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
[general]
NumberOfPoints={m}
SamplingPeriod=1

[NoiseModels.GGM.params]
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
            10923, # TimenoiseStart
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
[general]
NumberOfPoints={m}
SamplingPeriod=1

[NoiseModels.GGM.params]
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
                10213,
                control_data['NoiseModels'],
                None)

    exception = exc_info.value

    assert 'negative' in exception.args[0]


MODELS = ['White','Flicker','GGM','Powerlaw','Matern','AR1','VaryingAnnual','RandomWalk']
@pytest.mark.parametrize("model",MODELS)
def test_calls_proper_h(model):
    """
    Test that create_noise calls the corresponding hs functions
    """
    toml_string = f"""
[NoiseModels.{model}.params]
"""
    control_data = toml.loads(toml_string)

    h_mock = mock.Mock(return_value=(1,np.ones(100)))
    with mock.patch(f'eletor.simulatenoise.create_hs.{model}',h_mock):
        y = hps.create_noise(
                100, # number of points
                1, # sampling period
                0, # TimeNoiseStart
                control_data['NoiseModels'],
                None)

    h_mock.assert_called()

MODELS = ['White','Flicker','GGM','Powerlaw','Matern','AR1','VaryingAnnual','RandomWalk']

import eletor.create_hs
import eletor.parameter_generation

@pytest.mark.parametrize("model",MODELS)
def test_identity1_convolution(model):
    """
    Test that, if we mock scipy.signal.fftconvolve to return just the first
    argument create_noise function just returns the window that create_h_...
    would have created.
    """
    parameter_fun = getattr(eletor.parameter_generation,
                    model.lower())
    parameters = parameter_fun('1',m=100,dt=1)

    mock_fftconvolve = mock.Mock(side_effect=lambda x,y:x)

    with mock.patch('eletor.simulatenoise.signal.fftconvolve',mock_fftconvolve):
        y = hps.create_noise(
                100,
                1,
                0,
                parameters['NoiseModels'],
                rng=None)

    window_fun = getattr(eletor.create_hs,model)

    _,window = window_fun(**parameters['NoiseModels'][model]['1'])

    assert all(y == window)

from scipy.stats import normaltest

@pytest.mark.parametrize("model",MODELS)
def test_identity2_convolution_is_the_noise(model):
    """
    Test that, if we mock scipy.signal.fftconvolve with the identity
    create_noise function just returns just the white noise or a scaled
    version.
    """
    parameter_fun = getattr(eletor.parameter_generation,
                    model.lower())
    parameters = parameter_fun('1',m=100,dt=1)

    mock_fftconvolve = mock.Mock(side_effect=lambda x,y:y)

    ruido = np.random.randn(100)

    rng = lambda x: ruido
    rng.standard_normal = rng

    with mock.patch('eletor.simulatenoise.signal.fftconvolve',mock_fftconvolve):
        y = hps.create_noise(
                100,
                1,
                0,
                parameters['NoiseModels'],
                rng=rng)

    proportion = y / ruido
    scale = proportion[0]

    EPSILON = 1e-10

    assert all((proportion - scale) < 1e-10)

## Este test es EXPERIMENTAL, se basa en las propiedades estadísticas de la
## Salida, no en el código en sí.
## La idea es que los ruidos que generamos no deberían aparecer como que vienen
## de una normal salvo en situaciones excepcionales.
##
## Hay algunos parámetros de las distribuciones de ruido que
## producen cosas casi normales.
## Probablemente el test posta sería aplicar el test de normalidad
## Sobre ventanas de la salida o algo así, pero se empieza por algo.
## A veces falla. Por ahora, o lo desactivas o lo corres varias veces y probás
## que no falle siempre.
## Cuando tenga cabeza probablemente lo correcto es fijar algunos parámetros,
## sabiendo de antemano si, al fijar esos parámetros el test de normalidad
## debería saltar o no.

@pytest.mark.stat_properties
@pytest.mark.parametrize("model",MODELS)
def test_samples_not_normal(model):
    """
    Test that, if we mock scipy.signal.fftconvolve with the identity
    create_noise function just returns whit noise
    """

    pvalues = []

    # Deterministico, pero partimos de un ruido
    # para los test no sea demasiado poco NORMAL.
    pvalue0 = 0
    while pvalue0 < 0.90:
        ruido = np.random.randn(100)
        _, pvalue0 = normaltest(ruido)

    rng = lambda x: ruido
    rng.standard_normal = rng

    # Se hacen dos corridas porque para algunos parámetros el
    # proceso casi no se altera,
    # Podríamos evitando tuneando o fijando los parámetros.
    # Puede que eso esté out of the scope para esta etapa.
    pvalue = pvalue0
    for i in range(2):
        parameter_fun = getattr(eletor.parameter_generation,
                        model.lower())
        parameters = parameter_fun('1',sigma=1,m=100,dt=5)

        y = hps.create_noise(
                100,
                10,
                0,
                parameters['NoiseModels'],
                rng=rng)

        _, pvalue_ = normaltest(y-ruido)
        if pvalue_ < pvalue:
            pvalue = pvalue_

    if model == 'White':
        assert True # We already tested this case
    else:
        # Aplicar un modelo de ruido no blanco tiene que bajar sensiblemente el
        # p_valor del test de normalidad.
        assert ( pvalue0 - pvalue ) > 0.1

