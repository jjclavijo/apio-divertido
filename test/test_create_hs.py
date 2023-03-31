import pytest

import numpy as np

from eletor import create_hs
import eletor.parameter_generation


MODELS = ['White','Flicker','GGM','Powerlaw','Matern','AR1','VaryingAnnual','RandomWalk']

@pytest.mark.parametrize("model",MODELS)
def test_h_length(model):
    """
    Controlar que las ventanas tienen el largo de la señal deseada.
    """

    parameter_fun = getattr(eletor.parameter_generation, model.lower())
    parameters = parameter_fun('1',dt=1)['NoiseModels'][model]['1']

    m = parameters['m']

    h_fun = getattr(create_hs, model)

    _,y = h_fun(**parameters)

    assert m == len(y)

GMMODELS = ['Flicker','GGM','Powerlaw','RandomWalk']

@pytest.mark.parametrize("model",GMMODELS)
def test_normalizes_variance(model):
    """
    Los modelos basados en Gauss Markov Ponen una varianza en el primer
    casillero de la función de impulso, por eso después tienen que escalar la
    varianza.
    """
    parameter_fun = getattr(eletor.parameter_generation, model.lower())
    parameters = parameter_fun('1',dt=1)['NoiseModels'][model]['1']

    m = parameters['m']

    h_fun = getattr(create_hs, model)

    s,_ = h_fun(**parameters)

    assert s != parameters['sigma']

NOGMMODELS = list(set(MODELS) - set(GMMODELS))
@pytest.mark.parametrize("model",NOGMMODELS)
def test_dont_normalizes_variance(model):
    """
    Los modelos que no son basados en Gauss Markov no necesitan escalar la
    varianza.
    """
    parameter_fun = getattr(eletor.parameter_generation, model.lower())
    parameters = parameter_fun('1',dt=1)['NoiseModels'][model]['1']

    m = parameters['m']

    h_fun = getattr(create_hs, model)

    s,_ = h_fun(**parameters)

    assert s == parameters['sigma']

DECREASINGMODELS = ['Flicker']
@pytest.mark.parametrize("model",DECREASINGMODELS)
def test_decreases(model):
    """
    El modelo Flicker tiene una función de impulso estrictamente decreciente.
    """
    parameter_fun = getattr(eletor.parameter_generation, model.lower())
    parameters = parameter_fun('1',dt=1)['NoiseModels'][model]['1']

    m = parameters['m']

    h_fun = getattr(create_hs, model)

    _,y = h_fun(**parameters)

    # Para m grandes puede haber ceros al final (porque float)
    if y[-1] == 0:
        y = y[y != 0]

    assert all(y[:-1] > y[1:])

ABSDECREASINGMODELS = ['GGM','Powerlaw']

@pytest.mark.parametrize("model",ABSDECREASINGMODELS)
def test_abs_decreases(model):
    """
    Modelos que tienen una función de impulso con valor absoluto decreciente.
    """

    parameter_fun = getattr(eletor.parameter_generation, model.lower())
    parameters = parameter_fun('1',dt=1)['NoiseModels'][model]['1']

    m = parameters['m']

    h_fun = getattr(create_hs, model)

    _,y = h_fun(**parameters)

    if y[-1] == 0:
        y = np.trim_zeros(y,trim='b')

    y = np.abs(y)

    assert all(y[:-1] > y[1:])

def test_AR_abs_decreases():
    """
    El modelo AR1 tiene una funcion decreciente en valor absoluto, pero un 0
    en el primer casillero.
    """

    model = 'AR1'
    parameter_fun = getattr(eletor.parameter_generation, model.lower())
    parameters = parameter_fun('1',dt=1)['NoiseModels'][model]['1']

    m = parameters['m']

    h_fun = getattr(create_hs, model)

    _,y = h_fun(**parameters)

    if y[-1] == 0:
        y = np.trim_zeros(y,trim='b')

    y = np.abs(y)

    assert all(y[1:-1] > y[2:])

def test_Matern_increases_decreases():
    """
    La Función de Matern tiene un pico
    """
    model = 'Matern'
    parameter_fun = getattr(eletor.parameter_generation, model.lower())
    parameters = parameter_fun('1',dt=1)['NoiseModels'][model]['1']

    m = parameters['m']

    h_fun = getattr(create_hs, model)

    _,y = h_fun(**parameters)

    if y[-1] == 0:
        y = np.trim_zeros(y,trim='b')

    split = np.argmax(y)


    inc = y[:split]
    dec = y[split:]

    assert all(inc[:-1] < inc[1:])

    # admitimos hasta 5 errores numéricos. debe haber otra forma mejor de
    # garantizar esto.
    assert sum((dec[:-1] - dec[1:]) > 1) < 5

def test_VA_non_decreasing():
    """
    Chequeamos que VaryingAnnual NO ES DECRECIENTE, ES PERIODICA.
    """
    model = "VaryingAnnual"
    parameter_fun = getattr(eletor.parameter_generation, model.lower())

    # m greater than half a year
    m = np.random.randint(200,1000)
    parameters = parameter_fun('1',dt=1,m=m)['NoiseModels'][model]['1']

    h_fun = getattr(create_hs, model)

    _,y = h_fun(**parameters)

    y = np.abs(y)

    assert not all(y[:-1] > y[1:])
