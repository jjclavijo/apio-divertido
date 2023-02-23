#!/usr/bin/env python3

import io

from unittest import mock
import pytest

import logging

LOGGER = logging.getLogger(__name__)

import eletor.simulatenoise
import eletor.control
import eletor.create_hs

import importlib.util
import sys

import hectorp.control
import hectorp.simulatenoise

# spec = importlib.util.spec_from_file_location("hectorp.simulatenoise",
#                                               "./hectorp/src/hectorp/simulatenoise.py")
# sn_stable = importlib.util.module_from_spec(spec)
# sys.modules["hectorp.simulatenoise"] = sn_stable
# spec.loader.exec_module(sn_stable)

N = 10

def test_h_white(monkeypatch):

    monkeypatch.setattr('sys.stdin', io.StringIO('1.0\n'))

    mparams = mock.Mock(side_effect = {}.__getitem__)
    m = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = m)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(10,"White",1,"mom",control)


    snew,hnew = eletor.create_hs.create_h("White",10,1)

    assert all(hnew == h)

    assert s == 1

    return None

import numpy as np

sigma = np.random.uniform(0,10,N)
kappa = np.random.uniform(-2,2,N)
m = np.random.randint(10,1000,N)
dt = np.random.uniform(0.5,20,N)
#testdataPowerlaw = [ (1,0.5,10,1), (3.2,0.6,20,1) ]
testdataPowerlaw = zip(*map(lambda x: np.round(x,3),(sigma,kappa,m,dt)))

@pytest.mark.parametrize("sigma,kappa,m,dt",testdataPowerlaw)
def test_h_Powerlaw(monkeypatch,sigma,kappa,m,dt):

    # Kappa, Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{kappa:.3f}\n{sigma:.3f}\n'))

    mparams = mock.Mock(side_effect = {}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"Powerlaw",dt,"mom",control)


    snew,hnew = eletor.create_hs.create_h("Powerlaw",m,sigma,kappa,"mom",dt)

    assert all(hnew == h)
    assert s == snew

    return None


import numpy as np
sigma = np.random.uniform(0,10,N)
m = np.random.randint(10,1000,N)
dt = np.random.uniform(0.5,20,N)
#testdataFlicker = [ (1,10,1), (3.2,20,1) ]
testdataFlicker = zip(*map(lambda x: np.round(x,3),(sigma,m,dt)))

@pytest.mark.parametrize("sigma,m,dt",testdataFlicker)
def test_h_Flicker(monkeypatch,sigma,m,dt):

    # Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{sigma:.3f}\n'))

    mparams = mock.Mock(side_effect = {}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"Flicker",dt,"mom",control)


    snew,hnew = eletor.create_hs.create_h("Flicker",m,sigma,"mom",dt)

    assert all(hnew == h)
    assert s == snew

    return None

import numpy as np
sigma = np.random.uniform(0,10,N)
m = np.random.randint(10,1000,N)
dt = np.random.uniform(0.5,20,N)
#testdataRandomWalk = [ (1,10,1), (3.2,20,1) ]
testdataRandomWalk = zip(*map(lambda x: np.round(x,3),(sigma,m,dt)))
@pytest.mark.parametrize("sigma,m,dt",testdataRandomWalk)
def test_h_RandomWalk(monkeypatch,sigma,m,dt):
    # Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{sigma:.3f}\n'))

    mparams = mock.Mock(side_effect = {}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"RandomWalk",dt,"mom",control)


    snew,hnew = eletor.create_hs.create_h("RandomWalk",m,sigma,"mom",dt)

    assert all(hnew == h)
    assert s == snew

    return None

import numpy as np
m = np.random.randint(10,1000,N)
sigma = np.random.uniform(0,10,N)
kappa = np.random.uniform(-2,2,N)
one_minus_phi = np.random.uniform(1e-6,1,N)
dt = np.random.uniform(0.5,20,N)
#testdataGGM = [ (10,1,0.23,0.01,1), (20,2.3,0.51,0.01,1) ]
testdataGGM = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,kappa,one_minus_phi,dt)))

@pytest.mark.parametrize("m,sigma,kappa,one_minus_phi,dt",testdataGGM)
def test_h_GGM_1(monkeypatch,m,sigma,kappa,one_minus_phi,dt):

    # Kappa, Sigma
    monkeypatch.setattr('sys.stdin',
                        io.StringIO(f'{one_minus_phi:f}\n{kappa:.3f}\n{sigma:.3f}\n'))

    mparams = mock.Mock(side_effect = {}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"GGM",dt,"mom",control)


    snew,hnew = eletor.create_hs.create_h("GGM",m,sigma,kappa,one_minus_phi,"mom",dt)
    assert all(hnew == h)
    assert s == snew

    return None

testdataGGM = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,kappa,one_minus_phi,dt)))

@pytest.mark.parametrize("m,sigma,kappa,one_minus_phi,dt",testdataGGM)
def test_h_GGM_2(monkeypatch,m,sigma,kappa,one_minus_phi,dt):

    # Kappa, Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{kappa:.3f}\n{sigma:.3f}\n'))

    mparams = mock.Mock(side_effect = { "GGM_1mphi":one_minus_phi }.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"GGM",dt,"mom",control)


    snew,hnew = eletor.create_hs.create_h("GGM",m,sigma,kappa,one_minus_phi,"mom",dt)

    assert all(hnew == h)
    assert s == snew

    return None

m = np.random.randint(10,1000,N)
sigma = np.random.uniform(0,10,N)
phi = np.random.uniform(1e-6,1,N)
dt = np.random.uniform(0.5,20,N)
#testdataVaryingAnnual = [ (10,1,0.5,1) ]
testdataVaryingAnnual = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,phi,dt)))

@pytest.mark.parametrize("m,sigma,phi,dt",testdataVaryingAnnual)
def test_h_VaryingAnual(monkeypatch,m,sigma,phi,dt):

    # Phi, Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{phi:.3f}\n{sigma:.3f}\n'))

    mparams = mock.Mock(side_effect = {}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"VaryingAnnual",dt,"mom",control)

    snew,hnew = eletor.create_hs.create_h("VaryingAnnual",m,sigma,phi,"mom",dt)

    differ = np.abs(hnew - h)
    assert all(differ < 1e-13) #slight differences for cosine and sine
    assert s == snew

    assert True

    return None

m = np.random.randint(10,1000,N)
sigma = np.random.uniform(0,10,N)
phi = np.random.uniform(1e-6,1,N)
dt = np.random.uniform(0.5,20,N)
#testdataVaryingAnnual = [ (10,1,0.5,1) ]
testdataVaryingAnnual = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,phi,dt)))

@pytest.mark.parametrize("m,sigma,phi,dt",testdataVaryingAnnual)
def test_h_VaryingAnual_2(monkeypatch,m,sigma,phi,dt):

    # Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{sigma:.3f}\n'))

    mparams = mock.Mock(side_effect = {"phi_fixed":phi}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"VaryingAnnual",dt,"mom",control)

    snew,hnew = eletor.create_hs.create_h("VaryingAnnual",m,sigma,phi,"mom",dt)

    differ = np.abs(hnew - h)
    assert all(differ < 1e-13)
    assert s == snew

    mparams.assert_called_with('phi_fixed')

    return None

m = np.random.randint(10,1000,N)
sigma = np.random.uniform(0,10,N)
lamba = np.random.uniform(1,10,N) * 10 ** np.random.uniform(-3,-1,N)
kappa = np.random.uniform(-2,-0.5,N)
dt = np.random.uniform(0.5,20,N)
#testdataMatern = [ (10,1,0.5,1) ]
testdataMatern = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,lamba,kappa,dt)))

@pytest.mark.parametrize("m,sigma,lamba,kappa,dt",testdataMatern)
def test_h_Matern(monkeypatch,m,sigma,lamba,kappa,dt):

    # Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{lamba:.5f}\n{kappa:.5f}\n{sigma:.5f}\n'))

    mparams = mock.Mock(side_effect = {}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"Matern",dt,"mom",control)

    snew,hnew = eletor.create_hs.create_h("Matern",m,sigma,lamba,kappa)

    differ = np.abs(hnew - h)
    assert all(differ < 1e-11)
    assert s == snew

    mparams.assert_has_calls([mock.call('lambda_fixed'),mock.call('kappa_fixed')])

    return None

testdataMatern = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,lamba,kappa,dt)))

@pytest.mark.parametrize("m,sigma,lamba,kappa,dt",testdataMatern)
def test_h_Matern2(monkeypatch,m,sigma,lamba,kappa,dt):

    # Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{kappa:.5f}\n{sigma:.5f}\n'))

    mparams = mock.Mock(side_effect = {'lambda_fixed':lamba}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"Matern",dt,"mom",control)

    snew,hnew = eletor.create_hs.create_h("Matern",m,sigma,lamba,kappa)

    differ = np.abs(hnew - h)
    assert all(differ < 1e-11)
    assert s == snew

    mparams.assert_has_calls([mock.call('lambda_fixed'),mock.call('kappa_fixed')])

    return None

testdataMatern = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,lamba,kappa,dt)))

@pytest.mark.parametrize("m,sigma,lamba,kappa,dt",testdataMatern)
def test_h_Matern3(monkeypatch,m,sigma,lamba,kappa,dt):

    # Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{lamba:.5f}\n{sigma:.5f}\n'))

    mparams = mock.Mock(side_effect = {'kappa_fixed':kappa}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"Matern",dt,"mom",control)

    snew,hnew = eletor.create_hs.create_h("Matern",m,sigma,lamba,kappa)

    differ = np.abs(hnew - h)
    assert all(differ < 1e-11)
    assert s == snew

    mparams.assert_has_calls([mock.call('lambda_fixed'),mock.call('kappa_fixed')])

    return None

testdataMatern = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,lamba,kappa,dt)))

@pytest.mark.parametrize("m,sigma,lamba,kappa,dt",testdataMatern)
def test_h_Matern4(monkeypatch,m,sigma,lamba,kappa,dt):

    # Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{sigma:.5f}\n'))

    mparams = mock.Mock(side_effect = {'kappa_fixed':kappa,'lambda_fixed':lamba}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,"Matern",dt,"mom",control)

    snew,hnew = eletor.create_hs.create_h("Matern",m,sigma,lamba,kappa)

    differ = np.abs(hnew - h)
    assert all(differ < 1e-11)
    assert s == snew

    mparams.assert_has_calls([mock.call('lambda_fixed'),mock.call('kappa_fixed')])

    return None

m = np.random.randint(10,1000,N)
sigma = np.random.uniform(0,10,N)
phi = np.random.uniform(1e-6,1,N)
dt = np.random.uniform(0.5,20,N)
#testdataVaryingAnnual = [ (10,1,0.5,1) ]
testdataAR = zip(*map(lambda x: np.round(x,3),
                              (m,sigma,phi,dt)))

@pytest.mark.parametrize("m,sigma,phi,dt",testdataAR)
def test_h_AR1(monkeypatch,m,sigma,phi,dt):

    # Phi, Sigma
    monkeypatch.setattr('sys.stdin', io.StringIO(f'{phi:.3f}\n{sigma:.3f}\n'))

    mparams = mock.Mock(side_effect = {}.__getitem__)
    mm = mock.MagicMock(**{"params.__getitem__":mparams})
    cm = mock.Mock(return_value = mm)

    with mock.patch('hectorp.control.Control',cm) as mp, \
         mock.patch('hectorp.observations.Control',cm) as mpp:

        control = hectorp.control.Control('Archivo')
        observations = hectorp.simulatenoise.Observations()

        s,h = hectorp.simulatenoise.create_h(m,'AR1',dt,"mom",control)

    snew,hnew = eletor.create_hs.create_h("AR1",m,sigma,phi,"mom",dt)

    differ = np.abs(hnew - h)
    assert all(differ < 1e-13) #slight differences for cosine and sine
    assert s == snew

    assert True

    return None
