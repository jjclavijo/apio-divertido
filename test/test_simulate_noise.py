#!/usr/bin/env python3

import pytest

import toml
import tempfile

from pathlib import Path
from hashlib import md5

from eletor.simulatenoise import simulate_noise

#
# Test de simulate_noise por casos
#
def test_simulate_noise_cases(hector_test_cases,modelo,etiqueta,parametros):
    # Crear la estructura vacía de configuración
    data = {'general':{},'file_config':{},'NoiseModels':{modelo:{}}}

    # Opciones generales por defecto.
    data['general']['NumberOfSimulations'] = 1
    data['general']['TimeNoiseStart'] = 0
    data['general']['NumberOfPoints'] = parametros['m']
    data['general']['SamplingPeriod'] = parametros['dt']
    data['general']['DeterministicNoise'] = True

    # Agregar las opciones del modelo a testear
    data['NoiseModels'][modelo][etiqueta] = parametros

    # Usamos un archivo temporal, asi se borra solo.
    # El sufijo es importante porque por compatibilidad con
    # hector siempre se agregaba ese sufijo.
    with tempfile.NamedTemporaryFile(suffix='_0.mom') as f:
        # Configuracion de archivo de salida para hector
        fname = Path(f.name)
        data['file_config']['SimulationLabel'] = fname.name.replace('_0.mom','')
        data['file_config']['SimulationDir'] = str(fname.parent)

        # Correr la simulación
        simulate_noise(data)

        # Generar el Hash
        md5hash = md5()
        with fname.open(mode='rb') as f:
            bdata = f.read()
            md5hash.update(bdata)

        # En caso de que hubiera algún test que fallara y hubiera que
        # revisar el mom directamente, había que hacer algo como:
        # if md5hash.hexdigest() != parametros['md5']:
        #   # Grabar la salida nuestra
        #   with open('failed_test_{modelo}_{etiqueta}.mom','wb') as f:
        #     f.write(bdata)
        #
        #   # Llamar al hector original
        #   from call_old_hector import call_hector_with_toml
        #   call_hector_with_toml(data,return_type='file')
        #
        #   # Copiar la salida de hector.
        #   with fname.open(mode='rb') as f:
        #     with open('hector_{modelo}_{etiqueta}.mom','wb') as f:
        #       f.write(bdata)
        # Esto queda comentado por la premisa de que nada que
        # tenga que ver con hector tiene que quedar por default activo.

        assert md5hash.hexdigest() == parametros['md5']


