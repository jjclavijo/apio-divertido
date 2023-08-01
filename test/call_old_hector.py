#!/usr/bin/env python3

import io
import sys
from pathlib import Path
import tempfile
import hashlib
import toml

import click

from unittest import mock

import hectorp.simulatenoise
import eletor.compat

from eletor.not_rng import rng

# from array import array
# from itertools import cycle,islice
# import numpy as np
#
# with open('10000Random.dat','rb') as f:
#     _10000_random = array('d')
#     _10000_random.fromfile(f,10000)
#
# def rng(size):
#     return np.array([*islice(cycle(_10000_random),size)])

def call_hector(controlfile,stdinput):
    print("Recién llamé")

    stdin = io.StringIO(stdinput)

    rngMock = mock.MagicMock(side_effect=rng)
    rngMock.standard_normal.side_effect=rng
    rngMock.randn.side_effect=rng
    rngMock.default_rng.side_effect=lambda *x: rngMock

    # --- Mock control ---
    with mock.patch('hectorp.simulatenoise.np.random',rngMock),\
         mock.patch('sys.stdin',stdin),\
         mock.patch('sys.argv',['simulatenoise','-i',controlfile]):

        print(sys.argv)
        hectorp.simulatenoise.main()
        hectorp.simulatenoise.Control.clear_all()

        return 0

def call_hector_with_toml(data_or_file,return_type='file'):
    # Se trabaja en un directorio temporal
    with tempfile.TemporaryDirectory() as tmpdir:
        # No voy a hacer un dispatcher aparte para decidir si pasaron una data
        # de configuración o la ruta a un archivo.
        if isinstance(data_or_file,dict):
            data = data_or_file
            file = Path(tmpdir) / 'config.toml'
            with open(file,'w') as f:
                toml.dump(data,f)
        else:
            file = data_or_file
            with open(file,'r') as f:
                data = toml.load(f)

        # Crear los archivos de configuración estilo hector
        eletor.compat.toml_to_ctl_cli(file,tmpdir)

        # Los nombres para los archivos que saca la función anterior son
        # iguales al archivo de entrada pero con la extensón cambiada.
        fname = Path(tmpdir) / Path(file.name).with_suffix('.ctl')
        fcli = fname.with_suffix('.cli')
        with open(fcli,'r') as f:
            stdin = f.read()

        # Call hector
        call_hector(str(fname),stdin)

        file = Path(data['file_config'].get('SimulationDir',''))
        file /= '{}_0.mom'.format(data['file_config']['SimulationLabel'])

        if return_type == 'file':
            return file
        elif return_type == 'hash':
            with open(file,'br') as f:
                md5 = hashlib.md5()
                md5.update(f.read())
            with open(file,'r') as f:
                _ = f.readline()
                raw_mom = [i.split()[-1] for i in f.readlines()]
            key = [*data['NoiseModels'].keys()][0]
            label = [*data['NoiseModels'][key].keys()][0]
            data['NoiseModels'][key][label]['md5'] = md5.hexdigest()
            data['NoiseModels'][key][label]['mom'] = raw_mom
            return data

@click.command
@click.option('--commandfile','-i',type=Path,multiple=True,required=True,help='Control file, .ctl o .toml')
def main(commandfile):
    for file in commandfile:
        call_hector_with_toml(file,return_type='file')

if __name__ == '__main__':
    exit(main())

