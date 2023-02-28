#!/usr/bin/env python3

import io
import os
import sys
import toml
from pathlib import Path
import tempfile

import click

from unittest import mock

import logging

LOGGER = logging.getLogger(__name__)

import hectorp.simulatenoise
import eletor.control

from array import array
from itertools import cycle,islice
import numpy as np

with open('10000Random.dat','rb') as f:
    _10000_random = array('d')
    _10000_random.fromfile(f,10000)

def rng(size):
    return np.array([*islice(cycle(_10000_random),size)])

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

@click.command
@click.option('--commandfile','-i',type=Path,multiple=True,required=True,help='Control file, .ctl o .toml')
def main(commandfile):
    for file in commandfile:
        with tempfile.TemporaryDirectory() as tmpdir:
            eletor.control.toml_to_ctl_cli(file,tmpdir)
            fname = Path(tmpdir) / Path(file.name).with_suffix('.ctl')
            fcli = fname.with_suffix('.cli')
            with open(fcli,'r') as f:
                stdin = f.read()
            call_hector(str(fname),stdin)

if __name__ == '__main__':
    exit(main())

