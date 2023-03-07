import toml
import os

import eletor.parameter_generation as pgen
from call_old_hector import call_hector_with_toml
from tempfile import TemporaryDirectory

import numpy as np


def create_model_data(n,model):
    model = model.lower()
    model_fun = getattr(pgen,model)

    for i in range(n):
        label = f'{i}'
        yield label,model_fun(label)

MODELS = ['White','Powerlaw','Flicker','RandomWalk','AR1','VaryingAnnual','GGM','Matern']

def main(n):
    data_to_write = []
    cwdir = os.getcwd()
    for model in MODELS:
        for l,data in create_model_data(n,model):
            print(l,data)
            data['general'] = {}
            data['file_config'] = {}
            data['general']['NumberOfSimulations'] = 1
            data['general']['TimeNoiseStart'] = 0
            data['general']['NumberOfPoints'] = data['NoiseModels'][model][l]['m']
            data['general']['SamplingPeriod'] = data['NoiseModels'][model][l]['dt']
            data['file_config']['SimulationLabel'] = 'test_'
            data['file_config']['SimulationDir'] = '.'
            with TemporaryDirectory() as tempdir:
                os.chdir(tempdir)
                call_hector_with_toml(data,return_type='hash')
                data_to_write.append(data['NoiseModels'])

    os.chdir(cwdir)
    with open('test_cases.toml','w') as f:
        lines_to_write = map(toml.dumps,data_to_write)
        f.writelines(lines_to_write)

    return 0

if __name__ == "__main__":
    np.random.seed(0)
    exit(main(10))
