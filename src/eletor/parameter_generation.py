#!/usr/bin/env python3
import click
import numpy as np
import toml

def white(model_label,sigma=None,m=None,dt=None):
    """
    Devuelve un diccionario adecuado para incluir adentro del diccionario
    "NoiseModels" en el toml. Cuidado si hay mas de un modelo del mismo tipo,
    al agregarlos se pueden pisar.
    """
    model = "White"
    if sigma is None:
        sigma = np.random.uniform(0,10)
    if m is None:
        m = np.random.randint(10,1000)
    if dt is None:
        dt = np.random.uniform(0.5,20)

    params = { 'sigma':sigma,
               'm':m,
               'units':'mom',
               'dt':dt}

    return {"NoiseModels":{"White":{model_label:params}}}

def powerlaw(model_label,sigma=None,kappa=None,m=None,dt=None):
    """
    Devuelve un diccionario adecuado para incluir adentro del diccionario
    "NoiseModels" en el toml. Cuidado si hay mas de un modelo del mismo tipo,
    al agregarlos se pueden pisar.
    """
    model = "Powerlaw"
    if sigma is None:
        sigma = np.random.uniform(0,10)
    if kappa is None:
        kappa = np.random.uniform(-2,2)
    if m is None:
        m = np.random.randint(10,1000)
    if dt is None:
        dt = np.random.uniform(0.5,20)

    params= { 'sigma':sigma,
              'kappa':kappa,
              'm':m,
              'units':'mom',
              'dt':dt}

    return {"NoiseModels":{"Powerlaw":{model_label:params}}}

def flicker(model_label,sigma=None,m=None,dt=None):
    """
    Devuelve un diccionario adecuado para incluir adentro del diccionario
    "NoiseModels" en el toml. Cuidado si hay mas de un modelo del mismo tipo,
    al agregarlos se pueden pisar.
    """
    model = "Flicker"

    if sigma is None:
        sigma = np.random.uniform(0,10)
    if m is None:
        m = np.random.randint(10,1000)
    if dt is None:
        dt = np.random.uniform(0.5,20)

    params= { 'sigma':sigma,
              'm':m,
              'units':'mom',
              'dt':dt }

    return {'NoiseModels':{"Flicker":{model_label:params}}}

def randomwalk(model_label,sigma=None,m=None,dt=None):
    """
    Devuelve un diccionario adecuado para incluir adentro del diccionario
    "NoiseModels" en el toml. Cuidado si hay mas de un modelo del mismo tipo,
    al agregarlos se pueden pisar.
    """
    model = "RandomWalk"
    if sigma is None:
        sigma = np.random.uniform(0,10)
    if m is None:
        m = np.random.randint(10,1000)
    if dt is None:
        dt = np.random.uniform(0.5,20)

    params = { 'sigma':sigma,
               'm':m,
               'units':'mom',
               'dt':dt}

    return {"NoiseModels":{"RandomWalk":{model_label:params}}}

def ggm(model_label,sigma=None,one_minus_phi=None,kappa=None,m=None,dt=None):
    """
    Devuelve un diccionario adecuado para incluir adentro del diccionario
    "NoiseModels" en el toml. Cuidado si hay mas de un modelo del mismo tipo,
    al agregarlos se pueden pisar.
    """
    model = "GGM"
    if one_minus_phi is None:
        one_minus_phi = np.random.uniform(1e-6,1e-3)
    if sigma is None:
        sigma = np.random.uniform(0,10)
    if m is None:
        m = np.random.randint(10,1000)
    if dt is None:
        dt = np.random.uniform(0.5,20)
    if kappa is None:
        kappa = np.random.uniform(-2,2)

    params = { 'one_minus_phi':one_minus_phi,
               'kappa':kappa,
               'sigma':sigma,
               'm':m,
               'units':'mom',
               'dt':dt}

    return {"NoiseModels":{"GGM":{model_label:params}}}


def varyingannual(model_label,sigma=None,phi=None,m=None,dt=None):
    """
    Devuelve un diccionario adecuado para incluir adentro del diccionario
    "NoiseModels" en el toml. Cuidado si hay mas de un modelo del mismo tipo,
    al agregarlos se pueden pisar.
    """
    model = "VaryingAnnual"
    if phi is None:
        phi = np.random.uniform(1e-5,1)
    if sigma is None:
        sigma = np.random.uniform(0,10)
    if m is None:
        m = np.random.randint(10,1000)
    if dt is None:
        dt = np.random.uniform(0.5,20)

    params = { 'sigma':sigma,
               'phi':phi,
               'm':m,
               'units':'mom',
               'dt':dt}

    return {"NoiseModels":{"VaryingAnnual":{model_label:params}}}


def matern(model_label,sigma=None,lamba=None,kappa=None,m=None,dt=None):
    """
    Devuelve un diccionario adecuado para incluir adentro del diccionario
    "NoiseModels" en el toml. Cuidado si hay mas de un modelo del mismo tipo,
    al agregarlos se pueden pisar.
    """
    model = "Matern"
    if sigma is None:
        sigma = np.random.uniform(0,10)
    if m is None:
        m = np.random.randint(10,1000)
    if dt is None:
        dt = np.random.uniform(0.5,20)
    if kappa is None:
        kappa = np.random.uniform(-2,-0.5) # Kappa for Matern differs from other models
    if lamba is None:
        lamba = np.random.uniform(1,10) * 10 ** np.random.uniform(-3,-1)

    params = { 'lamba':lamba,
               'kappa':kappa,
               'sigma':sigma,
               'm':m,
               'units':'mom',
               'dt':dt}

    return {"NoiseModels":{"Matern":{model_label:params}}}


def ar1(model_label,sigma=None,phi=None,m=None,dt=None):
    """
    Devuelve un diccionario adecuado para incluir adentro del diccionario
    "NoiseModels" en el toml. Cuidado si hay mas de un modelo del mismo tipo,
    al agregarlos se pueden pisar.
    """
    model = "AR1"
    if phi is None:
        phi = np.random.uniform(1e-5,1)
    if sigma is None:
        sigma = np.random.uniform(0,10)
    if m is None:
        m = np.random.randint(10,1000)
    if dt is None:
        dt = np.random.uniform(0.5,20)

    params = { 'sigma':sigma,
               'phi':phi,
               'm':m,
               'units':'mom',
               'dt':dt}

    return {"NoiseModels":{"AR1":{model_label:params}}}

MODELS = ["white","powerlaw","flicker","randomwalk","ggm","varyingannual","matern","ar1"]

@click.command
@click.option('--n_tomls','-t',default=None,type=int,help="Number of tomls to generate")
@click.option('--n_models','-m',default=None,type=int,help="Number of model by toml")
@click.option('--n_models_max','-M',default=None,type=int,help="Maximum Number of models by toml")
@click.option('--toml_prefix',default='output_',type=str,help="prefix for toml file")
def main(n_tomls,n_models,n_models_max,toml_prefix):
    if (not n_models is None) and (not n_models_max is None):
        raise ValueError("n_models y n_models_max son mutualmente excluyentes.")

    noisemodels = []
    for i in range(n_tomls):
        tomls = []
        with open('template.toml','r') as f:
            tomls.append(f.read())

        if not n_models_max is None:
            n_models = np.random.randint(1,n_models_max+1)
        for j in range(n_models):
            ix = np.random.randint(0,len(MODELS))
            fun = globals().get(f'random_{MODELS[ix]}_parameters')
            if j == 0:
                dt = None
                m = None

            params = fun(f'{j}',dt=dt,m=m)
            values = [*params.values()][0]
            values = [*values.values()][0]
            values = [*values.values()][0]

            dt = values['dt']
            m = values['m']

            tomls.append(toml.dumps(params))

        toml_string = ''.join(tomls)

        toml_data = toml.loads(toml_string)

        toml_data['general']['NumberOfPoints'] = m
        toml_data['general']['SamplingPeriod'] = dt
        toml_data['file_config']['SimulationLabel'] = 'test_{}'.format(i)

        with open(f'{toml_prefix}{i:04d}.toml','w') as f:
            toml.dump(toml_data,f)

if __name__ == '__main__':
    exit(main())
