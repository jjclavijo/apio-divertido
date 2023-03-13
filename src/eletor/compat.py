# control.py
#
# This file is part of HectorP 0.1.1
#
# HectorP is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# HectorP is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# HectorP. If not, see <https://www.gnu.org/licenses/>.
#
# 29/12/2021  Machiel Bos, Santa Clara
#==============================================================================

import os
import sys
from pathlib import Path
from ast import literal_eval
from math import isnan
import toml

#==============================================================================
# Class definition
#==============================================================================

def cast(string):
    try:
        return literal_eval(string)
    except (ValueError,SyntaxError):
        # era string o Yes/No
        try:
            return {'yes':True,'no':False}[string.lower()]
        except KeyError:
            # era string
            return string

def ctl_parser(ctl_file):
    """This is my Control class

    Args:
        ctl_file (string) : name of text-file with parameters
    """
    # Read file
    with open(ctl_file,'r') as f:
        data = f.readlines()

    # Split Lines
    data = [*map(str.split,data)]

    # Cast string a Int, Float, bool o dejarlo como estaba
    # el map es porque es para cada elemento de cada línea
    data = [[*map(cast,x)] for x in data]

    # de cada línea el primer elemento es la etiqueta y el resto es parámetro
    # El if es para pasar elementos simples en lugar de listas de 1.
    # Es regla poner un oneliner en cada función para conservar mi identidad.
    data = {line.pop(0):(line[0] if len(line) == 1 else line) for line in data}

    return data

#===============================================================================
# Control parameter verification functions.
#===============================================================================
"""
Todos estos chequeos que vienen acá son para garantizar que las opciones que
no entran por el .ctl se pasen por línea de comandos en el momento que se
intenta transformar un .ctl a .toml.

De paso se chequea que estén en los rangos adecuados.

NO VOY A DOCUMENTAR ESTO, ES FEO, ESTÁ COPIADO Y PEGADO DE CUALQUIER MANERA,
PERO ANDA. Y ES MUY POCO PROBABLE QUE LO NECESITEMOS.
"""

class mustAsk(object):
    def __repr__(self):
        return 'MustAsk'
    def __str__(self):
        return 'MustAsk'

MustAsk = mustAsk() # A marker for "need to ask" for parameter.

# The order of parameter reading form commandline in the original version is:
# TODO: Keep this order for compatibility.

_parameter_order ={ 'White':['Sigma'],
                   'Powerlaw':['Kappa','Sigma'],
                   'Flicker':['Sigma'],
                   'RandomWalk':['Sigma'],
                   'GGM':['GGM_1mphi','Kappa','Sigma'],
                   'VaryingAnnual':['Phi','Sigma'],
                   'Matern':['Lambda','Kappa','Sigma'],
                   'AR1':['Phi','Sigma'] }

def _check_stdin_params(control):
    """
    Pide parámetros por línea de comandos si es necesario.
    """

    # Para cada modelo especificado
    for ix,model in enumerate(control['NoiseModels']):
        # Preguntar los parámetros en el mismo orden
        # que hector lo haría
        for parameter in _parameter_order[model]:
            # Si tenemos anotado que hay que preguntar
            # Y es un paámetro global
            if control[parameter] is MustAsk:
                print(f'Please Specify {parameter} for model type <{model}> : ', end='')
                parvalue = float(input())
                control[parameter] = parvalue
                continue
            try:
                # Si era un parámetro de lista, porque hay uno por modelo
                if control[parameter][ix] is MustAsk:
                    # Si anotamos preguntar
                    print(f'Please Specify {parameter} for model <{ix}:{model}> : ', end='')
                    parvalue = float(input())
                    control[parameter][ix] = parvalue
                    continue
            except TypeError as t:
                #only get here if control[...] is not a list
                assert 'subscriptable' in t.args[0]

    return None


def _check_missing_data(control):
    """
    if incomplete MissingData specification
    no missing data is used
    """
    # log.warning('Missing data options misspesified, ignoring')
    if 'MissingData' in control and\
            'PercMissingData' in control:
        pass
    else:
        control['MissingData'] = False
        control['PercMissingData'] = 0.0

def _check_trend_bias(control):
    """
    if incomplete linear terms specification
    no linear term is used
    """
    # log.warning('Linear terms options misspesified, ignoring')
    if 'Trend' in control and\
            'NominalBias' in control:
        pass
    else:
        control['Trend'] = 0.0
        control['NominalBias'] = 0.0

_paramorder ={ 'White':('Sigma'),
               'Powerlaw':('Kappa','Sigma'),
               'Flicker':('Sigma'),
               'RandomWalk':('Sigma'),
               'GGM':('GGM_1mphi','Kappa','Sigma'),
               'VaryingAnnual':('Phi','Sigma'),
               'Matern':('Lambda','Kappa','Sigma'),
               'AR1':('Phi','Sigma') }

def _chech_parameter_list_lengths(control):
    parameters = ['Sigma']
    models = control['NoiseModels']

    if set(models).intersection(('Powerlaw','GGM','Matern')):
        parameters.append('Kappa')

    if set(models).intersection('AR1','VaryingAnnual'):
        parameters.append('Phi')

    if 'Matern' in models:
        parameters.append('Lambda')

    for parameter in parameters:
        try:
            param = control[parameter][ix]
        except KeyError: #no parameter
            control[parameter] = [None] * len(models)
        except IndexError: # list index failed
            raise ValueError(f'{parameter} parameter list has wrong length, should be at least {ix}')

def _check_sigma(control):
    """
    check noise amplitude for any model
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        sigma = control['Sigma'][ix]

        if sigma is None:
            if control.get('unatended',False):
                raise ValueError(f'Should specify noise Amplitude Sigma for noise model {nm}')

            control['Sigma'][ix] = MustAsk

def _check_kappa(control):
    """
    check kappa parameter for GGM, Powerlaw or Matern Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm in ['Powerlaw', 'GGM', 'Matern']:
            if nm == 'Matern':
                # Mattern noise admits kappa parameter in .ctl
                try:
                    kappa = control['kappa_fixed']
                    control['Kappa'][ix] = kappa
                except KeyError:
                    pass

            kappa = control['Kappa'][ix]

            if kappa is None:
                if control.get('unatended',False):
                    raise ValueError(f'Should specify Spectral index kappa for noise model {nm}')

                control['Kappa'][ix] = MustAsk

            else:
                if (nm in ['Powerlaw', 'GGM']) and (kappa<-2.0-EPS or kappa>2.0+EPS):
                    raise ValueError('kappa shoud lie between -2 and 2 : {0:f}'.format(kappa))
                elif nm == 'Matern' and (kappa > -0.5):
                    # The cited paper on Matern processes limits alfa to > 0.5,
                    # implying this.
                    raise ValueError('kappa shoud be lower than -0.5 : {0:f}'.format(kappa))

def _check_lambda(control):
    """
    check lambda parameter for Matern Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm == 'Matern':
            try:
                lamba = control['lambda_fixed']
                control.get('Lambda')[ix] = lamba
            except KeyError:
                pass

            lamba = control['Lambda'][ix]

            if lamba is None:
                if control.get('unatended',False):
                    raise ValueError(f'Should specify Lambda parameter for noise model {nm}')

                control.get('Lambda')[ix] = MustAsk

            elif (lamba < EPS):
                # The cited paper on Matern processes limits lambda to be
                # positive
                raise ValueError(f'Lambda should be positive : {lamba:f}')

def _check_phi(control):
    """
    check Phi parameter for AR1 or VaryingAnual Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm in ['AR1','VaryingAnual']:
            if nm == 'VaryingAnual':
                try:
                    phi = control.get('phi_fixed')
                    control['Phi'][ix] = phi
                except ValueError:
                    pass

            phi = control['Phi'][ix]

            if phi is None:
                if control.get('unatended',False):
                    raise ValueError(f'Should specify Phi parameter for noise model {nm}')

                #print(f'Please Specify Phi Parameter for model <{ix}:{nm}> : ', end='')
                #phi = float(input())
                control.get('Phi')[ix] = MustAsk


def _check_ggmphi(control):
    """
    check 1-phi parameter for GGM Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm == 'GGM':
            try:
                phi = control.get['GGM_1mphi'][ix]
            except KeyError:
                    control['GGM_1mphi'] = [None] * len(control.get('NoiseModels'))
                    phi = None
            except TypeError:
                control['GGM_1mphi'] = [phi] * len(control.get('NoiseModels'))
            except IndexError:
                raise ValueError(f'GGM 1-phi parameter list has wrong length, should be at least {ix}')

            if phi is None:
                if control.get('unatended',False):
                    raise ValueError(f'Should specify 1-phi parameter for noise model {nm}')

                # print(f'Please Specify 1-phi Parameter for model <{ix}:{nm}> : ', end='')
                # phi = float(input())
                control.get('GGM_1mphi')[ix] = MustAsk

            elif phi<0.0 or phi>1.0+EPS:
                # The cited paper on Matern processes limits lambda to be
                # positive
                raise ValueError(f'1-phi should lie between 0 and 1: : {phi:f}')

def _check_noise_models_is_list(control):
    """
    Ensure NoiseModels is a list
    """
    if not isinstance(nm := control.get('NoiseModels'),list):
        control['NoiseModels'] = [nm]

#
# ================================================================
#  Parsear el archivo .ctl y generar el mismo dict que desde toml
# ================================================================
#

# Todas estas variables tienen que ver con saber qué parametros tengo que
# leer para cada modelo y saber los defaults que tengo que devolver.


# model specific parameters

_paramlist = {'White':('NumberOfPoints','Sigma'),
              'Powerlaw':('NumberOfPoints','Sigma','Kappa','TS_format','SamplingPeriod'),
              'Flicker':('NumberOfPoints','Sigma','TS_format','SamplingPeriod'),
              'RandomWalk':('NumberOfPoints','Sigma','TS_format','SamplingPeriod'),
              'GGM':('NumberOfPoints','Sigma','Kappa','GGM_1mphi','TS_format','SamplingPeriod'),
              'VaryingAnnual':('NumberOfPoints','Sigma','Phi','TS_format','SamplingPeriod'),
              'Matern':('NumberOfPoints','Sigma','Lambda','Kappa'),
              'AR1':('NumberOfPoints','Sigma','Phi')}

# TODO: nuevos nombres para las variables, preferentemente con prefijo
_param_conversion = {'NumberOfPoints':'m',
                     'TS_format':'units',
                     'SamplingPeriod':'dt',
                     'GGM_1mphi':'one_minus_phi',
                     'Lambda':'lamba'}

_param_conversion_r = {v:k for k,v in _param_conversion.items()}

def get_nm_parameter(ix,model,control,parameter):
    """
    Cada modelo tiene sus propios parámetros requeridos.

    En el archivo viejo los parámetros se pasan en listas.
    Por eso entra el index al que corresponde el modelo a analizar
    y el diccionario de control.

    Además, hay modelos en los que hay dos formas de indicar algunos
    parámetros (por línea de comandos o por una variable en control.
    por eso existe esta función.
    """
    if parameter == 'NumberOfPoints':
        return control['NumberOfPoints']
    if parameter == 'Sigma':
        return control['Sigma'][ix]
    if parameter == 'Kappa':
        if model == 'Matern':
            try:
                return control['kappa_fixed']
            except KeyError:
                pass
        return control['Kappa'][ix]
    if parameter == 'TS_format':
        return control['TS_format']
    if parameter == 'SamplingPeriod':
        return control['SamplingPeriod']
    if parameter == 'GGM_1mphi':
        if isinstance((ggm := control['GGM_1mphi']),list):
            return ggm[ix]
        else:
            return ggm
    if parameter == 'Phi':
        if model == 'VaryingAnnual':
            try:
                return control['phi_fixed']
            except KeyError:
                pass
        return control['Phi'][ix]
    if parameter == 'Lambda':
        if model == 'Matern':
            try:
                return control['lambda_fixed']
            except KeyError:
                pass
        return control['Lambda'][ix]

    raise KeyError(f'Incorrect parameter {parameter} for required model {model}')

def get_noisemodels(control):
    """
    esta funcion es un helper para leer los parámetros de cada modelo que esté
    en el .ctl, entra el diccionario en formato viejo y sale una lista de
    tipo_de_modelo, parametro
    """
    models = control["NoiseModels"]
    if not isinstance(models,list):
        models = [models]

    rmodels = []

    for ix,model in enumerate(models):
        params = {}
        # iterate over model_specific parameters
        for p in _paramlist[model]:
            # some parameters are renamed
            if p in _param_conversion:
                k = _param_conversion[p]
            # other are only lowered
            else:
                k = p.lower()
            # get and store each parameter
            params[k] = get_nm_parameter(ix,model,control,p)
        rmodels.append( (model,
                         params) )

    return rmodels

_control_defaults = {
        'SimulationDir':None,
        'SimulationLabel':None,
        'TS_format':'mom',
        'NumberOfSimulations':1,
        'NumberOfPoints':365,
        'SamplingPeriod':1,
        'TimeNoiseStart':0,
        'NoiseModels':None,
        'RepeatableNoise':False,
        'MissingData':False,
        'PercMissingData':0.0,
        'Offsets':False,
        'Trend':0,
        'NominalBias':0,
        'AnnualSignal':0}

_general_params = [
        'NumberOfSimulations',
        'NumberOfPoints',
        'TimeNoiseStart',
        'SamplingPeriod',
        'RepeatableNoise',
        'MissingData',
        'PercMissingData',
        'Offsets',
        'Trend',
        'NominalBias',
        'AnnualSignal']

_file_config_params = [
        'SimulationDir',
        'SimulationLabel',
        'TS_format']

def parse_retro_ctl(fname):
    """
    Parsear un viejo archivo ctl para convertirlo a un nuevo toml
    Devuelve un diccionario en el nuevo formato.
    """
    checks = [
            _check_missing_data,
            _check_trend_bias,
            _check_noise_models_is_list,
            _chech_parameter_list_lengths,
            _check_lambda,
            _check_kappa,
            _check_sigma,
            _check_ggmphi,
            _check_phi,
            _check_stdin_params,
            _check_lambda, # repeat boundary checks.
            _check_kappa, # repeat boundary checks.
            _check_sigma, # repeat boundary checks.
            _check_ggmphi, # repeat boundary checks.
            _check_phi # repeat boundary checks.
             ]

    # Parsear el control File. No mas Control, es un diccionario común.
    control_ = ctl_parser(fname)
    # reemplazar defaults si necesita:
    control = {**_control_defaults}
    control.update(**control_)

    # Correr los chequeos, pedir entrada del usuario y algun,
    # que no quiero ni recordar, como garantizar que NoiseModels es una
    # Lista.
    # Varios de estos chequeos son recortes de partes del hector original
    for check in checks:
        check(control)

    # Empezamos a formatear el diccionario igual que si viniera del toml
    new_control = {'file_config': {k:v for k,v in control.items() \
                                       if k in _file_config_params},
                   'general': {k:v for k,v in control.items() \
                                   if k in _general_params},
                   'NoiseModels': {k:{} for k in control['NoiseModels']}
                   }

    # Iteramos sobre los modelos
    for model,params in get_noisemodels(control):

        #OJO: cada tipo de modelo puede estar repetido.
        models_count= len(new_control['NoiseModels'][model])
        #OJO: en vez de params en el modelo va un contador.
        k = f'{models_count+1}'

        new_control['NoiseModels'][model][k] = params

    return new_control

from itertools import chain

def dict_to_ctl(control_data):
    """
    Convierte un diccionario cargado desde un toml de configuración al formato
    que usa hector:

    Devuelve:
        List[str] : para escribir al archivo .ctl
        List[float] : parámetros para pasar de a uno por .cli
    """

    # Todo lo que es file_config y general para de una
    lines = [f"{k:20}{v}" for k,v in control_data['file_config'].items() ]
    lines.extend([f"{k:20}{v}" for k,v in control_data['general'].items() ])

    # Obvio que hay formas menos villeras de hacer esto, pero la idea es
    # que cada tipo de modelo puede estar más de una vez. Y no tengo tiempo de
    # pensar ahora.
    # OJO: el nombre del modelo (por ejemplo "caca" en NoiseModels.GGM.caca)
    # Es ignorado alevosamente.

    noise_models_labels = [[(k,k1) for k1 in v.keys()] for k,v in control_data['NoiseModels'].items() ]
    noise_models_labels = [*chain(*noise_models_labels)]

    noise_models = [m for m,l in noise_models_labels]

    lines.append("{:20}{}".format("NoiseModels",
                                  ' '.join(noise_models)))

    # El resto de los parámetros se pasan por línea de commandos.
    _paramorder_new ={ 'White':['sigma'],
                       'Powerlaw':['kappa','sigma'],
                       'Flicker':['sigma'],
                       'RandomWalk':['sigma'],
                       'GGM':['one_minus_phi','kappa','sigma'],
                       'VaryingAnnual':['phi','sigma'],
                       'Matern':['lamba','kappa','sigma'],
                       'AR1':['phi','sigma'] }

    cli_options = []

    for model,label in noise_models_labels:
        params = control_data['NoiseModels'][model][label]
        # Si usaramos los parámetros opcionales _fixed o GGM_1mphi del
        # .ctl, deberíamos chequear aparte acá para no repetir. Un lío
        # no tiene sentido.
        cli_options.extend([params[i] for i in _paramorder_new[model]])

    return lines, cli_options

def toml_to_ctl_cli(fname,dirname=None):
    """
    Convertir el archivo toml a un archivo .ctl (input de hector) y uno .cli
    (para pasarle a hector por línea de comandos)

    Args:

    fname : nombre de archivo
    dirname : opcional directorio de salida
    """

    # Usar un Path para poder cambiár el sufijo más facil.
    fname = Path(fname)

    # Cargar los datos del toml
    with fname.open('r') as f:
        control_data = toml.load(f)

    # Convertir el toml separando los parámetros del archivo de control
    # y de la linea de comandos
    control_lines, cli = dict_to_ctl(control_data)

    # Agregar directorio al path si lo hay (y es válido)
    if not dirname is None:
        directory = Path(dirname)
        if not directory.is_dir():
            raise ValueError("bad directory")
        fname = directory/fname.name

    # Escribir el ctl
    with fname.with_suffix('.ctl').open('w') as f:
        # Este oneliner te lo dejo para molestarte.
        f.writelines(map(lambda x: x+'\n', control_lines))

    # Escribir el cli
    with fname.with_suffix('.cli').open('w') as f:
        f.writelines(map('{}\n'.format, cli))

    return 0

def vector_to_mom(*,y,ix=None,sampling_period=None):
    """
    Escribir los datos <y> con el índice <ix> en un archivo de nombre <fname>
    con el formato de archivo que usa hector (.mom).

    Hay que pasarle el sampling_period porque hector lo escribe en el
    encabezado.
    """

    if ix is None:
        if sampling_period is None:
            raise ValueError('Must pass one of ix|sampling_period')
        else:
            t0 = 51544.0
            ix = [t0+m*sampling_period for m in range(len(y))]
    else:
        # Asumimos sampling_period regular.
        # esto es todo herencia asi que no nos vamos a poner puristas
        sampling_period = ix[1]-ix[0]

    # generar las líneas sin formatear, eliminando las que sean nan.
    data = [(i,d) for i,d in zip(ix,y) if not isnan(d)]

    datalines = ['# sampling period {0:f}\n'.format(sampling_period)]

    # El tipo de cosas que le gustan a Gonzalo
    formatter=lambda x:'{:12.6f} {:13.6f}\n'.format(*x)

    # formatear líneas.
    datalines.extend(map(formatter, data))

    return datalines

"""
Usage: python -m eletor.control [-I] <file1> <file2> ....

Convert .ctl files to .toml

-I: inverse conversion. Output ctl and .cli file with cli parameters.

"""

if __name__ == '__main__':
    import sys
    import toml
    from pathlib import Path

    files = [i for i in sys.argv[1:] if i[0] != '-']

    if '-I' in sys.argv[1:]:
        for arg in files:
            toml_to_ctl_cli(arg)

        arg = Path(arg)
        print("Generated .ctl and .cli files.")
        print("Run hector using:")
        print("simulatenoise -i {} < {}".format(arg.with_suffix('.ctl'),
                                                arg.with_suffix('.cli')))

    else:
        for arg in files:
            fname = Path(arg)
            control_data = parse_retro_ctl(fname)
            with fname.with_suffix('.toml').open('w') as f:
                toml.dump(control_data,f)
