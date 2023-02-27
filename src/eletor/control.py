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

#==============================================================================
# Class definition
#==============================================================================

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]



    def clear(cls):
        _ = cls._instances.pop(cls, None)



    def clear_all(*args, **kwargs):
        SingletonMeta._instances = {}

class Control:
    """Class to store parameters that prescribe how the analysis should be done
    """
    def __init__(self, ctl_file,
                 defaults={},unatended=False):
        """This is my Control class

        Args:
            ctl_file (string) : name of text-file with parameters
        Oprional Args:
            defaults : dictionary of labels:defaultValue
                       for mandatory arguments.
                       Defaults for mandatory arguments will not be used unless
                       explicitly asked by de user or by passing the unatended
                       flag, in which case a warning will be logged.
        """

        if not os.path.exists(ctl_file):
            raise FileNotFoundError(2,"No such file or directory",ctl_file)

        self.params = {}
        self.defaults = defaults
        self.unatended = unatended

        with open(ctl_file,'r') as fp:
            for line in fp:
                cols = line.split()
                if len(cols) == 0: # Skip empty lines
                        continue

                label = cols[0]
                if len(cols) == 1:
                    if label in self.defaults:
                        raise ValueError(f'Control File: found label {label:s} but no value, is probably a mandatory option left blank for you to fill!')
                    else:
                        raise ValueError(f'Control File: found label {label:s} but no value!')

                if cols[1]=='Yes' or cols[1]=='yes':
                    self.params[label] = True
                elif cols[1]=='No' or cols[1]=='no':
                    self.params[label] = False
                elif cols[1].isdigit()==True:
                    self.params[label] = int(cols[1])
                else:
                    if self.is_float(cols[1])==True:
                        if len(cols)==2:
                            self.params[label] = float(cols[1])
                        else:
                            self.params[label] = []
                            for i in range(1,len(cols)):
                               self.params[label].append(float(cols[i]))
                    else:
                        if len(cols)==2:
                            self.params[label] = cols[1]
                        elif len(cols)>2:
                            self.params[label] = cols[1:]
                        else:
                            assert False # shouldn't come here
                            # print('found label {0:s} but no value!'.\
                            # format(label))
                            # sys.exit()

    def is_float(self,x):
        """ Check if string is float

        Args:
           x (string) : is this a float string?

        Returns:
           True is number is float
        """
        try:
            float(x)
            return True
        except ValueError:
            return False

    def get(self,parameter):
        if parameter in self.params:
            return self.params[parameter]

        elif parameter in self.defaults:
            p = self.defaults[parameter]
            if p is None:
                raise ValueError(f"Parameter {parameter} is mandatory and has no default value")

            elif self.unatended or \
                 (input(f'Parameter {parameter} is mandatory, use default value of <{p}>? [y/N]: ')\
                  in ('y','Y')):
                return p

            raise ValueError(f"Parameter f{parameter} is mandatory")

        raise ValueError("Trying to get undefined parameter f{parameter}")

    def __getitem__(self,key):
        return self.get(key)

    def __setitem__(self,key,newvalue):
        self.params[key] = newvalue

#===============================================================================
# Control parameter verification functions.
#===============================================================================
"""
These are functions meant to be run just after initialization of the
Control instance.
Each function checks and ask for correction of errors in the parameter structure
"""

class mustAsk(object):
    def __repr__(self):
        return 'MustAsk'
    def __str__(self):
        return 'MustAsk'

MustAsk = mustAsk() # A marker for need to ask for parameter.

# The order of parameter reading form commandline in the original version is:
# TODO: Keep this order for compatibility.

_paramorder =( ('White',('Sigma')),
               ('Powerlaw',('Kappa','Sigma')),
               ('Flicker',('Sigma')),
               ('RandomWalk',('Sigma')),
               ('GGM',('GGM_1mphi','Kappa','Sigma')),
               ('VaryingAnnual',('Phi','Sigma')),
               ('Matern',('Lambda','Kappa','Sigma')),
               ('AR1',('Phi','Sigma')) )

def _check_stdin_params(control):
    modelorder = []
    for model,_ in _paramorder:
        modelorder.extend(\
             [ix for ix, v in enumerate(control['NoiseModels']) if v == model]\
             )
    for ix in modelorder:
        model = control['NoiseModels'][ix]
        for parameter in {i:j for i,j in _paramorder}[model]:
            if control[parameter] is MustAsk:
                print(f'Please Specify {parameter} for model type <{model}> : ', end='')
                parvalue = float(input())
                control[parameter] = parvalue
                continue
            try:
                if control[parameter][ix] is MustAsk:
                    print(f'Please Specify {parameter} for model <{ix}:{model}> : ', end='')
                    parvalue = float(input())
                    control[parameter][ix] = parvalue
                    continue
            except TypeError as t:
                assert 'subscriptable' in t.args[0] #only get here if control[...] is not a list

    return None


def _check_missing_data(control):
    """
    if incomplete MissingData specification
    no missing data is used
    """
    # log.warning('Missing data options misspesified, ignoring')
    if 'MissingData' in control.params and\
            'PercMissingData' in control.params:
        pass
    else:
        control.params['MissingData'] = False
        control.params['PercMissingData'] = 0.0

def _check_trend_bias(control):
    """
    if incomplete linear terms specification
    no linear term is used
    """
    # log.warning('Linear terms options misspesified, ignoring')
    if 'Trend' in control.params and\
            'NominalBias' in control.params:
        pass
    else:
        control.params['Trend'] = 0.0
        control.params['NominalBias'] = 0.0


def _check_sigma(control):
    """
    check noise amplitude for any model
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        try:
            sigma = control.get('Sigma')[ix]
        except ValueError:
            control['Sigma'] = [None] * len(control.get('NoiseModels'))
            sigma = None
        except IndexError:
            raise ValueError(f'Sigma parameter list has wrong length, should be at least {ix}')

        if sigma is None:
            if control.unatended:
                raise ValueError(f'Should specify noise Amplitude Sigma for noise model {nm}')

            # print(f'Please Specify noise Amplitude Sigma for model <{ix}:{nm}> : ', end='')
            # sigma = float(input())
            control.get('Sigma')[ix] = MustAsk

def _check_kappa(control):
    """
    check kappa parameter for GGM, Powerlaw or Matern Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm in ['Powerlaw', 'GGM', 'Matern']:
            try:
                kappa = control.get('Kappa')[ix]
            except ValueError:
                control['Kappa'] = [None] * len(control.get('NoiseModels'))
                kappa = None
            except IndexError:
                raise ValueError(f'Kappa parameter list has wrong length, should be at least {ix}')

            if nm == 'Matern':
                try:
                    kappa = control.get('kappa_fixed')
                    control.get('Kappa')[ix] = kappa
                except ValueError:
                    pass

            if kappa is None:
                if control.unatended:
                    raise ValueError(f'Should specify Spectral index kappa for noise model {nm}')

                # print(f'Please Specify Spectral index kappa for model <{ix}:{nm}> : ', end='')
                # kappa = float(input())
                control.get('Kappa')[ix] = MustAsk

            else:
                if (nm in ['Powerlaw', 'GGM']) and (kappa<-2.0-EPS or kappa>2.0+EPS):
                    raise ValueError('kappa shoud lie between -2 and 2 : {0:f}'.format(kappa))
                elif nm == 'Matern' and (kappa > -0.5):
                    # The cited paper on Matern processes limits alfa to > 0.5,
                    # implying this.
                    raise ValueError('kappa shoud be lower than -0.5 : {0:f}'.format(kappa))

def _check_lambda(control):
    """
    check kappa parameter for Matern Noise models
    """
    EPS = 1.0e-8
    for ix,nm in enumerate(control.get('NoiseModels')):
        if nm == 'Matern':
            try:
                lamba = control.get('Lambda')[ix]
            except ValueError:
                control['Lambda'] = [None] * len(control.get('NoiseModels'))
                lamba = None
            except IndexError:
                raise ValueError(f'Lambda parameter list has wrong length, should be at least {ix}')

            try:
                lamba = control.get('lambda_fixed')
                control.get('Lambda')[ix] = lamba
            except ValueError:
                pass

            if lamba is None:
                if control.unatended:
                    raise ValueError(f'Should specify Lambda parameter for noise model {nm}')

                # print(f'Please Specify Lambda Parameter for model <{ix}:{nm}> : ', end='')
                # lamba = float(input())
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
            try:
                phi = control.get('Phi')[ix]
            except ValueError:
                    control['Phi'] = [None] * len(control.get('NoiseModels'))
                    phi = None
            except IndexError:
                if isinstance( (phi := control.get('Phi'))
                              ,float):
                    control['Phi'] = [phi] * len(control.get('NoiseModels'))
                else:
                    raise ValueError(f'Phi parameter list has wrong length, should be at least {ix}')

            try:
                phi = control.get('phi_fixed')
                control.get('Phi')[ix] = lamba
            except ValueError:
                pass

            if phi is None:
                if control.unatended:
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
                phi = control.get('GGM_1mphi')[ix]
            except ValueError:
                    control['GGM_1mphi'] = [None] * len(control.get('NoiseModels'))
                    phi = None
            except (IndexError, TypeError):
                if isinstance( (phi := control.get('GGM_1mphi'))
                              ,float):
                    control['GGM_1mphi'] = [phi] * len(control.get('NoiseModels'))
                else:
                    raise ValueError(f'GGM 1-phi parameter list has wrong length, should be at least {ix}')

            if phi is None:
                if control.unatended:
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


control_defaults = {
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

def get_nm_parameter(ix,model,control,parameter):
    if parameter == 'NumberOfPoints':
        return control['NumberOfPoints']
    if parameter == 'Sigma':
        return control['Sigma'][ix]
    if parameter == 'Kappa':
        if model == 'Matern':
            try:
                return control['kappa_fixed']
            except (ValueError,KeyError):
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
            except (ValueError,KeyError):
                pass
        return control['Phi'][ix]
    if parameter == 'Lambda':
        if model == 'Matern':
            try:
                return control['lambda_fixed']
            except (ValueError,KeyError):
                pass
        return control['Lambda'][ix]

    raise KeyError(f'Incorrect parameter {parameter} for required model {model}')

def get_noisemodels(control):
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

def parse_retro_ctl(fname):
    checks = [
            _check_missing_data,
            _check_trend_bias,
            _check_noise_models_is_list,
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

    # Parsear el control File
    control = Control(fname,control_defaults)

    # Correr los chequeos y pedir entrada del usuario
    for check in checks:
        check(control)

    # tiramos la Clase molesta a la basura, queda el dict.
    control = control.params

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

