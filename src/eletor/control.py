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

from functools import singledispatchmethod

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

class ControlClass:
    """Class to store parameters that prescribe how the analysis should be done
    """
    # Implement singledispatchmethod for dict or file
    # https://realpython.com/python-multiple-constructors/
    @singledispatchmethod
    def __init__(self, ctl_file=None,
                 defaults={},opt_defaults={},path_provided=False,unatended=False):
        raise NotImplementedError(f"could not initialize control from {type(ctl_file)}")

    @__init__.register(type(None))
    def _from_none(self, ctl,
                 defaults={},opt_defaults={},path_provided=False,unatended=False):
        fname = input(f'No config file, write defaults to: ')
        if os.path.exists(ctl_file):
            raise FileExistsError("file already exists")
        else:
            self.generate_default(fname,defaults,opt_defaults)

    @__init__.register(str)
    def _from_file(self, ctl_file,
                 defaults={},opt_defaults={},unatended=False,hooks=[]):
        """This is my Control class

        Args:
            ctl_file (string) : name of text-file with parameters
        Oprional Args:
            defaults : dictionary of labels:(defaultValue,validValues,comments)
                       for mandatory arguments.
                       Defaults for mandatory arguments will not be used unless
                       explicitly asked by de user or by passing the unatended
                       flag, in which case a warning will be logged.
            opt_defaults : dictionary of labels:(defaultValue,validValues,comments)
                           for optional argumens

        ** ValidValues and comments are ignored in this version, planed for
        ** automatic checks and automatic writing of detailed sample control files
        """

        self.params = {}

        self.defaults = defaults
        self.opt_defaults = opt_defaults
        self.unatended = unatended

        if not os.path.exists(ctl_file):
            # print('Cannot open {0:s}'.format(ctl_file))
            # sys.exit()

            # If path was not explicitly provided by user and default options
            # Where provided, offer writing a default configuration file.
            if defaults:
                if unatended or \
                   (input(f'No config file, write defaults to <{ctl_file}> [y/N]: ')\
                    in ('y','Y')):
                    self.generate_default(ctl_file,defaults,opt_defaults)
            else:
                raise FileNotFoundError(2,"No such file or directory",ctl_file)

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

        #run check hooks
        for f in hooks:
            f(self)

    @__init__.register(dict)
    def _from_dict(self, ctl_dict,
                 defaults={},opt_defaults={},path_provided=False,unatended=True,hooks=[]):
        """This is my Control class

        Args:
            ctl_dict (dictionary) : dictionary with parameters
        Oprional Args:
            defaults : dictionary of labels:(defaultValue,validValues,comments)
                       for mandatory arguments.
                       Defaults for mandatory arguments will not be used unless
                       explicitly asked by de user or by passing the unatended
                       flag, in which case a warning will be logged.
            opt_defaults : dictionary of labels:(defaultValue,validValues,comments)
                           for optional argumens

        ** ValidValues and comments are ignored in this version, planed for
        ** automatic checks and automatic writing of detailed sample control files
        """
        self.params = ctl_dict
        self.defaults = defaults
        self.opt_defaults = opt_defaults
        self.unatended = unatended

        #run check hooks
        for f in hooks:
            f(self)

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

    def generate_default(self,fname,defaults,optdefaults):
        if os.path.exists(fname):
            raise FileExistsError("file already exists")

        with open(fname,'w') as f:
            for k,v in defaults.items():
                if isinstance(v[0],bool):
                    if v[0]:
                        v = 'Yes'
                    else:
                        v = 'No'
                elif v[0] is None:
                    v = ''
                else:
                    v = v[0]
                print(f'{k:20s}{v}', file=f)

        raise FileNotFoundError(2,"File not found, Created:",fname)

    def get(self,parameter):
        if parameter in self.params:
            return self.params[parameter]

        elif parameter in self.defaults:
            p = self.defaults[parameter]
            if p is None:
                raise ValueError(f"Parameter f{parameter} is mandatory and has no default value")

            elif self.unatended or \
                 (input(f'Parameter f{parameter} is mandatory, use default value of <{p}>? [y/N]: ')\
                  in ('y','Y')):
                return p

            raise ValueError(f"Parameter f{parameter} is mandatory")

        elif parameter in self.opt_defaults:
            return self.opt_defaults[parameter]

        raise ValueError("Trying to get an undefined parameter")

    def __getitem__(self,key):
        return self.get(key)

    def __setitem__(self,key,newvalue):
        if key in self.params:
            self.params[key] = newvalue

        elif key in self.defaults:
            self.defaults[key] = newvalue

        elif key in self.opt_defaults:
            self.opt_defaults[key] = newvalue

        else:
            # if unexistent, assign high priority
            self.params[key] = newvalue

class Control(ControlClass,metaclass=SingletonMeta):
    """
    Class to store parameters that prescribe how the analysis should be done,
    Singleton Version in order to share configuration between routines.
    """
    pass
