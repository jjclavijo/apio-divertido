# observations.py
#
# A simple interface that reads and writes mom-files and stores
# them into a Python class 'observations'.
#
# This file is part of HectorP 0.1.1.
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
#  31/1/2019  Machiel Bos, Santa Clara
#   1/5/2020  David Bugalho
# 29/12/2021  Machiel Bos, Santa Clara
#  7/ 2/2022  Machiel Bos, Santa Clara
# 10/ 7/2022  Machiel Bos, Santa Clara
#==============================================================================

import pandas as pd
import numpy as np
import os
import sys
import math
import re
from pathlib import Path

#==============================================================================
# Class definition
#==============================================================================

class Observations():
    """Class to store my time series together with some metadata

    Methods
    -------
    momread(fname)
        read mom-file fname and store the data into the mom class
    momwrite(fname)
        write the momdata to a file called fname
    msfread(fname)
        read msf-file fname and store the data into the mom class
    msfwrite(fname)
        write the msfdata to a file called fname
    make_continuous()
        make index regularly spaced + fill gaps with NaN's
    """

    def __init__(self,control={}):
        """This is my time series class

        This constructor defines the time series in pandas DataFrame data,
        list of offsets and the sampling period (unit days)

        """

        #--- Get control parameters
        self.verbose = control.get('Verbose',True)

        #--- Scale factor
        self.scale_factor = float(control.get('ScaleFactor',1.0))

        #--- class variables
        self.data = pd.DataFrame()
        self.offsets = []
        self.postseismicexp = []
        self.postseismiclog = []
        self.ssetanh = []
        self.sampling_period = 0.0
        self.F = None
        self.percentage_gaps = None
        self.m = 0
        self.column_name=''

        #--- Read filename with observations and the directory
        self.datafile = control.get('DataFile',None)
        directory = Path(control.get('DataDirectory',''))

        if self.datafile is None:
            fname = None
        else:
            fname = str(directory / self.datafile)

        #--- Which format?
        self.ts_format = control.get('TS_format','mom')

        if self.ts_format == 'mom':
            if not fname is None:
                self.momread(fname)
        elif self.ts_format == 'msf':
            #--- Are there columns with estimated trajectory models
            self.use_residuals = control.get('UseResiduals',False)
            #--- Which column
            if not 'ObservationColumn' in control.keys():
                raise ValueError('ObservationColumn is mandatory for msf files')

            self.observation_column = control['ObservationColumn']

            if not fname is None:
                self.msfread(fname)
        else:
            raise ValueError('Unknown format: {0:s}'.format(self.ts_format))

        #--- Inform the user
        if self.verbose==True:
            if not fname is None:
                print("\nFilename                   : {0:s}".format(fname))
            print("TS_format                  : {0:s}".format(self.ts_format))
            print("ScaleFactor                : {0:f}".\
                                                    format(self.scale_factor))
            if self.ts_format == 'msf':
                print("Observation Column         : {0:d}".\
                                             format(self.observation_column))
                print("Use Residuals              : {0:}".\
                                                    format(self.use_residuals))
            if not fname is None:
                print("Number of observations+gaps: {0:d}".format(self.m))
                print("Percentage of gaps         : {0:5.1f}".\
					                           format(self.percentage_gaps))



    def create_dataframe_and_F(self,t,obs,mod,period):
        """ Convert np.arrays into Panda DataFrame and create matrix F

        Args:
            t (np.array): array with MJD or sod
            obs (np.array): array with observations
            mod (np.array): array with modelled values
            period (floatt): sampling period (unit is days or seconds)
        """

        #--- Store sampling period in this class
        self.sampling_period = period

        #---- Create pandas DataFrame
        self.data = pd.DataFrame({'obs':np.asarray(obs)}, \
                                              index=np.asarray(t))
        if len(mod)>0:
            self.data['mod']=np.asarray(mod)

        #--- Create special missing data matrix F
        self.m = len(self.data.index)
        n = self.data['obs'].isna().sum()
        self.F = np.zeros((self.m,n))
        j=0
        for i in range(0,self.m):
            if np.isnan(self.data.iloc[i,0])==True:
                self.F[i,j]=1.0
                j += 1

        #--- Compute percentage of gaps
        self.percentage_gaps = 100.0 * float(n) /float(self.m)



    def momread(self,fname):
        """Read mom-file fname and store the data into the mom class

        Args:
            fname (string) : name of file that will be read
        """
        #--- Constants
        TINY = 1.0e-6

        #--- Check if file exists
        if os.path.isfile(fname)==False:
            print('File {0:s} does not exist'.format(fname))
            sys.exit()

        #--- Read the file (header + time series)
        t = []
        obs = []
        mod = []
        mjd_old = 0.0
        with open(fname,'r') as fp:
            for line in fp:
                cols = line.split()
                if line.startswith('#')==True:
                    if len(cols)>3:
                        if cols[1]=='sampling' and cols[2]=='period':
                            self.sampling_period = float(cols[3])
                    if len(cols)>2:
                        if cols[0]=='#' and cols[1]=='offset':
                            self.offsets.append(float(cols[2]))
                        elif cols[0]=='#' and cols[1]=='exp':
                            mjd = float(cols[2])
                            T   = float(cols[3])
                            self.postseismicexp.append([mjd,T])
                        elif cols[0]=='#' and cols[1]=='log':
                            mjd = float(cols[2])
                            T   = float(cols[3])
                            self.postseismiclog.append([mjd,T])
                        elif cols[0]=='#' and cols[1]=='tanh':
                            mjd = float(cols[2])
                            T   = float(cols[3])
                            self.ssetanh.append([mjd,T])
                else:
                    if len(cols)<2 or len(cols)>3:
                        print('Found illegal row: {0:s}'.format(line))
                        sys.exit()

                    mjd = float(cols[0])
                    #--- Fill gaps with NaN's
                    if mjd_old>0.0:
                        while abs(mjd-mjd_old-self.sampling_period)>TINY:
                            mjd_old += self.sampling_period
                            t.append(mjd_old)
                            obs.append(np.NAN)
                            if len(cols)==3:
                                mod.append(float(np.NAN))
                            if mjd_old>mjd-TINY:
                                print('Someting is very wrong here....')
                                print('mjd={0:f}'.format(mjd))
                                sys.exit()
                    t.append(mjd)
                    mjd_old = mjd
                    obs.append(self.scale_factor * float(cols[1]))
                    if len(cols)==3:
                        mod.append(self.scale_factor * float(cols[2]))

        self.create_dataframe_and_F(t,obs,mod,self.sampling_period)



    def msfread(self,fname):
        """Read msf-file fname and store the data into the Observation class

        Args:
            fname (string) : name of file that will be read
        """
        #--- Constants
        TINY = 1.0e-6

        #--- Check if file exists
        if os.path.isfile(fname)==False:
            print('File {0:s} does not exist'.format(fname))
            sys.exit()

        #--- Read the file (header + time series)
        t = []
        obs = []
        mjd0 = 0.0
        tt_old = 0.0
        inside_observations = False
        inside_models = False
        inside_offsets = False
        self.model_column = 0
        self.fname_in = fname
        offset_lines = []
        with open(fname,'r') as fp:
            for line in fp:
                cols = line.split()
                if line.startswith('#')==True:
                    if len(cols)>2:
                        if cols[1]=='sampling' and cols[2]=='period':
                            self.sampling_period = float(cols[3])
                    if len(cols)>1:
                        if cols[1]=='Observations':
                            inside_observations = True
                        elif cols[1]=='Models':
                            inside_observations = False
                            inside_models = True
                        elif cols[1]=='Offsets':
                            inside_models = False
                            inside_offset = True

                        #-- Does the line contain column definition?
                        m = re.match('#\s+(\d+)\.\s+(.+)',line)
                        if m:
                            column = int(m.group(1))
                            if inside_observations==True and \
                                              column==self.observation_column:
                                self.column_name = m.group(2)
                            elif inside_models==True and \
                                              m.group(2)==self.column_name:
                                self.model_column = column
                            elif inside_offsets==True and \
                                              column==self.observation_column:
                                offset_lines.append(m.group(2))

                else:
                    #--- sanity check
                    if self.use_residuals==True:
                        if self.model_column==0:
                            print('Could not find requested model column')
                            sys.exit()
                    if len(cols)<3:
                        print('Found illegal row: {0:s}'.format(line))
                        sys.exit()

                    mjd = float(cols[0])
                    if mjd0==0.0:
                        mjd0 = mjd
                    sod = float(cols[1])
                    tt = 86400.0*(mjd-mjd0) + sod
                    #--- Fill gaps with NaN's
                    if tt_old>0.0:
                        while abs(tt-tt_old-self.sampling_period)>TINY:
                            tt_old += self.sampling_period
                            t.append(tt_old)
                            obs.append(np.NAN)
                            if tt_old>tt-TINY:
                                print('Someting is very wrong here....')
                                print('mjd={0:f}, sod={1:f}'.format(mjd,sod))
                                sys.exit()
                    t.append(tt)
                    tt_old = tt
                    if cols[self.observation_column-1]=='NaN':
                        obs.append(np.NAN)
                    else:
                        if self.use_residuals==True:
                            y = float(cols[self.observation_column-1])-\
                                       float(cols[self.model_column-1])
                        else:
                            y = float(cols[self.observation_column-1])
                        obs.append(self.scale_factor * y)

        #--- Now that we know MJD0, store offset times
        for line in offset_lines:
            cols = line.split()
            tt = 86400.0*(float(cols[0]) - mjd0) + float(cols[1])
            self.offsets.append(tt)

        #---- Create pandas DataFrame
        self.create_dataframe_and_F(t,obs,[],self.sampling_period)



    def momwrite(self,fname):
        """Write the momdata to a file called fname

        Args:
            fname (string) : name of file that will be written
        """
        #--- Try to open the file for writing
        try:
            fp = open(fname,'w')
        except IOError:
           print('Error: File {0:s} cannot be opened for written.'. \
                                                         format(fname))
           sys.exit()
        if self.verbose==True:
            print('--> {0:s}'.format(fname))

        #--- Write header
        fp.write('# sampling period {0:f}\n'.format(self.sampling_period))

        #--- Write header offsets
        for i in range(0,len(self.offsets)):
            fp.write('# offset {0:10.4f}\n'.format(self.offsets[i]))
        #--- Write header exponential decay after seismic event
        for i in range(0,len(self.postseismicexp)):
            [mjd,T] = self.postseismicexp[i]
            fp.write('# exp {0:10.4f} {1:5.1f}\n'.format(mjd,T))
        #--- Write header logarithmic decay after seismic event
        for i in range(0,len(self.postseismiclog)):
            [mjd,T] = self.postseismiclog[i]
            fp.write('# exp {0:10.4f} {1:5.1f}\n'.format(mjd,T))
        #--- Write header slow slip event
        for i in range(0,len(self.ssetanh)):
            [mjd,T] = self.sshtanh[i]
            fp.write('# tanh {0:10.4f} {1:5.1f}\n'.format(mjd,T))

        #--- Write time series
        for i in range(0,len(self.data.index)):
            if not math.isnan(self.data.iloc[i,0])==True:
                fp.write('{0:12.6f} {1:13.6f}'.format(self.data.index[i],\
                                                  self.data.iloc[i,0]))
                if len(self.data.columns)==2:
                    fp.write(' {0:13.6f}\n'.format(self.data.iloc[i,1]))
                else:
                    fp.write('\n')

        fp.close()



    def msfwrite(self,fname):
        """Write the msf data to a file called fname

        Args:
            fname (string) : name of file that will be written

        self.observation_column, self.column_name are assumed to be known
        after msfread call. If not, then self.fname_in is also '' and a new
        header will be created, setting these two variables

        """

        #--- Constant
        TINY = 1.0e-8

        #--- Initial value, to be changed later
        mjd0 = 0.0

        #--- Create header from scratch
        if self.datafile=="None":
            self.column_name='unknown'
            self.observation_columnn = 3

            lines = []
            lines.append('# sampling period {0:f}\n#\n'.\
                                                format(self.sampling_period))
            lines.append('# Observations\n# ------------\n')
            lines.append('# 1. MJD - Modified Julian Date\n')
            lines.append('# 2. sod - seconds of day\n')
            lines.append('# 3. {0:s}\n#\n'.format(self.column_name))

            lines.append('# Models\n# ------------\n#\n')

            #--- Write offsets in header
            mjd0 = 58849.0
            lines.append('# Offsets\n# ------------\n')
            for i in range(0,len(self.offsets)):
                sod = self.offsets[i]
                mjd = mjd0
                while sod>86400.0:
                    mjd += 1
                    sod -= 86400.0
                lines.append('# 3. {0:6.0f} {1:10.4f}\n'.format(mjd,sod))

            lines.append('#\n#=====================================' + \
                                '==========================================\n')

            #--- Add MJD, sod & observation lines
            for i in range(0,self.m):
                mjd = mjd0
                sod = self.data.index[i]
                while sod>86400.0:
                    mjd += 1
                    sod -= 86400.0
                lines.append('{0:6.0f} {1:10.4f} {2:12.6f}\n'.\
                                          format(mjd,sod,self.data.iloc[i,0]))

        else:
            with open(self.fname_in,'r') as fp:
                lines = fp.readlines()

        #--- Try to open the file for writing
        try:
            fp = open(fname,'w')
        except IOError:
           print('Error: File {0:s} cannot be opened for written.'. \
                                                         format(fname))
           sys.exit()

        if self.verbose==True:
            print('--> {0:s}'.format(fname))

        #--- Case of adding anther column
        inside_observations = False
        inside_models = False
        inside_offsets = False
        i = 0

        for line in lines:

            #--- header stuff
            if line.startswith('#'):
                #--- If new trajectory model is estimated, add name in header
                if len(self.data.columns)==2:
                    if inside_observations==True:
                        #-- Does the line contain column definition?
                        m = re.match('# (\d+)\.\s+(.+)',line)
                        if m:
                            column = int(m.group(1))
                    elif inside_models==True:
                        #-- Does the line contain column definition?
                        m = re.match('# (\d+)\.\s+(.+)',line)
                        if m:
                            column = int(m.group(1))
                            if m.group(2)==self.column_name:
                                self.model_column = column
                        else:
                            if not line.startswith('# ------------') and \
                                                        self.model_column==0:
                                self.model_column = column+1
                                fp.write('# {0:d}. {1:s}\n'.format(column+1,
                                                            self.column_name))
                                inside_models=False

                    #--- Which block are we processing?
                    cols = line.split()
                    if len(cols)>1:
                        if cols[1]=='Observations':
                            inside_observations = True
                        if cols[1]=='Models':
                            inside_observations = False
                            inside_models = True
                        elif cols[1]=='Offsets':
                            inside_models = False

                #--- Just copy header line to new file
                fp.write(line)
            else:
                #--- Get time to see if this is equal to data.index[i]
                cols = line.split()
                mjd = float(cols[0])
                sod = float(cols[1])
                if mjd0==0.0:
                    mjd0 = mjd

                tt = 86400.0*(mjd - mjd0) + sod
                while self.data.index[i]<tt:
                    i += 1
                if abs(tt - self.data.index[i])>TINY:
                    print('Expected regularly spaced data!')
                    print('file time = {0:f}, array time = {1:f}'.\
                                            format(tt,self.data.index[i]))
                    sys.exit()

                #--- Replace observation values
                cols = line.split()
                fp.write('{0:>6s} {1:>10s}'.format(cols[0],cols[1]))
                for j in range(2,self.observation_column-1):
                    fp.write('{0:>12s} '.format(cols[j]))
                if math.isnan(self.data.iloc[i,0])==True:
                    fp.write('{0:>12s}'.format('NaN'))
                else:
                    fp.write('{0:12.6f}'.format(self.data.iloc[i,0]))

                if len(self.data.columns)==2:
                    #--- Replace Model values
                    for j in range(self.observation_column,\
                                                         self.model_column-1):
                        fp.write(' {0:>12s}'.format(cols[j]))
                    if math.isnan(self.data.iloc[i,0])==True:
                        fp.write('{0:>12s}'.format('NaN'))
                    else:
                        fp.write('{0:12.6f}'.format(self.data.iloc[i,1]))
                    for j in range(self.model_column,len(cols)):
                        fp.write(' {0:>12s}'.format(cols[j]))
                else:
                    for j in range(self.observation_column,len(cols)):
                        fp.write(' {0:>12s}'.format(cols[j]))
                fp.write('\n')

                #--- Go to next item in array
                i += 1

        fp.close()



    def show_results(self,output):
        """ add info to json-ouput dict
        """

        output['N'] = self.m
        output['gap_percentage'] = self.percentage_gaps
        if self.ts_format=='mom':
            output['TimeUnit'] = 'days'
        else:
            output['TimeUnit'] = 'seconds'



    def add_offset(self,t):
        """ Add time t to list of offsets

        Args:
            t (float): modified julian date or second of day of offset
        """

        EPS   = 1.0e-6
        found = False
        i     = 0
        while i<len(self.offsets) and found==False:
            if abs(self.offsets[i]-t)<EPS:
                found = True
            i += 1
        if found==False:
            self.offsets.append(t)



    def set_NaN(self,index):
        """ Set observation at index to NaN and update matrix F

        Args:
            index (int): index of array which needs to be set to NaN
        """

        self.data.iloc[index,0] = np.nan
        dummy = np.zeros(self.m)
        dummy[index] = 1.0
        self.F = np.c_[ self.F, dummy ] # add another column to F



    def add_mod(self,xhat):
        """ Add estimated model as column in DataFrame

        Args:
            xhat (array float) : estimated model
        """

        self.data['mod']=np.asarray(xhat)



    def write(self,fname):
        """ Select correct subroutine for writing to file

        Args:
            fname (string): complete name of file
        """

        if self.ts_format=='mom':
            self.momwrite(fname)
        elif self.ts_format=='msf':
            self.msfwrite(fname)
        else:
            print('unknown ts_format: {0:s}'.format(self.ts_format))
            sys.exit()
