# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:47:58 2019

Scalar.py - Define the Scalar class for BCIP

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums

import numpy as np

class Scalar(BCIP):
    
    _valid_types = [int, float, complex, str, bool]
    
    def __init__(self,sess,value_type,val,is_virtual,ext_src):
        super().__init__(BcipEnums.SCALAR,sess)
        self._data_type = value_type

        self._ext_src = ext_src
        self.data = val
        
        self._virtual = is_virtual        
        if ext_src is None:
            self._volatile = False
        else:
            self._volatile = True
            
    # API Getters
    @property
    def volatile(self):
        return self._volatile
    
    @property
    def virtual(self):
        return self._virtual
    
    @property
    def data(self):
        return self._data
    
    @property
    def data_type(self):
        return self._data_type
    
    @property
    def ext_src(self):
        return self._ext_src
    
    
    # API Setters
    @data.setter
    def data(self,data):
        # if the data passed in is a numpy array, check if its a single value
        if type(data) == np.ndarray and data.shape == (1,):
            # convert from the np type to native python type
            data = data[0]
            if isinstance(data, np.integer):
                data = int(data)
            elif isinstance(data, np.float):
                data = float(data)
            elif isinstance(data,np.complex):
                data = complex(data)
            
        if type(data) == self.data_type:
            self._data = data
        else:
            raise ValueError(("BCIP Scalar contains data of type {}. Cannot" +\
                              " set data to type {}").format(self.data_type,
                                                             type(data))) 
    
    def make_copy(self):
        """
        Produce and return a deep copy of the scalar
        """
        cpy = Scalar(self.session,
                     self.data_type,
                     self.data,
                     self.virtual,
                     self.ext_src)
        
        return cpy
    
    def copy_to(self,dest_scalar):
        """
        Copy all the elements of the scalar to another scalar
        """
        dest_scalar.data = self.data
        
        # for now, don't copy the type, virtual and ext_src attributes because these
        # should really be set during creation not later
    
    
    def poll_volatile_data(self,label):
        
        # check if the data is actually volatile, if not just return
        if not self.is_voltatile:
            return BcipEnums.SUCCESS
        
        self.data = self.ext_src.pollData(label)
        
        return BcipEnums.SUCCESS
        
    
    @classmethod
    def valid_numeric_types(cls):
        return ['int','float','complex']
    
    # Factory Methods
    @classmethod
    def create(cls,sess,data_type):
        if not (data_type in Scalar._valid_types):
            return
        s = cls(sess,data_type,None,False,None)
        
        sess.add_data(s)
        return s
    
    @classmethod
    def create_virtual(cls,sess,data_type):
        if not (data_type in Scalar._valid_types):
            return
        s = cls(sess,data_type,None,True,None)
        
        # add the scalar to the session
        sess.add_data(s)
        return s
    
    @classmethod
    def create_from_value(cls,sess,value):
        data_type = type(value)
        if not (data_type in Scalar._valid_types):
            return
        
        s = cls(sess,data_type,value,False,None)
        
        # add the scalar to the session
        sess.add_data(s)
        return s
    
    @classmethod
    def create_from_handle(cls,sess,data_type,src):
        if not (data_type in Scalar._valid_types):
            return
        s = cls(sess,data_type,None,False,src)
        
        # add the scalar to the session
        sess.add_data(s)
        return s
    