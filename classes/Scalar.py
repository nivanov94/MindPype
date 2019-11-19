# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:47:58 2019

Scalar.py - Define the Scalar class for BCIP

@author: ivanovn
"""

from bcip import BCIP
from bcip_type import BcipEnums

class Scalar(BCIP):
    
    _valid_types = [int, float, complex, str, bool]
    
    def __init__(self,scalar_type,val,is_virtual,ext_src):
        super().__init__(BcipEnums.SCALAR)
        self.scalar_type = scalar_type
        self.is_virtual = is_virtual
        self.ext_src = ext_src
        self.val = val
        
        if ext_src is None:
            self.volatile = False
        else:
            self.volatile = True
            
    def getValue(self):
        return self.val
    
    def setValue(self,val):
        if self._validateValue(val):
            self.val = val
    
    def pollVolatileData(self):
        
        # check if the data is actually volatile, if not just return
        if not self.is_voltatile:
            return True
        
        # TODO - READ DATA FROM SOURCE HERE
        
    def _validateValue(self,val):
        if type(val) is self.scalar_type:
            return True
        else:
            return False
    
    # Factory Methods
    @classmethod
    def create(scalar_type):
        if not (scalar_type in Scalar._valid_types):
            return
        return Scalar(scalar_type,None,False,None)
    
    @classmethod
    def createVirtual(scalar_type):
        if not (scalar_type in Scalar._valid_types):
            return
        return Scalar(scalar_type,None,True,None)
    
    @classmethod
    def createFromValue(value):
        scalar_type = type(value)
        if not (scalar_type in Scalar._valid_types):
            return
        
        return Scalar(scalar_type,value,False,None)
    
    @classmethod
    def createFromHandle(scalar_type,src):
        if not (scalar_type in Scalar._valid_types):
            return
        return Scalar(scalar_type,None,False,src)
    