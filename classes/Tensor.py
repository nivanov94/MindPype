# -*- coding: utf-8 -*-
"""
Tensor.py - Defines the Tensor class for BCIP

@author: ivanovn
"""

from bcip import BCIP
from bcip_types import BcipEnums

import numpy as np

class Tensor(BCIP):
    """
    Tensor data
    """
    
    def __init__(self,shape,data,is_virtual,ext_src):
        super().__init__(BcipEnums.TENSOR)
        self.shape = shape
        self.is_virtual = is_virtual
        self.ext_src = ext_src
        
        if not (data is None):
            self.data = data
        else:
            self.data = np.zeros(shape)
        
        if ext_src is None:
            self.volatile = False
        else:
            self.volatile = True
            
    def getData(self):
        return self.data
    
    def setData(self,data):
        if Tensor.validateData(data.shape,self.data):
            self.data = data
    
    def pollVolatileData(self):
        
        # check if the data is actually volatile, if not just return
        if not self.is_voltatile:
            return True
        
        # TODO - READ DATA FROM SOURCE HERE
        
    
    # Factory Methods
    @classmethod
    def create(shape):
        return Tensor(shape,None,False,None)
    
    @classmethod
    def createVirtual(shape):
        return Tensor(shape,None,True,None)
    
    @classmethod
    def createFromData(shape,data):
        # make sure data is valid
        if not Tensor.validateData(shape,data):
            # data invalid!
            return 
        return Tensor(shape,data,False,None)
    
    @classmethod
    def createFromHandle(shape,src):
        return Tensor(shape,None,False,src)
    
    # utility static methods
    @staticmethod
    def validateData(shape,data):
        if not (data is None):
            return False
        
        if (not (type(data) is np.ndarray)) or (shape != data.shape):
            return False
            
        return True
        
    
    