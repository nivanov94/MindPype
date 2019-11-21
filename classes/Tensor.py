# -*- coding: utf-8 -*-
"""
Tensor.py - Defines the Tensor class for BCIP

@author: ivanovn
"""

from .bcip import BCIP
from .bcip_enums import BcipEnums

import numpy as np

class Tensor(BCIP):
    """
    Tensor data
    """
    
    def __init__(self,sess,shape,data,is_virtual,ext_src):
        super().__init__(BcipEnums.TENSOR)
        self.sess = sess
        self.shape = shape
        self.is_virtual = is_virtual
        self.ext_src = ext_src
        
        if not (data is None):
            self.data = data
        else:
            self.data = np.zeros(shape)
        
        if ext_src is None:
            self._volatile = False
        else:
            self._volatile = True
            
    def getData(self):
        return self.data
    
    def setData(self,data):
        if Tensor.validateData(data.shape,self.data):
            self.data = data
    
    def setShape(self,shape):
        zero_tensor = np.zeros(shape)
        self.shape = shape
        self.setData(zero_tensor)
    
    def isVirtual(self):
        return self.is_virtual
    
    def isVolatile(self):
        return self._volatile
    
    def pollVolatileData(self):
        
        # check if the data is actually volatile, if not just return
        if not self.is_voltatile:
            return True
        
        # TODO - READ DATA FROM SOURCE HERE
        
    
    # Factory Methods
    @classmethod
    def create(cls,sess,shape):
        t = cls(sess,shape,None,False,None)
        
        # add the tensor to the session
        sess.addData(t)
        return t
    
    @classmethod
    def createVirtual(cls,sess,shape=()):
        t = cls(sess,shape,None,True,None)
        
        # add the tensor to the session
        sess.addData(t)
        return t
    
    @classmethod
    def createFromData(cls,sess,shape,data):
        # make sure data is valid
        if not cls.validateData(shape,data):
            # data invalid!
            return 
        t = cls(sess,shape,data,False,None)
        
        # add the tensor to the session
        sess.addData(t)
        return t
        
    @classmethod
    def createFromHandle(cls,sess,shape,src):
        t = cls(sess,shape,None,False,src)
        
        # add the tensor to the session
        sess.addData(t)
        return t
    
    
    # utility static methods
    @staticmethod
    def validateData(shape,data):
        if data is None:
            return False
        
        if (not (type(data) is np.ndarray)) or (shape != data.shape):
            return False
            
        return True
        
    
    