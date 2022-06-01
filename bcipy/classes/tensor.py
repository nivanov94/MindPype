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
        super().__init__(BcipEnums.TENSOR,sess)
        self._shape = tuple(shape)
        self._virtual = is_virtual
        self._ext_src = ext_src
        
        if not (data is None):
            self.data = data
        else:
            self.data = np.zeros(shape)
        
        if ext_src is None:
            self._volatile = False
        else:
            self._volatile = True
    
    # API Getters
    @property
    def data(self):
        return self._data
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def virtual(self):
        return self._virtual
    
    @property
    def volatile(self):
        return self._volatile
    
    @property
    def ext_src(self):
        return self._ext_src
    
    #API setters
    @data.setter
    def data(self,data):
        # special case where every dimension is a singleton
        if (np.prod(np.asarray(data.shape)) == 1 and 
            np.prod(np.asarray(self.shape)) == 1):
            while len(self.shape) > len(data.shape):
                data = np.expand_dims(data,axis=0)
            
            while len(self.shape) < len(data.shape):
                data = np.squeeze(data,axis=0)
        
        if self.shape == data.shape:
            self._data = data
        else:
            raise ValueError("Mismatched shape")
    
    @shape.setter
    def shape(self,shape):
        if self.virtual:
            self._shape = shape
            # when changing the shape write a zero tensor to data
            self.data = np.zeros(shape)
        else:
            raise ValueError("Cannot change shape of non-virtual tensor")
            
            
    def make_copy(self):
        """
        Create and return a deep copy of the tensor
        """
        #TODO determine what to do when copying virtual
        cpy = Tensor(self.session,
                     self.shape,
                     self.data,
                     self.virtual,
                     self.ext_src)
        
        # add the copy to the session
        sess = self.session
        sess.add_data(cpy)
        
        return cpy
    
    def copy_to(self,dest_tensor):
        """
        Copy the attributes of the tensor to another tensor object
        """
        if dest_tensor.virtual:
            dest_tensor.shape = self.shape
        dest_tensor.data = self.data
        
        return BcipEnums.SUCCESS
        
        # Not copying virtual and ext_src attributes because these should 
        # only be set during creation and modifying could cause unintended
        # consequences
    
    def poll_volatile_data(self,label):
        
        # check if the data is actually volatile, if not just return
        if not self.volatile:
            return BcipEnums.SUCCESS
        
        data = self.ext_src.poll_data(label)
        try:
            # if we only pulled one trial, remove the first dimension
            data = np.squeeze(data,axis=0)
        except ValueError:
            pass # just ignore the error for now
        
        # set the data 
        self.data = data
        
        return BcipEnums.SUCCESS
    
    # Factory Methods
    @classmethod
    def create(cls,sess,shape):
        t = cls(sess,shape,None,False,None)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
    
    @classmethod
    def create_virtual(cls,sess,shape=()):
        t = cls(sess,shape,None,True,None)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
    
    @classmethod
    def create_from_data(cls,sess,shape,data):
        if type(data) is list:
            data = np.asarray(data)
        
        # make sure data is valid
        if not cls.validate_data(shape,data):
            # data invalid!
            return 
        t = cls(sess,shape,data,False,None)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
        
    @classmethod
    def create_from_handle(cls,sess,shape,src):
        t = cls(sess,shape,None,False,src)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
    
    
    # utility static methods
    @staticmethod
    def validate_data(shape,data):
        if data is None:
            return False
        
        if (not (type(data) is np.ndarray)) or (tuple(shape) != data.shape):
            return False
            
        return True
        
    
    