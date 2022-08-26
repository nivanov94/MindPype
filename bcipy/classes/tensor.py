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

    Tensor (or n-dimensional matrices), are defined by the tensor class. 
    BCIP tensors can either be volatile (are updated/change each trial, generally reserved for tensors containing current trial data), virtual (empty, dimensionless tensor object). Like scalars and array, tensors can be created from data, copied from a different variable, or created virtually, so they don’t initially contain a value. 
    Each of the scalars, tensors and array data containers also have an external source (_ext_src) attribute, which indicates, if necessary, the source from which the data is being pulled from. This is especially important if trial/training data is loaded into a tensor each trial from an LSL stream or MAT file.

    Parameters
    ----------
    sess : Session object
        - Session where Tensor will exist
    
    shape : shape_like
        - Shape of the Tensor
    
    data : ndarray
        - Data to be stored within the array
    
    is_virtual : bool
        - If False, the Tensor is non-virtual

    ext_src : BCIPy input Source
        - Data source the tensor pulls data from (only applies to Tensors created from a handle)
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
        """
        Set data of a Tensor. If the current shape of the Tensor is different from the shape of the data being inputted, you must first change 
        the shape of the Tensor before adding the data, or an error will be thrown

        Parameters
        ----------
        data : nd_array
            - Data to have the Tensor data changed to
        """

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
        """
        Method to set the shape of a Tensor. Only applies to non-virtual tensors and sets all values in the modified tensor to 0.
        """

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

        Parameters 
        ----------
        dest_tensor : Tensor object
            - Tensor object where the attributes with the referenced Tensor will copied to

        """
        if dest_tensor.virtual:
            dest_tensor.shape = self.shape
        dest_tensor.data = self.data
        
        return BcipEnums.SUCCESS
        
        # Not copying virtual and ext_src attributes because these should 
        # only be set during creation and modifying could cause unintended
        # consequences
    
    def poll_volatile_data(self,label=None):
        """
        Pull data from external sources or BCIPy input data sources.
        """
        
        # check if the data is actually volatile, if not just return
        if not self.volatile:
            return BcipEnums.SUCCESS
        
        data = self.ext_src.poll_data(label)
        try:
            # if we only pulled one trial, remove the first dimension
            data = np.squeeze(data)
        except ValueError:
            pass # just ignore the error for now
        
        # set the data 
        self.data = data
        
        return BcipEnums.SUCCESS
    
    # Factory Methods
    @classmethod
    def create(cls,sess,shape):
        """
        Factory Method to create a generic, non-virtual, Tensor object. The shape must be known to create this object 
        
        sess : Session object
            - Session where Tensor will exist
        
        shape : shape_like
            - Shape of the Tensor
        
        """

        t = cls(sess,shape,None,False,None)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
    
    @classmethod
    def create_virtual(cls,sess,shape=()):
        """
        Factory method to create a virtual Tensor

        Parameters
        ----------

        sess : Session object
            - Session where Tensor will exist
        
        shape : shape_like, default = ()
            - Shape of the Tensor, can be changed for virtual tensors
        """

        t = cls(sess,shape,None,True,None)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
    
    @classmethod
    def create_from_data(cls,sess,shape,data):
        """
        Factory method to create a Tensor from data

        Parameters
        ----------

        sess : Session object
            - Session where Tensor will exist
        
        shape : shape_like
            - Shape of the Tensor
        
        data : ndarray
            - Data to be stored within the array
            
        """

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
        """
        Factory method to create a Tensor from a handle/external source

        Parameters
        ----------

        sess : Session object
            - Session where Tensor will exist
        
        shape : shape_like
            - Shape of the Tensor
        
        ext_src : BCIPy input Source
            - Data source the tensor pulls data from (only applies to Tensors created from a handle)
        
        """
        t = cls(sess,shape,None,False,src)
        
        # add the tensor to the session
        sess.add_data(t)
        return t
    
    
    # utility static methods
    @staticmethod
    def validate_data(shape,data):
        """
        Method that returns True if  the data within the tensor is the right shape and is a numpy ndarray. False otherwise.
        """
        if data is None:
            return False
        
        if (not (type(data) is np.ndarray)) or (tuple(shape) != data.shape):
            return False
            
        return True
        
    
    