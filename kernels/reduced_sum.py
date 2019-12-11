# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:14:26 2019

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classse.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

class ReducedSumKernel(Kernel):
    """
    Kernel to compute the sum of the input tensor's 
    element along the provided axes
    """
    
    def __init__(self,graph,inA,outA,axes=None,keep_dims=False):
        super().__init__('ReducedSum',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axes = axes
        self._keep_dims = keep_dims
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the inputs and outputs are the appropriate type
        if not isinstance(self._inA,Tensor):
            return BcipEnums.INVALID_PARAMETERS
        
        if not (isinstance(self._outA,Tensor) or isinstance(self._outA,Scalar)):
            return BcipEnums.INVALID_PARAMETERS

        if isinstance(self._outA,Scalar) and \
           not (self._outA.data_type in Scalar.valid_numeric_types()):
            return BcipEnums.INVALID_PARAMETERS
        
        inA_shape = self._inA.shape
        axes = self._axes if self._axes != None else ()
        
        if self._keep_dims:
            # all reduced dimensions will be '1'
            out_shape = tuple([1 if i in axes else inA_shape[i] 
                                          for i in range(len(inA_shape))])
        else:
            out_shape = tuple([inA_shape[i] for i in range(len(inA_shape))
                                                   if i not in axes])
        
        # if the output is a virtual tensor and has no defined shape, set the shape now
        if isinstance(self._outA,Tensor) and self._outA.virtual \
           and len(self._outA.shape) == 0:
            self._outA.shape = out_shape
        
        # ensure the output shape equals the expected output shape
        if isinstance(self._outA,Tensor) and self._outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif isinstance(self._outA,Scalar) and out_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        try:
            self._outA.data = np.sum(self._inA.data, 
                                     axes=self._axes,
                                     keep_dims=self._keep_dims)

        except ValueError:
            return BcipEnums.EXE_ERROR
            
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_reduced_sum_node(cls,graph,inA,outA,axes=None,keep_dims=False):
        """
        Factory method to create a reduced sum kernel 
        and add it to a graph as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axes,keep_dims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node