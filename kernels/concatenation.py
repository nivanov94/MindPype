# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:52:43 2019

concate.py  - Defines a concatenation kernel that concatenates multiple tensors
into a single tensor

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classse.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

class ConcatenationKernel(Kernel):
    """
    Kernel to concatenate multiple tensors into a single tensor
    """
    
    def __init__(self,graph,outA,inA,inB,axis):
        super().__init__('Concatenation',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
        self._outA = outA
        self._axis = axis
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        if self._axis != None:
            # check that the axis is a scalar object
            if not isinstance(self._axis,Scalar):
                return BcipEnums.INVALID_PARAMETERS
            
            # get the scalar type
            if self._axis.data_type != 'int':
                return BcipEnums.INVALID_PARAMETERS
            
            # the scalar object should be non-volatile so it doesn't change 
            # during runtime
            if self._axis.volatile:
                return BcipEnums.INVALID_PARAMETERS
            
            concat_axis = self._axis.data
        else:
            concat_axis = 0 # default axis
        
        
        # inA and inB must be a tensor
        if not (isinstance(self._inA,Tensor) and isinstance(self._inB,Tensor) \
                and isinstance(self._outA,Tensor)):
            return BcipEnums.INVALID_PARAMETERS
        
            
        # the dimensions along the catcat axis must be equal
        sz_A = self._inA.shape
        sz_B = self._inB.shape
        
        noncat_sz_A = [d for i,d in enumerate(sz_A) if i!=concat_axis]
        noncat_sz_B = [d for i,d in enumerate(sz_B) if i!=concat_axis]
        
        # check if the remaining dimensions are the same
        if (len(noncat_sz_A) != len(noncat_sz_B) or \
            len(noncat_sz_A) != sum([1 for i,j in 
                                     zip(noncat_sz_A,noncat_sz_B) if i==j])):
            return BcipEnums.INVALID_PARAMETERS
        
        
        output_sz = noncat_sz_A[:]
        output_sz.insert(concat_axis,sz_A[concat_axis]+sz_B[concat_axis])
        
        # check the output dimensions are valid
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = output_sz
        
        # ensure the output shape equals the expected output shape
        if self._outA.shape != output_sz:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy functions
        """
        concat_axis = self._axis.data if self._axis != None else 0
        
        try:
            out_tensor = np.concatenate((self._inA.data,
                                         self._inB.data),
                                        axis=concat_axis)
        except ValueError:
            # dimensions are invalid
            return BcipEnums.EXE_FAILURE
        
        # set the data in the output tensor
        self._outA.data = out_tensor
        
        return BcipEnums.SUCCESS
    
    
    @classmethod
    def add_concatenation_node(cls,graph,outA,inA,inB,axis=None):
        """
        Factory method to create a concatenation kernel and add it to a graph
        as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,outA,inA,inB,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(outA,BcipEnums.OUTPUT),
                  Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT))
        
    
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node