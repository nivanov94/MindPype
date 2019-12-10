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
        self.inA  = inA
        self.inB  = inB
        self.outA = outA
        self.axis = axis
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        if self.axis != None:
            # check that the axis is a scalar object
            if not isinstance(self.axis,Scalar):
                return BcipEnums.INVALID_PARAMETERS
            
            # get the scalar type
            if self.axis.getType() != 'int':
                return BcipEnums.INVALID_PARAMETERS
            
            # the scalar object should be non-volatile so it doesn't change 
            # during runtime
            if self.axis.isVolatile():
                return BcipEnums.INVALID_PARAMETERS
            
            concat_axis = self.axis.getData()
        else:
            concat_axis = 0 # default axis
        
        
        # inA and inB must be a tensor
        if not (isinstance(self.inA,Tensor) and isinstance(self.inB,Tensor) \
                and isinstance(self.outA,Tensor)):
            return BcipEnums.INVALID_PARAMETERS
        
            
        # the dimensions along the catcat axis must be equal
        sz_A = self.inA.getData().shape
        sz_B = self.inB.getData().shape
        
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
        if self.outA.isVirtual() and len(self.outA.shape) == 0:
            self.outA.setShape(output_sz)
        
        # ensure the output shape equals the expected output shape
        if self.outA.shape != output_sz:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy functions
        """
        if self.axis == None:
            concat_axis = 0
        else:
            concat_axis = self.axis.getData()
        
        try:
            out_tensor = np.concatenate((self.inA.getData(),
                                         self.inB.getData()),
                                        axis=concat_axis)
        except ValueError:
            # dimensions are invalid
            return BcipEnums.EXE_FAILURE
        
        # set the data in the output tensor
        self.outA.setData(out_tensor)
        
        return BcipEnums.SUCCESS
    
    
    @classmethod
    def addConcatenationNode(cls,graph,outA,inA,inB,axis=None):
        """
        Factory method to create a concatenation kernel and add it to a graph
        as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,outA,inA,inB,axis)
        
        # create parameter objects for the input and output
        params = [Parameter(outA,BcipEnums.OUTPUT), \
                  Parameter(inA,BcipEnums.INPUT),   \
                  Parameter(inB,BcipEnums.INPUT)]
        
        
        if axis != None:
            params.append(Parameter(axis,BcipEnums.INPUT))
        
        params = tuple(params)
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.addNode(node)
        
        return node