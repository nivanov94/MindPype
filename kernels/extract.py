"""
Created on Thu Dec 12 12:12:25 2019

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.bcip_enums import BcipEnums
from classes.array import Array
from classes.tensor import Tensor
from classes.bcip import BCIP

import numpy as np


class ExtractKernel(Kernel):
    """
    Kernel to extract a portion of a tensor or array
    """
    
    def __init__(self,graph,inA,axis,index,outA):
        super().__init__('Extract',BcipEnums.INIT_FROM_NONE,graph)
        self._in   = inA
        self._out  = outA
        self._axis = axis
        self._index  = index
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # input must be a tensor or array
        # additionally, if the input is a tensor, the output should also be a
        # tensor
        if (not (isinstance(self._in,Tensor) and isinstance(self._out,Tensor)))\
        and (not (isinstance(self._in,Array) and isinstance(self._out,BCIP))):
            return BcipEnums.INVALID_PARAMETERS
        

        # if the input is an array, then the there should only be a single 
        # dimension to extract with a value of zero
        if isinstance(self._in,Array):
            if self._axis != 0:
                return BcipEnums.INVALID_PARAMETERS
            
            # check that the index to extract do not exceed the capacity
            if self._index >= self._in.capacity:
                return BcipEnums.INVALID_PARAMETERS
        
            # if the output is a tensor, check the shape
            if isinstance(self._out,Tensor):
                if not isinstance(self._in.get_element(self._index),Tensor):
                    return BcipEnums.INVALID_PARAMETERS
                
                if self._out.virtual and len(self._out.shape) == 0:
                    self._out.shape = self._in.get_element(self._index).shape
                
                if self._in.get_element(self._index) != self._out.shape:
                    return BcipEnums.INVALID_PARAMETERS
        
        elif isinstance(self._in,Tensor):
            # check that the number of dimensions indicated does not exceed 
            # the tensor's rank
            if self._axis >= len(self._in.shape):
                return BcipEnums.INVALID_PARAMETERS
            
            # check that the index is valid for the given axis
            if self._index < 0 or self._index >= self._in.shape[self._axis]:
                return BcipEnums.INVALID_PARAMETERS
        
            
            # check that the output tensor's dimensions are valid
            input_shape = self._in.shape
            output_shape = tuple([input_shape[i] 
                                            for i in range(len(input_shape)) 
                                            if i!=self._axis])
            
            if self._out.virtual and len(self._out.shape) == 0:
                self._out.shape = output_shape
            
            if self._out.shape != output_shape:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        if isinstance(self._in, Array):
            # extract the elements and set in the output array
            self._out = self._in.get_element(self._index).copy()
        else:
            # tensor case
            ix_grid = []
            for i in len(self._in.shape):
                if i == self._axis:
                    ix_grid.append(list(self._index))
                else:
                    ix_grid.append([_ for _ in range(self._in.shape[i])])
            
            ixgrid = np.ix_(ix_grid)
            self._out.data = self._in.data[ixgrid]
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_extract_node(cls,graph,inA,axis,index,outA):
        """
        Factory method to create an extract kernel 
        and add it to a graph as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,axis,index,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(index,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node