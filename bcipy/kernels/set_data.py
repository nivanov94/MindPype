# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:55:13 2019

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


class SetKernel(Kernel):
    """
    Kernel to set a portion of a tensor or array
    """
    
    def __init__(self,graph,container,data,axis,index,out):
        super().__init__('Extract',BcipEnums.INIT_FROM_NONE,graph)
        self._container = container
        self._out = out
        self._inA = data
        self._axis = axis
        self._index  = index

        self._init_in_inA = None
        self._init_container = None

        self._labels = None

    
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
        if (not (isinstance(self._container,Tensor) and \
                 isinstance(self._inA,Tensor) and \
                 isinstance(self._out,Tensor)))\
        and (not (isinstance(self._container,Array) \
                  and isinstance(self._inA,BCIP) \
                  and isinstance(self._out,Array))):
            return BcipEnums.INVALID_PARAMETERS
        

        # if the input is an array, then the axis must be zero
        if isinstance(self._container,Array):
            if self._axes != 0:
                return BcipEnums.INVALID_PARAMETERS
        
            # check the the size of the output array is large enough
            if self._out.virtual and self._out.capacity == 0:
                # set the capacity of the array
                self._out.capacity = self._container.capacity
            
            if self._container.capacity != len(self._container.capacity):
                return BcipEnums.INVALID_PARAMETERS
            
            # check that all the index to set does not exceed the capacity
            if self._index >= self._out.capacity:
                return BcipEnums.INVALID_PARAMETERS
                
        
        elif isinstance(self._container,Tensor):
            
            # check that the output tensor's dimensions are valid
            output_shape = self._container.shape
            
            if self._out.virtual and len(self._out.shape) == 0:
                self._out.shape = output_shape
            
            if self._out.shape != output_shape:
                return BcipEnums.INVALID_PARAMETERS
            
            # check that the axis specified does not exceed the tensor's rank
            if self._axis >= len(self._out.shape):
                return BcipEnums.INVALID_PARAMETERS
            
            # check that the index is a valid location in the container tensor
            if self._index >= self._container.shape[self._axis]:
                return BcipEnums.INVALID_PARAMETERS
        
            
            # check if the dimensions of the data to set match the shape 
            # fit the output shape
            ix_grid = []
            for i in len(self._out.shape):
                if i == self._axis:
                    ix_grid.append(list(self._index))
                else:
                    ix_grid.append([_ for _ in range(self._out.shape[i])])
            
            ixgrid = np.ix_(ix_grid)
            set_shape = self._out.data[ixgrid].shape
            if set_shape != self._inA.shape:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        if isinstance(self._inA, Array):
            # copy all the elements of the input container except the the 
            # data to set
            for i in range(self._out.capacity):
                if i == self._index:
                    self._out.set_element(i,self._inA)
                else:
                    self._out.set_element(i,self._container.get_element(i))
        else:
            # tensor case
            ix_grid = []
            for i in len(self._out.shape):
                if i == self._axis:
                    ix_grid.append(list(self._index))
                else:
                    ix_grid.append([_ for _ in range(self._out.shape[i])])
            
            ixgrid = np.ix_(ix_grid)
            out_data = self._container.data
            out_data[ixgrid] = self._inA
            self._out.data = out_data
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_set_node(cls,graph,container,data,axis,index,out):
        """
        Factory method to create a set kernel 
        and add it to a graph as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,container,data,axis,index,out)
        
        # create parameter objects for the input and output
        params = (Parameter(container,BcipEnums.INPUT),
                  Parameter(data,BcipEnums.INPUT),
                  Parameter(index,BcipEnums.INPUT),
                  Parameter(out,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node