# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 18:20:03 2019

Transpose.py - Define the transpose kernel for BCIP

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.bcip_enums import BcipEnums

import numpy as np

class TransposeKernel(Kernel):
    """
    Kernel to compute the tensor transpose
    """
    
    def __init__(self,graph,inputA,outputA,axes=None):
        super().__init__('Transpose',BcipEnums.INIT_FROM_NONE,graph)
        self.inputA  = inputA
        self.outputA = outputA
        self.axes = axes
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self.inputA,Tensor)) or \
            (not isinstance(self.outputA,Tensor)) or \
            (self.axes != None and ((not isinstance(self.axes,tuple)) or \
            (len(self.axes) != len(self.inputA.shape)))):
                return BcipEnums.INVALID_PARAMETERS
        
        # check the shape
        input_shape = self.inputA.shape
        input_rank = len(input_shape)
        
        # determine what the output shape should be
        if input_rank == 0:
            return BcipEnums.INVALID_PARAMETERS
        
        if self.axes == None:
            output_shape = reversed(input_shape)
        else:
            output_shape = input_shape[self.axes]
               
        # if the output is virtual and has no defined shape, set the shape now
        if self.outputA.isVirtual() and len(self.outputA.shape) == 0:
            self.outputA.setShape(output_shape)
        
        # ensure the output tensor's shape equals the expected output shape
        if self.outputA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using the numpy transpose function
        """
        
        self.outputA.data = np.transpose(self.inputA.data,axes=self.axes)
        return BcipEnums.SUCCESS


    @classmethod
    def addTransposeNode(cls,graph,inputA,outputA):
        """
        Factory method to create a transpose kernel and add it to a graph
        as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inputA,outputA)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.addNode(node)
        
        return node