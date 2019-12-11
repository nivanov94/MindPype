"""
Created on Mon Dec  9 16:15:12 2019

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.bcip_enums import BcipEnums

import numpy as np

from pyriemann.utils import mean_riemann

class RiemannMeanKernel(Kernel):
    """
    Calculates the Riemann mean of covariances contained in a tensor
    """
    
    def __init__(self,graph,inA,outA):
        """
        Kernel takes 3D Tensor input and produces 2D Tensor representing mean
        """
        super().__init__('RiemannMean',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        
    
    def initialize(self):
        """
        No internal state to setup
        """
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self._inA,Tensor)) or \
            (not isinstance(self._outA,Tensor)):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self._inA.shape
        input_rank = len(input_shape)
        
        # input tensor must be rank 3
        if input_rank != 3:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and self._outA.shape == None):
            self._outA.shape = input_shape[1:]
        
        
        # output tensor should be one dimensional
        if len(self._outputA.shape) > 2:
            return BcipEnums.INVALID_PARAMETERS
        
        # check that the dimensions of the output match the dimensions of
        # input
        if self._inA.shape[1:] != self._outputA.shape:
            return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        # calculate the mean using pyRiemann
        self._outputA.data = mean_riemann(self._inA.data)
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_riemann_mean_node(cls,graph,inA,outA):
        """
        Factory method to create a Riemann mean calculating kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
