"""
Created on Mon Dec  9 16:15:12 2019

@author: ivanovn
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.bcip_enums import BcipEnums

import numpy as np

from pyriemann.utils.mean import mean_riemann

class RiemannMeanKernel(Kernel):
    """
    Calculates the Riemann mean of covariances contained in a tensor

    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor object
        - Input data

    outA : Tensor object
        - Output trial data

    axis : int
        - Axis over which the mean should be calculated (see np.mean for more info)

    weights : array_like
        - Weights for each sample
    """
    
    def __init__(self,graph,inA,outA,weights):
        """
        Kernel takes 3D Tensor input and produces 2D Tensor representing mean
        """
        super().__init__('RiemannMean',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA

        self._init_inA = None
        self._init_outA = None
        
        self._w = weights
    
        

        self._labels = None
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        if self._init_outA.__class__ != NoneType:
            return self.initialization_execution()
        
        return BcipEnums.SUCCESS
        
    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        output_data.data = mean_riemann(input_data.data,sample_weight=self._w)
        
        return BcipEnums.SUCCESS

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if ((not isinstance(self._inA,Tensor)) or 
            (not isinstance(self._outA,Tensor))):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self._inA.shape
        input_rank = len(input_shape)
        
        # input tensor must be rank 3
        if input_rank != 3:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = input_shape[1:]
        
        
        # output tensor should be one dimensional
        if len(self._outA.shape) > 2:
            return BcipEnums.INVALID_PARAMETERS
        
        # check that the dimensions of the output match the dimensions of
        # input
        if self._inA.shape[1:] != self._outA.shape:
            return BcipEnums.INVALID_PARAMETERS
        
        if self._w != None:
            if len(self._w) != self._inA.shape[0]:
                return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS

        
    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        
        # calculate the mean using pyRiemann
        return self.process_data(self._inA, self._outA)
    
    @classmethod
    def add_riemann_mean_node(cls,graph,inA,outA,weights=None):
        """
        Factory method to create a Riemann mean calculating kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,weights)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
