"""
Created on Thu Feb 13 16:12:17 2020

Implements a cumulative riemann mean kernel

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.circle_buffer import CircleBuffer
from classes.bcip_enums import BcipEnums

import numpy as np

from pyriemann.utils.mean import mean_riemann

class CumulativeRiemannMeanKernel(Kernel):
    """
    Calculates the cumulative Riemann mean of covariance matrices
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('CumulativeRiemannMean',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
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
        
        # first ensure the inA is a circle buffer and the output is a tensor
        if (not isinstance(self._inA,CircleBuffer)) or \
            (not isinstance(self._outA,Tensor)):
                return BcipEnums.INVALID_PARAMETERS
        
        # if input B is not none, ensure it is a tensor
        if self._inB != None and not isinstance(self._inB,Tensor):
            return BcipEnums.INVALID_PARAMETERS
        
        # ensure that the elements of inA are tensors
        for i in range(self._inA.capacity):
            e = self._inA.get_element(i)
            if not isinstance(e,Tensor):
                return BcipEnums.INVALID_PARAMETERS
        
        # get the shape of the tensors within inA
        cov_dims = self._inA.get_element(0).shape
        
        # ensure that the tensors are rank 2 with equal rows and cols
        if len(cov_dims) != 2 or cov_dims[0] != cov_dims[1]:
            return BcipEnums.INVALID_PARAMETERS
        
        # if inB exists, check that the dimensions match
        if self._inB != None and self._inB.shape != cov_dims:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = cov_dims
        
        
        # output tensor should be two dimensional
        if len(self._outA.shape) != 2 or self._outA.shape != cov_dims:
            return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        
        try:
            cov_dims = self._inA.get_element(0).shape
            n_covs = self._inA.num_elements
            if self._inB != None:
                n_covs += 1
            
            d = np.zeros(((n_covs,) + cov_dims))
            # extract all the circle buffer's tensors into a single numpy array
            for i in range(self._inA.num_elements):
                d[i,:,:] = self._inA.get_queued_element(i).data
            
            # add the data from inB
            if self._inB != None:
                d[-1,:,:] = self._inB.data
            
            
            # calculate the mean using pyRiemann
            self._outA.data = mean_riemann(d)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_cumulative_riemann_mean_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a cumulative Riemann mean calculating kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
