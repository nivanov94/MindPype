# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:20:03 2019

Covariance.py - Define the Covariance kernel for BCIP

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.bcip_enums import BcipEnums

import numpy as np

class CovarianceKernel(Kernel):
    """
    Kernel to compute the covariance of tensors. If the input tensor is 
    unidimensional, will compute the variance. For higher rank tensors,
    highest order dimension will be treated as variables and the second
    highest order dimension will be treated as observations. 
    
    Tensor size examples:
        Input:  A (kxmxn)
        Output: B (kxnxn)
        
        Input:  A (m)
        Output: B (1)
        
        Input:  A (mxn)
        Output: B (nxn)
        
        Input:  A (hxkxmxn)
        Output: B (hxkxnxn)
    """
    
    def __init__(self,graph,inputA,outputA):
        super().__init__('Covariance',BcipEnums.INIT_FROM_NONE,graph)
        self.inputA  = inputA
        self.outputA = outputA
    
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
            (not isinstance(self.outputA,Tensor)):
                return BcipEnums.INVALID_PARAMETERS
        
        # check the shape
        input_shape = self.inputA.shape
        input_rank = len(input_shape)
        
        # determine what the output shape should be
        if input_rank == 0:
            return BcipEnums.INVALID_PARAMETERS
        elif input_rank == 1:
            output_shape = (1,)
        else:
            output_shape = input_shape[:-2] + (input_shape[-1],input_shape[-1])
        
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
        Execute the kernel function using the numpy cov function
        """
        
        shape = self.inputA.shape
        rank = len(shape)
        
        input_data = self.inputA.data
        
        
        if rank <= 2:
            self.outputA.data = np.cov(input_data,rowvar=False)
        else:
            # reshape the input data so it's rank 3
            input_data = np.reshape(input_data,(-1,) + shape[-2:])
            output_data = np.zeros((input_data.shape[0],input_data.shape[2], \
                                    input_data[2]))
            
            # calculate the covariance for each 'trial'
            for i in range(output_data.shape[0]):
                output_data[i,:,:] = np.cov(input_data,rowvar=False)
            
            # reshape the output
            self.outputA.data = np.reshape(output_data,self.outputA.shape)
            
        return BcipEnums.SUCCESS
    
    @classmethod
    def addCovarianceNode(cls,graph,inputA,outputA):
        """
        Factory method to create a covariance kernel and add it to a graph
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