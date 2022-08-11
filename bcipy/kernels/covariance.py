"""
Created on Wed Nov 20 16:20:03 2019

Covariance.py - Define the Covariance kernel for BCIP

@author: ivanovn
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.bcip_enums import BcipEnums

import numpy as np

# for debugging
import matplotlib
import matplotlib.pyplot as plt

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
    
    def __init__(self,graph,inputA,outputA,regularization):
        super().__init__('Covariance',BcipEnums.INIT_FROM_NONE,graph)
        self._inputA  = inputA
        self._outputA = outputA
        self._r = regularization

        self._init_inA = None
        self._init_outA = None

        self._labels = None

    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        if self._init_outA.__class__ != NoneType:
            return self.initialization_execution()

        return BcipEnums.SUCCESS

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self._inputA,Tensor)) or \
            (not isinstance(self._outputA,Tensor)):
                return BcipEnums.INVALID_PARAMETERS
        
        if self._r > 1 or self._r < 0:
            return BcipEnums.INVALID_PARAMETERS
        
        # check the shape
        input_shape = self._inputA.shape
        input_rank = len(input_shape)
        
        # determine what the output shape should be
        if input_rank == 0:
            return BcipEnums.INVALID_PARAMETERS
        elif input_rank == 1:
            output_shape = (1,)
        else:
            output_shape = (input_shape[-2],input_shape[-2])
        
        # if the output is virtual and has no defined shape, set the shape now
        if self._outputA.virtual and len(self._outputA.shape) == 0:
            self._outputA.shape = output_shape
        
        print(input_shape, output_shape)
        # ensure the output tensor's shape equals the expected output shape
        if self._outputA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS
        
    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data1, output_data1):
        shape = input_data1.shape
        rank = len(shape)
        
        input_data = input_data1.data
        
        
        if rank <= 2:
            covmat = np.cov(input_data)
            output_data1.data = 1/(1+self._r) * \
                                    (covmat + self._r*np.eye(covmat.shape[0]))
        else:
            # reshape the input data so it's rank 3
            input_data = np.reshape(input_data,(-1,) + shape[-2:])
            output_data = np.zeros((input_data.shape[0],input_data.shape[1], \
                                    input_data[1]))
            
            # calculate the covariance for each 'trial'
            for i in range(output_data.shape[0]):
                covmat = np.cov(input_data)
                output_data[i,:,:] = 1/(1+self._r) * \
                                     (covmat + self._r*np.eye(covmat.shape[0]))
            
            # reshape the output
            output_data1.data = np.reshape(output_data,self._outputA.shape)
            
#        # for debugging
#        d = self._outputA.data
#        plt.matshow(d)
#        plt.colorbar()
#        plt.figure()
#        plt.show()
            
        return BcipEnums.SUCCESS
    
    def execute(self):
        """
        Execute the kernel function using the numpy cov function
        """
        
        return self.process_data(self._inputA, self._outputA)
    
    @classmethod
    def add_covariance_node(cls,graph,inputA,outputA,regularization=0):
        """
        Factory method to create a covariance kernel and add it to a graph
        as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inputA,outputA,regularization)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
