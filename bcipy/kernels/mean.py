"""
Created on Fri Mar  6 10:55:07 2020

@author: ivanovn
"""

"""
Created on Mon Dec  9 16:15:12 2019

@author: ivanovn
"""

from ..classes.kernel import Kernel
from ..classes.node import Node
from ..classes.parameter import Parameter
from ..classes.tensor import Tensor
from ..classes.bcip_enums import BcipEnums

import numpy as np

class MeanKernel(Kernel):
    """
    Calculates the mean of values in a tensor
    """
    
    def __init__(self,graph,inA,outA,axis):
        """
        Kernal calculates arithmetic mean of values in tensor or array
        """
        super().__init__('Mean',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axis = axis

        self._init_inA = None
        self._init_outA = None
        
    
    def initialize(self):
        """
        No internal state to setup
        """
        sts = self.initialization_execution()
        return sts
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self._inA,Tensor)) or \
            (not isinstance(self._outA,Tensor)):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self._inA.shape
        
        if self._axis == None:
            output_shape = (1,1)
        else:
            output_shape = tuple([x for i,x in enumerate(input_shape) if i != self._axis])

        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = output_shape
        
        # check output shape
        if self._outA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS

    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        try:
            output_data.data = np.mean(input_data.data,axis=self._axis)
        except:
            return BcipEnums.EXE_FAILURE
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        
        return self.process_data(self._inA, self._outA)
    
    @classmethod
    def add_mean_node(cls,graph,inA,outA,axis=None):
        """
        Factory method to create a Riemann mean calculating kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
