"""
Created on Mon Dec  9 17:33:21 2019

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classse.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

class MultiplicationKernel(Kernel):
    """
    Kernel to multiply two BCIPP data containers (i.e. tensor or scalar)
    together
    
    Note: This is element-wise multiplication
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('Multiplication',BcipEnums.INIT_FROM_NONE,graph)
        self.inA  = inA
        self.inB  = inB
        self.outA = outA
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the inputs and outputs are the appropriate type
        if not (isinstance(self.inA,Tensor) or isinstance(self.inA,Scalar)):
            return BcipEnums.INVALID_PARAMETERS
        
        if not (isinstance(self.inA,Tensor) or isinstance(self.inA,Scalar)):
            return BcipEnums.INVALID_PARAMETERS
        
        if (isinstance(self.inA,Tensor) or isinstance(self.inB,Tensor)) and \
           (not isinstance(self.outA,Tensor)):
            # if one of the inputs is a tensor, the output will be a tensor
            return BcipEnums.INVALID_PARAMETERS
        elif not isinstance(self.outA,Scalar):
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS
        
        # if the inputs are scalars, ensure they are numberic
        if isinstance(self.inA,Scalar) and \
           not self.inA.getType() in Scalar.validNumbericTypes():
            return BcipEnums.INVALID_PARAMETERS
        
        if isinstance(self.inB,Scalar) and \
           not self.inB.getType() in Scalar.validNumbericTypes():
            return BcipEnums.INVALID_PARAMETERS
        
        
        # check the shapes
        if isinstance(self.inA,Tensor):
            inA_shape = self.inA.shape
        else:
            inA_shape = (1,)
        
        if isinstance(self.inB,Tensor):
            inB_shape = self.inB.shape
        else:
            inB_shape = (1,)
        
        # determine what the output shape should be
        try:
            phony_a = np.zeros(inA_shape)
            phony_b = np.zeros(inB_shape)
            
            phony_out = phony_a * phony_b
        
        except ValueError:
            # these dimensions cannot be broadbast together
            return BcipEnums.INVALID_PARAMETERS
        
        out_shape = phony_out.shape
        
        # if the output is a virtual tensor and has no defined shape, set the shape now
        if isinstance(self.outA,Tensor) and self.outA.isVirtual() \
           and len(self.outA.shape) == 0:
            self.outA.setShape(out_shape)
        
        # ensure the output shape equals the expected output shape
        if isinstance(self.outA,Tensor) and self.outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif isinstance(self.outA,Scalar) and out_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        try:
            inA_data = self.inA.getData()
            inB_data = self.inB.getData()
            out_data = inA_data * inB_data
        
            self.outA.setData(out_data)

        except ValueError:
            return BcipEnums.EXE_ERROR
            
        return BcipEnums.SUCCESS
    
    @classmethod
    def addMultiplicationNode(cls,graph,inA,inB,outA):
        """
        Factory method to create a multiplication kernel and add it to a graph
        as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(inB,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.addNode(node)
        
        return node