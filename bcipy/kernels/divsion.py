"""
Created on Mon Dec  9 17:38:30 2019

@author: ivanovn
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

class DivisionKernel(Kernel):
    """
    Kernel to divide two BCIP data containers (i.e. tensor or scalar)
    together
    
    Note: This is element-wise division
    """
    
    def __init__(self,graph,inA,inB,outA):
        super().__init__('Division',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
        self._outA = outA

        self._init_inA = None
        self._init_inB = None
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
        
        # first ensure the inputs and outputs are the appropriate type
        if not (isinstance(self._inA,Tensor) or isinstance(self._inA,Scalar)):
            return BcipEnums.INVALID_PARAMETERS
        
        if not (isinstance(self._inB,Tensor) or isinstance(self._inB,Scalar)):
            return BcipEnums.INVALID_PARAMETERS
        
        if isinstance(self._inA,Tensor) or isinstance(self._inB,Tensor):
            if not isinstance(self._outA,Tensor):
                # if one of the inputs is a tensor, the output will be a tensor
                return BcipEnums.INVALID_PARAMETERS
        elif not isinstance(self._outA,Scalar):
            # o.w. the output should be a scalar
            return BcipEnums.INVALID_PARAMETERS
        
        # if the inputs are scalars, ensure they are numeric
        if isinstance(self._inA,Scalar) and \
           not self._inA.data_type in Scalar.valid_numberic_types():
            return BcipEnums.INVALID_PARAMETERS
        
        if isinstance(self._inB,Scalar) and \
           not self._inB.data_type in Scalar.valid_numeric_types():
            return BcipEnums.INVALID_PARAMETERS
        
        if isinstance(self._outA,Scalar) and \
           not self._outA.data_type in Scalar.valid_numeric_types():
            return BcipEnums.INVALID_PARAMETERS
        
        # check the shapes
        if isinstance(self._inA,Tensor):
            inA_shape = self._inA.shape
        else:
            inA_shape = (1,)
        
        if isinstance(self._inB,Tensor):
            inB_shape = self._inB.shape
        else:
            inB_shape = (1,)
        
        # determine what the output shape should be
        try:
            phony_a = np.zeros(inA_shape)
            phony_b = np.ones(inB_shape)
            
            phony_out = phony_a / phony_b
        
        except ValueError:
            # these dimensions cannot be broadbast together
            return BcipEnums.INVALID_PARAMETERS
        
        out_shape = phony_out.shape
        
        # if the output is a virtual tensor and has no defined shape, set the shape now
        if isinstance(self._outA,Tensor) and self._outA.virtual \
           and len(self._outA.shape) == 0:
            self._outA.shape = out_shape
        
        # ensure the output shape equals the expected output shape
        if isinstance(self._outA,Tensor) and self._outA.shape != out_shape:
            return BcipEnums.INVALID_PARAMETERS
        elif isinstance(self._outA,Scalar) and out_shape != (1,):
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS
    
    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_inB, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data1, input_data2, output_data):
        try:
            output_data.data = input_data1.data / input_data2.data

        except ValueError:
            return BcipEnums.EXE_FAILURE
            
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        return self.process_data(self._inA, self._inB, self._outA)
    
    @classmethod
    def add_division_node(cls,graph,inA,inB,outA):
        """
        Factory method to create a divsion kernel and add it to a graph
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
        graph.add_node(node)
        
        return node
