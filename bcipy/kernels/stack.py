"""
Created on Tue Dec 10 13:05:29 2019

stack.py - Define the stack kernel that combines several tensors along
a new dimension

@author: ivanovn
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.array import Array
from classes.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

class StackKernel(Kernel):
    """
    Kernel to stack multiple tensors into a single tensor
    """
    
    def __init__(self,graph,inA,outA,axis=None):
        super().__init__('stack',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._outA = outA
        self._axis = axis

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
                
        # inA must be an array and outA must be a tensor
        if not (isinstance(self._inA,Array) and isinstance(self._outA,Tensor)):
            return BcipEnums.INVALID_PARAMETERS
        
        # if an axis was provided, it must be a scalar
        if self._axis != None and not isinstance(self._axis,Scalar):
            return BcipEnums.INVALID_PARAMETERS
        
        stack_axis = self._axis.data if self._axis != None else 0
        
        # ensure that all the tensors in inA are the same size
        tensor_shapes = [self._inA.get_element(i).shape
                                    for i in range(self._inA.capacity)]
        
        if len(set(tensor_shapes)) != 1:
            # tensors in array are different sizes OR array is empty
            return BcipEnums.INVALID_PARAMETERS
        
        # determine the output dimensions
        output_shape = tensor_shapes[0][:stack_axis] + (self._inA.capacity,) \
                         + tensor_shapes[0][stack_axis:]
        
        # check the output dimensions are valid
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = output_shape
        
        # ensure the output shape equals the expected output shape
        if self._outA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS

    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        stack_axis = self._axis.data if self._axis != None else 0
        
        try:
            input_tensors = [self._inA.get_element(i) for i 
                                             in range(self._inA.capacity)]
            
            input_data = [t.data for t in input_tensors]
            output_data = np.stack(input_data,axis=stack_axis)
        
        except ValueError:
            return BcipEnums.EXE_FAILURE
        
        # set the data of the output tensor
        self._outA.data = output_data
        
        return BcipEnums.SUCCESS


    def execute(self):
        """
        Execute the kernel function using numpy functions
        """
        
        stack_axis = self._axis.data if self._axis != None else 0
        
        try:
            input_tensors = [self._inA.get_element(i) for i 
                                             in range(self._inA.capacity)]
            
            input_data = [t.data for t in input_tensors]
            output_data = np.stack(input_data,axis=stack_axis)
        
        except ValueError:
            return BcipEnums.EXE_FAILURE
        
        # set the data of the output tensor
        self._outA.data = output_data
        
        return BcipEnums.SUCCESS
    
    
    @classmethod
    def add_stack_node(cls,graph,inA,outA,axis=None):
        """
        Factory method to create a stack kernel and add it to a graph
        as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
