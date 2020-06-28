from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.array import Array
from classes.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

class TensorStackKernel(Kernel):
    """
    Kernel to stack 2 tensors into a single tensor
    """
    
    def __init__(self,graph,inA,inB,outA,axis=None):
        super().__init__('TensorStack',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
        self._outA = outA
        self._axis = axis
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
                
        # inA must be an array and outA must be a tensor
        if (not (isinstance(self._inA,Tensor) and 
                 isinstance(self._inB,Tensor) and
                 isinstance(self._outA,Tensor))):
            return BcipEnums.INVALID_PARAMETERS
        
        stack_axis = self._axis
        
        # ensure that all the tensors in inA are the same size
        tensor_shapes = [self._inA.shape, self._inB.shape]
        
        if len(set(tensor_shapes)) != 1:
            # tensors in array are different sizes OR array is empty
            return BcipEnums.INVALID_PARAMETERS
        
        # determine the output dimensions
        output_shape = self._inA.shape[:stack_axis] + (2,) + self._inA.shape[stack_axis:]
        
        # check the output dimensions are valid
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = output_shape
        
        # ensure the output shape equals the expected output shape
        if self._outA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using numpy functions
        """
        
        stack_axis = self._axis
        
        try:
            input_tensors = [self._inA.data, self._inB.data]
            
            output_data = np.stack(input_tensors,axis=stack_axis)
        
        except ValueError:
            return BcipEnums.EXE_FAILURE
        
        # set the data of the output tensor
        self._outA.data = output_data
        
        return BcipEnums.SUCCESS
    
    
    @classmethod
    def add_tensor_stack_node(cls,graph,inA,inB,outA,axis=0):
        """
        Factory method to create a tensor stack kernel and add it to a graph
        as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,inB,outA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node


