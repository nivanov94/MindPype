"""
Created on Thu Nov 20 18:20:03 2019

Transpose.py - Define the transpose kernel for BCIP

@author: ivanovn
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.bcip_enums import BcipEnums

import numpy as np

class TransposeKernel(Kernel):
    """
    Kernel to compute the tensor transpose
    
    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inputA : Tensor or Scalar object
        - Input trial data

    outputA : Tensor or Scalar object
        - Output trial data

    axes : tuple or list of ints, optional
        - If specified, it must be a tuple or list which contains a permutation of [0,1,..,N-1] where N is the number of axes of a. The i'th axis of the returned array will correspond to the axis numbered axes[i] of the input. If not specified, defaults to range(a.ndim)[::-1], which reverses the order of the axes.
    
    """
    
    def __init__(self,graph,inputA,outputA,axes):
        super().__init__('Transpose',BcipEnums.INIT_FROM_NONE,graph)
        self._inputA  = inputA
        self._outputA = outputA
        self._axes = axes

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
            (not isinstance(self._outputA,Tensor)) or \
            (self._axes != None and ((not isinstance(self._axes,tuple)) or \
            (len(self._axes) != len(self._inputA.shape)))):
                return BcipEnums.INVALID_PARAMETERS
        
        # check the shape
        input_shape = self._inputA.shape
        input_rank = len(input_shape)
        
        # determine what the output shape should be
        if input_rank == 0:
            return BcipEnums.INVALID_PARAMETERS
        
        if self._axes == None:
            output_shape = reversed(input_shape)
        else:
            output_shape = input_shape[self._axes]
               
        # if the output is virtual and has no defined shape, set the shape now
        if self._outputA.virtual and len(self._outputA.shape) == 0:
            self._outputA.shape = output_shape
        
        # ensure the output tensor's shape equals the expected output shape
        if self._outputA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS

    def initialization_execution(self):
        """
        Process initialization data. Called when downstream nodes are missing training data
        """
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts
    
    def process_data(self, input_data, output_data):
        """
        Process data according to outlined kernel function
        """
        output_data.data = np.transpose(input_data.data,axes=self._axes)

        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using the numpy transpose function
        """
        
        return self.process_data(self._inputA, self._outputA)


    @classmethod
    def add_transpose_node(cls,graph,inputA,outputA,axes=None):
        """
        Factory method to create a transpose kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------
        graph : Graph Object
            - Graph that the kernel should be added to

        inputA : Tensor or Scalar object
            - Input trial data

        outputA : Tensor or Scalar object
            - Output trial data

        axes : tuple or list of ints, default = None
            - If specified, it must be a tuple or list which contains a permutation of [0,1,..,N-1] where N is the number of axes of a. The i'th axis of the returned array will correspond to the axis numbered axes[i] of the input. If not specified, defaults to range(a.ndim)[::-1], which reverses the order of the axes.
        

        """
        
        # create the kernel object
        k = cls(graph,inputA,outputA,axes)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
