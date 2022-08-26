"""
Created on Thu Nov 21 15:13:37 2019

FilterKernel.py - Define the filter kernel for BCIP

@author: ivanovn
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.filter import Filter
from classes.bcip_enums import BcipEnums

from scipy import signal
import numpy as np

class FilterKernel(Kernel):
    """
    Filter a tensor along the first non-singleton dimension

    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inputA : Tensor or Scalar object
        - Input trial data

    filt : Filter object
        - BCIP Filter object outputted by bcipy.classes

    outputA : Tensor or Scalar object
        - Output trial data
    """
    
    def __init__(self,graph,inputA,filt,outputA):
        super().__init__('Filter',BcipEnums.INIT_FROM_COPY,graph)
        self._inputA  = inputA
        self._filt = filt
        self._outputA = outputA
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
            (not isinstance(self._filt,Filter)):
                return BcipEnums.INVALID_PARAMETERS
        
        # do not support filtering directly with zpk filter repesentation
        if self._filt.implementation == 'zpk':
            return BcipEnums.NOT_SUPPORTED
        
        # check the shape
        input_shape = self._inputA.shape
        input_rank = len(input_shape)
        
        # determine what the output shape should be
        if input_rank == 0:
            return BcipEnums.INVALID_PARAMETERS
        else:
            output_shape = input_shape
        
        # if the output is virtual and has no defined shape, set the shape now
        if self._outputA.virtual and len(self._outputA.shape) == 0:
            self._outputA.shape = output_shape
        
        # ensure the output tensor's shape equals the expected output shape
        if self._outputA.shape != output_shape:
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using the scipy module function
        """
        return self.process_data(self._inputA, self._outputA)
        

    def process_data(self, input_data, output_data):
        shape = input_data.shape
	# TODO make an axis input parameter instead
        try:
            axis = next((i for i, x in enumerate(shape) if x != 1))
        
            if self._filt.implementation == 'ba':
                output_data.data = signal.lfilter(self._filt.coeffs['b'],\
                                                self._filt.coeffs['a'],\
                                                input_data.data, \
                                                axis=axis)
            else:
                output_data.data = signal.sosfilt(self._filt.coeffs['sos'],\
                                                input_data.data,\
                                                axis=axis)
            return BcipEnums.SUCCESS

        except:
            return BcipEnums.EXE_FAILURE

    def initialization_execution(self):
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts


    @classmethod
    def add_filter_node(cls,graph,inputA,filt,outputA):
        """
        Factory method to create a filter kernel and add it to a graph
        as a generic node object.

        graph : Graph Object
            - Graph that the node should be added to

        inputA : Tensor or Scalar object
            - Input trial data

        filt : Filter object
            - BCIP Filter object outputted by bcipy.classes

        outputA : Tensor or Scalar object
            - Output trial data
        """
        
        # create the kernel object
        k = cls(graph,inputA,filt,outputA)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
