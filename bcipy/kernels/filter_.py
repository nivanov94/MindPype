"""
Created on Thu Nov 21 15:13:37 2019

FilterKernel.py - Define the filter kernel for BCIP

@author: ivanovn
"""

from ..classes.kernel import Kernel
from ..classes.node import Node
from ..classes.parameter import Parameter
from ..classes.tensor import Tensor
from ..classes.filter import Filter
from ..classes.bcip_enums import BcipEnums

from scipy import signal
import numpy as np

class FilterKernel(Kernel):
    """
    Filter a tensor along the first non-singleton dimension
    """
    
    def __init__(self,graph,inputA,filt,outputA):
        super().__init__('Filter',BcipEnums.INIT_FROM_COPY,graph)
        self._inputA  = inputA
        self._filt = filt
        self._outputA = outputA
        self._init_inA = None
        self._init_outA = None  
        self.graph = graph
    
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        if self.graph._missing_data:
            self.batch_processing()
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
        
        shape = self._inputA.shape
	# TODO make an axis input parameter instead
        axis = next((i for i, x in enumerate(shape) if x != 1))
        
        if self._filt.implementation == 'ba':
            self._outputA.data = signal.lfilter(self._filt.coeffs['b'],\
                                               self._filt.coeffs['a'],\
                                               self._inputA.data, \
                                               axis=axis)
        else:
            self._outputA.data = signal.sosfilt(self._filt.coeffs['sos'],\
                                               self._inputA.data,\
                                               axis=axis)
        
        return BcipEnums.SUCCESS
            
    def batch_processing(self):
        """
        Execute the kernel function using the scipy module function
        """
        shape = np.shape(self._init_inA)

        # TODO make an axis input parameter instead
        axis = next((i for i, x in enumerate(shape) if x != 1))
        
        if self._filt.implementation == 'ba':
            self._init_outA = signal.lfilter(self._filt.coeffs['b'],\
                                               self._filt.coeffs['a'],\
                                               self._init_inA, \
                                               axis=axis)
        else:
            self._init_outA = signal.sosfilt(self._filt.coeffs['sos'],\
                                               self._init_inA,\
                                               axis=axis)

        return BcipEnums.SUCCESS


    @classmethod
    def add_filter_node(cls,graph,inputA,filt,outputA):
        """
        Factory method to create a filter kernel and add it to a graph
        as a generic node object.
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
