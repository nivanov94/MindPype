# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:13:37 2019

FilterKernel.py - Define the filter kernel for BCIP

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.filter import Filter
from classes.bcip_enums import BcipEnums

from scipy import signal

class FilterKernel(Kernel):
    """
    Filter a tensor along the first non-singleton dimension
    """
    
    def __init__(self,graph,inputA,filt,outputA):
        super().__init__('Filter',BcipEnums.INIT_FROM_COPY,graph)
        self.inputA  = inputA
        self.filt = filt
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
            (not isinstance(self.outputA,Tensor)) or \
            (not isinstance(self.filt,Filter)):
                return BcipEnums.INVALID_PARAMETERS
        
        # do not support filtering directly with zpk filter repesentation
        if self.filt.implementation == 'zpk':
            return BcipEnums.NOT_SUPPORTED
        
        # check the shape
        input_shape = self.inputA.shape
        input_rank = len(input_shape)
        
        # determine what the output shape should be
        if input_rank == 0:
            return BcipEnums.INVALID_PARAMETERS
        else:
            output_shape = input_shape
        
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
        Execute the kernel function using the scipy module function
        """
        
        shape = self.inputA.shape
        axis = next((i for i, x in enumerate(shape) if x != 1))
        
        if self.filt.implementation == 'ba':
            self.outputA.data = signal.lfilter(self.filt.coeffs['b'],\
                                               self.filt.coeffs['a'],\
                                               self.inputA.data, \
                                               axis=axis)
        else:
            self.outputA.data = signal.sosfilt(self.filt.coeffs['sos'],\
                                               self.inputA.data,\
                                               axis=axis)
        
        return BcipEnums.SUCCESS
            
    
    @classmethod
    def addFilterNode(cls,graph,inputA,filt,outputA):
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
        graph.addNode(node)
        
        return node