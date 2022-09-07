# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:52:43 2019

concate.py  - Defines a concatenation kernel that concatenates multiple tensors
into a single tensor

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

class ConcatenationKernel(Kernel):
    """
    Kernel to concatenate multiple tensors into a single tensor

    Parameters
    ----------

    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Scalar object
        - First input trial data

    inB : Tensor or Scalar object
        - Second input trial data

    outA : Tensor or Scalar object
        - Output trial data

    axis : int or tuple of ints, default = 0
        - The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. Default is 0. 
        - See numpy.concatenate for more information
    """
    
    def __init__(self,graph,outA,inA,inB,axis):
        super().__init__('Concatenation',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._inB  = inB
        self._outA = outA
        self._axis = axis
        self._init_inA = None
        self._init_inB = None
        self._init_outA = None

        self._labels = None


    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        
        if self._init_outA != None:
            return self.initialization_execution()
        
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        if self._axis != None:            
            concat_axis = self._axis
        else:
            concat_axis = 0 # default axis
        
        
        # inA and inB must be a tensor
        if not (isinstance(self._inA,Tensor) and isinstance(self._inB,Tensor) \
                and isinstance(self._outA,Tensor)):
            return BcipEnums.INVALID_PARAMETERS
        
            
        # the dimensions along the catcat axis must be equal
        sz_A = self._inA.shape
        sz_B = self._inB.shape
        
        if len(sz_A) == len(sz_B):
            noncat_sz_A = [d for i,d in enumerate(sz_A) if i!=concat_axis]
            noncat_sz_B = [d for i,d in enumerate(sz_B) if i!=concat_axis]
            output_sz = noncat_sz_A[:]
            output_sz.insert(concat_axis,sz_A[concat_axis]+sz_B[concat_axis])
        elif len(sz_A) == len(sz_B)+1:
            # appending B to A
            noncat_sz_A = [d for i,d in enumerate(sz_A) if i!=concat_axis]
            noncat_sz_B = sz_B
            output_sz = noncat_sz_A[:]
            output_sz.insert(concat_axis,sz_A[concat_axis]+1)
        elif len(sz_A) == len(sz_B)-1:
            noncat_sz_B = [d for i,d in enumerate(sz_B) if i!=concat_axis]
            noncat_sz_A = sz_A
            output_sz = noncat_sz_B[:]
            output_sz.insert(concat_axis,sz_B[concat_axis]+1)
        else:
            return BcipEnums.INVALID_PARAMETERS
        
        # check if the remaining dimensions are the same
        if ((len(noncat_sz_A) != len(noncat_sz_B)) or 
            len(noncat_sz_A) != sum([1 for i,j in 
                                     zip(noncat_sz_A,noncat_sz_B) if i==j])):
            return BcipEnums.INVALID_PARAMETERS
        
        
        output_sz = tuple(output_sz)
        # check the output dimensions are valid
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = output_sz
        
        # ensure the output shape equals the expected output shape
        if self._outA.shape != output_sz:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS

    def initialization_execution(self):
        """
        Process initialization data. Called if downstream nodes are missing training data
        """
        sts = self.process_data(self._init_inA, self._init_inB, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data1, input_data2, output_data):
        """
        Process input data according to outlined kernel function
        """

        concat_axis = self._axis if self._axis != None else 0
        
        inA_data = input_data1.data
        inB_data = input_data2.data
        
        if len(inA_data.shape) == len(inB_data.shape)+1:
            # add a leading dimension for input B
            inB_data = np.expand_dims(inB_data,axis=0)
        elif len(inB_data.shape) == len(inA_data.shape)+1:
            inA_data = np.expand_dims(inA_data,axis=0)
        
        try:
            out_tensor = np.concatenate((inA_data,
                                         inB_data),
                                        axis=concat_axis)
        except ValueError:
            # dimensions are invalid
            return BcipEnums.EXE_FAILURE
        
        # set the data in the output tensor
        output_data.data = out_tensor
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function using numpy functions
        """
        return self.process_data(self._inA, self._inB, self._outA)
    
    
    @classmethod
    def add_concatenation_node(cls,graph,inA,inB,outA,axis=None):
        """
        Factory method to create a concatenation kernel and add it to a graph
        as a generic node object.

        Parameters
        ----------
        graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Scalar object
            - First input trial data

        inB : Tensor or Scalar object
            - Second input trial data

        outA : Tensor or Scalar object
            - Output trial data

        axis : int or tuple of ints, default = 0
            - The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. Default is 0. 
            - See numpy.concatenate for more information
        """
        
        # create the kernel object
        k = cls(graph,outA,inA,inB,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(outA,BcipEnums.OUTPUT),
                  Parameter(inA,BcipEnums.INPUT),
                  Parameter(inB,BcipEnums.INPUT))
        
    
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
