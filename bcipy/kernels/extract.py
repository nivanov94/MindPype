"""
Created on Thu Dec 12 12:12:25 2019

@author: ivanovn
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.bcip_enums import BcipEnums
from classes.array import Array
from classes.tensor import Tensor
from classes.bcip import BCIP

import numpy as np


class ExtractKernel(Kernel):
    """
    Kernel to extract a portion of a tensor or array

    Parameters
    ----------
    graph : Graph Object
        - Graph that the kernel should be added to

    inA : Tensor or Array object
        - Input trial data

    Indicies : list slices, list of ints
        - Indicies within inA from which to extract data

    outA : Tensor object
        - Extracted trial data

    reduce_dims : bool, default = False
        - Remove singleton dimensions if true, don't squeeze otherwise
    """
    
    def __init__(self,graph,inA,indices,outA,reduce_dims):
        super().__init__('Extract',BcipEnums.INIT_FROM_NONE,graph)
        self._in = inA
        self._out = outA
        self._indices = indices
        self._reduce_dims = reduce_dims

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
        
        # input must be a tensor or array
        # additionally, if the input is a tensor, the output should also be a
        # tensor
        if (not (isinstance(self._in,Tensor) and isinstance(self._out,Tensor)))\
        and (not (isinstance(self._in,Array) and isinstance(self._out,BCIP))):
            return BcipEnums.INVALID_PARAMETERS
        

        # if the input is an array, then the there should only be a single 
        # dimension to extract with a value of zero
        if isinstance(self._in,Array):
            if len(self._indices) != 1:
                return BcipEnums.INVALID_PARAMETERS
            
            # check that the index to extract do not exceed the capacity
            for index in self._indices[0]:
                if index >= self._in.capacity:
                    return BcipEnums.INVALID_PARAMETERS
                
            array_indices = self._indices[0]
        
            # if the output is a tensor, check the shape
            if isinstance(self._out,Tensor):
                if not isinstance(self._in.get_element(array_indices[0]),Tensor):
                    return BcipEnums.INVALID_PARAMETERS
                
                depth = len(array_indices)
                
                if self._out.virtual and len(self._out.shape) == 0:
                    if depth == 1 and self._reduce_dims:
                        self._out.shape = self._in.get_element(array_indices[0]).shape
                    else:
                        self._out.shape = (depth,) + \
                                        self._in.get_element(array_indices[0]).shape

                if depth == 1:
                    if len(self._out.shape) == 2:
                        output_sz = self._in.get_element(array_indices[0])
                        if self._out.shape != output_sz:
                            return BcipEnums.INVALID_PARAMETERS
                    elif len(self._out.shape) == 3:
                        if depth == 1:
                            return BcipEnums.INVALID_PARAMETERS
                        
                        output_sz = (1,) + self._in.get_element(array_indices[0])
                        if self._out.shape != output_sz:
                            return BcipEnums.INVALID_PARAMETERS
                    else:
                        # invalid shape
                        return BcipEnums.INVALID_PARAMETERS
                else:
                    output_sz = (depth,) + self._in.get_element(array_indices[0])
                    if self._out.shape != output_sz:
                        return BcipEnums.INVALID_PARAMETERS
                    
        
        elif isinstance(self._in,Tensor):
            # check that the number of dimensions indicated does not exceed 
            # the tensor's rank
            if len(self._indices) != len(self._in.shape):
                return BcipEnums.INVALID_PARAMETERS
            
            output_sz = []
            for axis in range(len(self._indices)):
                if self._indices[axis] != ":":
                    axis_indices = self._indices[axis]
                    if isinstance(axis_indices,int):
                        axis_indices = (axis_indices,)
                    for index in axis_indices:
                        # check that the index is valid for the given axis
                        if index < 0 or index >= self._in.shape[axis]:
                            return BcipEnums.INVALID_PARAMETERS
                    
                    if not self._reduce_dims or len(self._indices[axis]) > 1:
                        output_sz.append(len(axis_indices))
                else:
                    output_sz.append(self._in.shape[axis])
            
            # check that the output tensor's dimensions are valid
            output_sz = tuple(output_sz)
            
            if self._out.virtual and len(self._out.shape) == 0:
                self._out.shape = output_sz
            
            if self._out.shape != output_sz:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
        
    def initialization_execution(self):
        """
        Update initialization output if downstream nodes are missing training data
        """
        sts = self.process_data(self._init_inA, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    def process_data(self, input_data, output_data):
        """
        Process trial data according to the Numpy function
        """
        
        if isinstance(input_data, Array):
            # extract the elements and set in the output array
            for i in range(len(self._indices[0])):
                if isinstance(output_data,Array):
                    elem = input_data.get_element(self._indices[0][i])
                    output_data.set_element(i,elem)
                elif isinstance(output_data,Tensor):
                    return BcipEnums.NOT_YET_IMPLEMENTED
        else:
            # tensor case
            ix_grid = []
            for axis in range(len(self._indices)):
                axis_indices = self._indices[axis]
                if axis_indices == ":":
                    ix_grid.append([_ for _ in range(input_data.shape[axis])])
                else:
                    if isinstance(axis_indices,int):
                        ix_grid.append([axis_indices])
                    else:
                        ix_grid.append(list(axis_indices))

            npixgrid = np.ix_(*ix_grid)
            extr_data = self._in.data[npixgrid]
            
            if self._reduce_dims:
                extr_data = np.squeeze(extr_data)
                
            output_data.data = extr_data
        
        return BcipEnums.SUCCESS

    def execute(self):
        """
        Execute the kernel function
        """
        
        return self.process_data(self._in, self._out)
    
    @classmethod
    def add_extract_node(cls,graph,inA,indices,outA,reduce_dims=False):
        """
        Factory method to create an extract kernel 
        and add it to a graph as a generic node object.

         graph : Graph Object
            - Graph that the kernel should be added to

        inA : Tensor or Array object
            - Input trial data

        Indicies : list slices, list of ints
            - Indicies within inA from which to extract data

        outA : Tensor object
            - Extracted trial data

        reduce_dims : bool, default = False
            - Remove singleton dimensions if true, don't squeeze otherwise
        """
        
        # create the kernel object
        k = cls(graph,inA,indices,outA,reduce_dims)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node