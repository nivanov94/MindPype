# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:38:34 2019

@author: ivanovn
"""

from ..classes.kernel import Kernel
from ..classes.node import Node
from ..classes.parameter import Parameter
from ..classes.bcip import BCIP
from ..classes.bcip_enums import BcipEnums
from ..classes.circle_buffer import CircleBuffer
from ..classes.scalar import Scalar
from ..classes.tensor import Tensor


class EnqueueKernel(Kernel):
    """
    Kernel to enqueue a BCIP object into a BCIP circle buffer
    """
    
    def __init__(self,graph,inA,queue):
        super().__init__('Enqueue',BcipEnums.INIT_FROM_NONE,graph)
        self._inA  = inA
        self._circle_buff = queue

        
    
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
        if not isinstance(self._inA,BCIP):
            return BcipEnums.INVALID_PARAMETERS
        
        if not isinstance(self._outA,CircleBuffer):
            return BcipEnums.INVALID_PARAMETERS

        # check that the buffer's capacity is at least 1
        if self._outA.capacity <= 1:
            return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
    
    def execute(self):
        """
        Execute the kernel function using numpy function
        """
        
        # need to make a deep copy of the object to enqueue
        cpy = self._inA.copy()
        self._outA.enqueue(cpy)
            
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_enqueue_node(cls,graph,inA,queue):
        """
        Factory method to create a enqueue kernel 
        and add it to a graph as a generic node object.
        """
        
        # create the kernel object
        k = cls(graph,inA,queue)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(queue,BcipEnums.INOUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node