# -*- coding: utf-8 -*-
"""
Node.py - Generic node class for BCIP

@author: ivanovn
"""

from numpy import block
from bcip import BCIP
from bcip_enums import BcipEnums


class Node(BCIP):
    """
    Generic node object containing a kernel function
    """
    
    def __init__(self,graph,kernel,params):
        sess = graph.session
        super().__init__(BcipEnums.NODE,sess)
        
        self._kernel = kernel
        self._params = params
        
    
    # API getters
    @property
    def kernel(self):
        return self._kernel
    
    def extract_inputs(self):
        """
        Return a list of all the node's inputs
        """
        inputs = []
        for p in self._params:
            if p.direction == BcipEnums.INPUT:
                inputs.append(p.data)
        
        return inputs
    
    def extract_outputs(self):
        """
        Return a list of all the node's outputs
        """
        outputs = []
        for p in self._params:
            if p.direction == BcipEnums.OUTPUT:
                outputs.append(p.data)
        
        return outputs
    
    def verify(self):
        """
        Verify the node is executable
        """
        return self.kernel.verify()
    
    def initialize(self):
        """
        Initialize the kernel function for execution
        """
        return self.kernel.initialize()
      