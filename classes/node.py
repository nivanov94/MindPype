# -*- coding: utf-8 -*-
"""
Node.py - Generic node class for BCIP

@author: ivanovn
"""

from bcip import BCIP
from bcip_types import BcipEnums


class Node(BCIP):
    """
    Generic node object containing a kernel function
    """
    
    def __init__(self,kernel,n_params,params):
        super().__init__(BcipEnums.NODE)
        
        self.kernel = kernel
        self.n_params = n_params
        self.params = params
        
    
    def getInputs(self):
        """
        Return a list of all the node's inputs
        """
        inputs = []
        for p in self.params:
            if p.getDirection() == BcipEnums.INPUT:
                inputs.append(p.data)
        
        return inputs
    
    def getOutputs(self):
        """
        Return a list of all the node's outputs
        """
        outputs = []
        for p in self.params:
            if p.getDirection() == BcipEnums.OUTPUT:
                outputs.append(p.data)
        
        return outputs
            