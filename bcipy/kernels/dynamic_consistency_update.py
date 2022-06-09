# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:29:49 2020

@author: Nick
"""


from ..classes.kernel import Kernel
from ..classes.node import Node
from ..classes.parameter import Parameter
from ..classes.tensor import Tensor
from ..classes.array import Array
from ..classes.bcip_enums import BcipEnums

import numpy as np

from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

class DynamicConsistencyUpdateKernel(Kernel):
    """
    Calculates an updated consistency metric using block trials
    """
    
    def __init__(self,graph,mean,last_trial,block_trials,consist):
        """
        Kernel calculates an exponentially weighted riemann mean
        """
        super().__init__('DynamicConsistUpdate',BcipEnums.INIT_FROM_NONE,graph)
        self._mean = mean
        self._block_trials = block_trials
        self._last_trial = last_trial
        self._consist = consist
            
    def initialize(self):
        """
        No internal state to setup
        """
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        ## TODO implement verification logic
  
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        
        # create array of trials executed this block
        num_block_trials = self._block_trials.num_elements
        block_trials = []
        if num_block_trials > 0:
            block_trials = []
            for i in range(num_block_trials):
                block_trials.append(self._block_trials.get_element(i).data)
        
        block_trials.append(self._last_trial.data)
        
        # just calculate the variance of trials within the block, ignore the
        # running mean for now
        block_mean = mean_riemann(np.stack(block_trials))
        
        consist = np.zeros((1,1))
        consist[0,0] = sum([distance_riemann(block_mean,t) for t in block_trials])
        consist[0,0] /= len(block_trials)
        
        self._consist.data = consist
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_dynamic_consistency_update_node(cls,graph,mean,
                                            last_trial,block_trials,
                                            consist):
        """
        Factory method to create a dynamic consistency metric update
        """
        
        # create the kernel object
        k = cls(graph,mean,last_trial,block_trials,consist)
        
        # create parameter objects for the input and output
        params = (Parameter(mean,BcipEnums.INPUT),
                  Parameter(block_trials,BcipEnums.INPUT),
                  Parameter(last_trial,BcipEnums.INPUT),
                  Parameter(consist,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    