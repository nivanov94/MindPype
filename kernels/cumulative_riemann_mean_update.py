# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:10:39 2020

@author: Nick
"""


from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.array import Array
from classes.bcip_enums import BcipEnums

import numpy as np
from scipy.linalg import fractional_matrix_power

from pyriemann.utils.mean import mean_riemann

class CumulativeRiemannMeanUpdateKernel(Kernel):
    """
    Calculates the Riemann mean of covariances contained in a tensor
    """
    
    def __init__(self,graph,prev_mean,block_trials,last_trial,new_mean,weights):
        """
        Kernel calculates an exponentially weighted riemann mean
        """
        super().__init__('ExpRiemannMeanUpdate',BcipEnums.INIT_FROM_NONE,graph)
        self._prev_mean = prev_mean
        self._block_trials = block_trials
        self._last_trial = last_trial
        self._new_mean = new_mean
        
        self._w = weights
    
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
        if num_block_trials > 0:
            block_trials = []
            for i in range(num_block_trials):
                block_trials.append(self._block_trials.get_element(i).data)
            new_trials = np.stack(block_trials)
            last_trial_data = np.expand_dims(self._last_trial.data,axis=0)
            new_trials = np.concatenate((new_trials,last_trial_data),axis=0)
        else:
            new_trials = np.expand_dims(self._last_trial.data,axis=0)
        
        # # concat new trials to existing mean
        # prev_mean_data = np.expand_dims(self._prev_mean.data,axis=0)
        # mean_calc_input = np.concatenate((prev_mean_data,new_trials),axis=0)
        # mean_calc_weights = self._w[:mean_calc_input.shape[0]]
        block_mean = mean_riemann(new_trials)
        
        #new_mean = mean_riemann(mean_calc_input,sample_weight=mean_calc_weights)
        prev_mean_data = self._prev_mean.data
        prev_mean_inv_sqrt = fractional_matrix_power(prev_mean_data,-1/2)
        prev_mean_sqrt = fractional_matrix_power(prev_mean_data,1/2)
        
        inner = np.matmul(prev_mean_inv_sqrt,np.matmul(block_mean,prev_mean_inv_sqrt))
        inner_pw = fractional_matrix_power(inner,(1-self._w[0]))
        new_mean = np.matmul(prev_mean_sqrt,np.matmul(inner_pw,prev_mean_sqrt))
        
        self._new_mean.data = new_mean
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_cumulative_rmean_update_node(cls,graph,prev_mean,
                                         block_trials,last_trial,
                                         new_mean,weights):
        """
        Factory method to create a cumulative Riemann mean calculating kernel
        """
        
        # create the kernel object
        k = cls(graph,prev_mean,block_trials,last_trial,new_mean,weights)
        
        # create parameter objects for the input and output
        params = (Parameter(prev_mean,BcipEnums.INPUT),
                  Parameter(block_trials,BcipEnums.INPUT),
                  Parameter(last_trial,BcipEnums.INPUT),
                  Parameter(new_mean,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
