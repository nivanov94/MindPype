# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:53:06 2020

@author: Nick
"""

from ..classes.kernel import Kernel
from ..classes.node import Node
from ..classes.parameter import Parameter
from ..classes.tensor import Tensor
from ..classes.array import Array
from ..classes.bcip_enums import BcipEnums

import numpy as np

from pyriemann.utils.distance import distance_riemann
from itertools import combinations as iter_combs

class PairwiseRiemannDistanceKernel(Kernel):
    """
    Calculates pairwise riemann distances between covariance matrices in array 
    """
    
    def __init__(self,graph,covs,dists):
        super().__init__('PWRiemannDists',BcipEnums.INIT_FROM_NONE,graph)
        self._covs  = covs
        self._dists = dists
            
    def initialize(self):
        """
        No internal state to setup
        """
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        for p in (self._covs,self._dists):
            # first ensure the input and output are appropriate types
            if ((not isinstance(p,Tensor)) and
                (not isinstance(p,Array))):
                    return BcipEnums.INVALID_PARAMETERS
            
            if isinstance(self._covs,Tensor): #TODO
                return BcipEnums.NotImplemented
                
        
        ## TODO add remaining verification logic
  
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        
        Ndists = self._covs.capacity
        i = 0
        for (i1,i2) in iter_combs(range(Ndists), 2):
            m1 = self._covs.get_element(i1).data
            m2 = self._covs.get_element(i2).data
            pw_dist = np.zeros((1,1))
            pw_dist[0,0] = distance_riemann(m1,m2)
            pw_dist_tensor = self._dists.get_element(i)
            pw_dist_tensor.data = pw_dist
            i += 1
            
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_pairwise_dist_node(cls,graph,means,dists):
        """
        Factory method to create a pairwise distance kernel
        """
        
        # create the kernel object
        k = cls(graph,means,dists)
        
        # create parameter objects for the input and output
        params = (Parameter(means,BcipEnums.INPUT), 
                  Parameter(dists,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
