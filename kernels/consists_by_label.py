# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:41:02 2020

@author: Nick
"""


from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.array import Array
from classes.bcip_enums import BcipEnums

import numpy as np

from pyriemann.utils.distance import distance_riemann

class ConsistencyByLabelKernel(Kernel):
    """
    Calculates multiple consistency metrics using labelled tensors and means in arrays 
    """
    
    def __init__(self,graph,covs,labels,means,consists):
        super().__init__('ConsistByLabel',BcipEnums.INIT_FROM_NONE,graph)
        self._covs  = covs
        self._labels = labels
        self._means = means
        self._consists = consists
            
    def initialize(self):
        """
        No internal state to setup
        """
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        for p in (self._covs,self._means,self._labels,self._consists):
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
        
        # extract all the labels
        unique_labels = set()
        label_covs = {}
        for i in range(self._labels.num_elements):
            l = self._labels.get_queued_element(i).data
            unique_labels.add(l)
            
            if l in label_covs:
                label_covs[l].append(self._covs.get_queued_element(i).data)
            else:
                label_covs[l] = [self._covs.get_queued_element(i).data]
        
        unique_labels = list(unique_labels)
        unique_labels.sort()
        
        # calculate the means for each label
        for i in range(len(unique_labels)):
            l = unique_labels[i]
            l_mean = self._means.get_element(i).data
            disp = sum([distance_riemann(l_mean, t) for t in label_covs[l]])
            consist = np.zeros((1,1))
            consist[0,0] = disp / len(label_covs[l])
            consist_tensor = self._consists.get_element(i)
            consist_tensor.data = consist
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_consist_by_label_node(cls,graph,covs,labels,means,consists):
        """
        Factory method to create a Riemann mean be label calculating kernel
        """
        
        # create the kernel object
        k = cls(graph,covs,labels,means,consists)
        
        # create parameter objects for the input and output
        params = (Parameter(covs,BcipEnums.INPUT), 
                  Parameter(labels,BcipEnums.INPUT),
                  Parameter(means,BcipEnums.INPUT),
                  Parameter(consists,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    