# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:12:33 2019

@author: ivanovn
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.bcip_enums import BcipEnums

import numpy as np

from pyriemann import classification

class RiemannMDMClassifierKernel(Kernel):
    """
    Riemannian Minimum Distance to the Mean Classifier
    """
    
    def __init__(self,block,inputA,outputA,n_classes):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('Transpose',block)
        self.inputA  = inputA
        self.outputA = outputA
        
        self._initialized = False
        self._n_classes = n_classes
        self._classifier = classification.MDM()
        
    
    def initialize(self,means):
        """
        Set the means for the classifier
        """
        self._initialized = True
        
    def trainClassifier(self,training_data,labels):
        """
        Train the classifier
        
        training_data - Tensor of training data. Should be of rank 3
                        with expected dimensions:
                            (trials x channels x channels)
                        
        labels - 1D tensor containing labels for each training sample
        
        The method will update the kernel's internal representation of the
        classifier
        """
        if (not isinstance(training_data,Tensor)) or \
            (not isinstance(labels,Tensor)):
                return BcipEnums.INITIALIZATION_FAILURE
        
        X = training_data.getData()
        y = labels.getData()
        
        # ensure the shpaes are valid
        if len(X.shape) != 3 or len(y.shape) != 1:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[0] != y.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE
        
        self._classifier.fit(X,y)
        
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self.inputA,Tensor)) or \
            (not (isinstance(self.outputA,Tensor) or 
                  isinstance(self.outputA,Scalar))):
                return BcipEnums.INVALID_PARAMETERS
        
        
        
    def execute(self):
        """
        Execute the kernel function using the numpy transpose function
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        self.outputA.setData(self._classifier.predict(self.inputA.getData()))
    
    @classmethod
    def addRiemannMDMKernel(cls,block,inputA,outputA):
        """
        Factory method to create a riemann minimum distance to the mean
        classifier kernel and add it to a block
        as a generic node object.
        """
        
        # create the kernel object
        k = cls(block,inputA,outputA)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(block,k,2,params)
        
        # add the node to the block
        block.addNode(node)
        
        return node