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
    
    def __init__(self,block,inputA,outputA):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('Transpose',block)
        self.inputA  = inputA
        self.outputA = outputA
        
        self._initialized = False
        self._classifier = classification.MDM()
        
    
    def initialize(self):
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
        
        self.initialize()
        
        return BcipEnums.SUCCESS
    
    def loadClassifier(self,mdm_classifier):
        """
        Copy the reference to a previously trained MDM classifier and store
        it within the node for use. Calling this method will initialize the
        kernel making it ready for execution.
        
        This method allows you to bypass training the kernel directly by using
        a pre-trained model.
        """
        # sanity check that the input is actually an MDM model
        if not isinstance(mdm_classifier,classification.MDM):
            return BcipEnums.FAILURE_INVALID_TYPE
        
        self._classifier = mdm_classifier
        
        self.initialize()
        
        return BcipEnums.SUCCESS
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self.inputA,Tensor)) or \
            (not (isinstance(self.outputA,Tensor) or 
                  isinstance(self.outputA,Scalar))):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self.inputA.shape
        input_rank = len(input_shape)
        
        # input tensor should not be greater than rank 3
        if input_rank > 3 or input_rank < 2:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (isinstance(self.outputA,Tensor) and self.outputA.isVirtual() \
            and self.outputA.getShape == None):
            if input_rank == 2:
                self.outputA.setData(np.zeros((1,)))
            else:
                self.outputA.setData(np.zeros((input_shape[0],)))
        
        
        # check for dimensional alignment
        
        if isinstance(self.outputA,Scalar):
            # input tensor should only be a single trial
            if len(self.inputA.shape) == 3:
                # first dimension must be equal to one
                if self.inputA.shape[0] != 1:
                    return BcipEnums.INVALID_PARAMETERS
        else:
            # check that the dimensions of the output match the dimensions of
            # input
            if self.inputA.shape[0] != self.outputA.shape[0]:
                return BcipEnums.INVALID_PARAMETERS

            # output tensor should be one dimensional
            if len(self.outputA.shape) > 1:
                return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using the numpy transpose function
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        # pyriemann library requires input data to have 3 dimensions with the 
        # first dimension being 1
        input_data = self.inputA.getData()
        input_data = input_data[np.newaxis,:,:]
        
        # TODO handle tensors and scalar outputs differently
        self.outputA.setData(self._classifier.predict(input_data))
    
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