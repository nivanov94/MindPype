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
    
    def __init__(self,block,inputA,outputA,init_style,initialize_params):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('RiemannMDM',init_style,block)
        self.inputA  = inputA
        self.outputA = outputA
        
        self._initialize_params = initialize_params
        
        if init_style == BcipEnums.INIT_FROM_DATA:
            # model will be trained using data in tensor object at later time
            self._initialized = False
            self._classifier = None
        elif init_style == BcipEnums.INIT_FROM_COPY:
            # model is copy of predefined MDM model object
            self._classifier = initialize_params['model']
            self._initialized = True
        
    
    def initialize(self):
        """
        Set the means for the classifier
        """
        
        if self.init_style == BcipEnums.INIT_FROM_DATA:
            return self.trainClassifier()
        else:
            # kernel contains a reference to a pre-existing MDM object, no
            # need to train here
            self._initialized = True
            return BcipEnums.SUCCESS
        
    def trainClassifier(self):
        """
        Train the classifier
        
        The method will update the kernel's internal representation of the
        classifier
        """
        
        if (not isinstance(self._initialize_params['training_data'],Tensor)) or \
            (not isinstance(self._initialize_params['labels'],Tensor)):
                return BcipEnums.INITIALIZATION_FAILURE
        
        X = self._initialize_params['training_data'].getData()
        y = self._initialize_params['labels'].getData()
        
        # ensure the shpaes are valid
        if len(X.shape) != 3 or len(y.shape) != 1:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[0] != y.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE
        
        self._classifier = classification.MDM()
        self._classifier.fit(X,y)
        
        self._initialized = True
        
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
        Execute the kernel by classifying the input trials
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
    def addUntrainedRiemannMDMKernel(cls,block,inputA,outputA,\
                                     training_data,labels):
        """
        Factory method to create an untrained riemann minimum distance 
        to the mean classifier kernel and add it to a block
        as a generic node object.
        
        Note that the node will have to be initialized (i.e. trained) prior 
        to execution of the kernel.
        """
        
        # create the kernel object
        init_params = {'training_data' : training_data, 
                       'labels'        : labels}
        k = cls(block,inputA,outputA,BcipEnums.INIT_FROM_DATA,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(block,k,2,params)
        
        # add the node to the block
        block.addNode(node)
        
        return node
    
    
    @classmethod
    def addTrainedRiemannMDMKernel(cls,block,inputA,outputA,\
                                     model):
        """
        Factory method to create a riemann minimum distance 
        to the mean classifier kernel containing a copy of a pre-trained
        MDM classifier and add it to a block as a generic node object.
        
        The kernel will contain a reference to the model rather than making a 
        deep-copy. Therefore any changes to the classifier object outside
        will effect the classifier here.
        """

        # sanity check that the input is actually an MDM model
        if not isinstance(model,classification.MDM):
            return None
        
        # create the kernel object
        init_params = {'model' : model}
        k = cls(block,inputA,outputA,BcipEnums.INIT_FROM_COPY,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(block,k,2,params)
        
        # add the node to the block
        block.addNode(node)
        
        return node