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
    
    def __init__(self,graph,inputA,outputA,init_style,initialize_params):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('RiemannMDM',init_style,graph)
        self._inputA  = inputA
        self._outputA = outputA
        
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
            return self.train_classifier()
        else:
            # kernel contains a reference to a pre-existing MDM object, no
            # need to train here
            self._initialized = True
            return BcipEnums.SUCCESS
        
    def train_classifier(self):
        """
        Train the classifier
        
        The method will update the kernel's internal representation of the
        classifier
        """
        
        if (not isinstance(self._initialize_params['training_data'],Tensor)) or \
            (not isinstance(self._initialize_params['labels'],Tensor)):
                return BcipEnums.INITIALIZATION_FAILURE
        
        X = self._initialize_params['training_data'].data
        y = self._initialize_params['labels'].data
        
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
        if (not isinstance(self._inputA,Tensor)) or \
            (not (isinstance(self._outputA,Tensor) or 
                  isinstance(self._outputA,Scalar))):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self._inputA.shape
        input_rank = len(input_shape)
        
        # input tensor should not be greater than rank 3
        if input_rank > 3 or input_rank < 2:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (isinstance(self._outputA,Tensor) and self._outputA.virtual \
            and self._outputA.shape == None):
            if input_rank == 2:
                self._outputA.shape = (1,)
            else:
                self._outputA.shape = (input_shape[0],)
        
        
        # check for dimensional alignment
        
        if isinstance(self._outputA,Scalar):
            # input tensor should only be a single trial
            if len(self._inputA.shape) == 3:
                # first dimension must be equal to one
                if self._inputA.shape[0] != 1:
                    return BcipEnums.INVALID_PARAMETERS
        else:
            # check that the dimensions of the output match the dimensions of
            # input
            if self._inputA.shape[0] != self._outputA.shape[0]:
                return BcipEnums.INVALID_PARAMETERS

            # output tensor should be one dimensional
            if len(self._outputA.shape) > 1:
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
        input_data = self._inputA.data
        input_data = input_data[np.newaxis,:,:]
        
        self._outputA.data = self._classifier.predict(input_data)
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_untrained_riemann_MDM_node(cls,graph,inputA,outputA,\
                                       training_data,labels):
        """
        Factory method to create an untrained riemann minimum distance 
        to the mean classifier kernel and add it to a graph
        as a generic node object.
        
        Note that the node will have to be initialized (i.e. trained) prior 
        to execution of the kernel.
        """
        
        # create the kernel object
        init_params = {'training_data' : training_data, 
                       'labels'        : labels}
        k = cls(graph,inputA,outputA,BcipEnums.INIT_FROM_DATA,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
    
    @classmethod
    def add_trained_riemann_MDM_node(cls,graph,inputA,outputA,\
                                     model):
        """
        Factory method to create a riemann minimum distance 
        to the mean classifier kernel containing a copy of a pre-trained
        MDM classifier and add it to a graph as a generic node object.
        
        The kernel will contain a reference to the model rather than making a 
        deep-copy. Therefore any changes to the classifier object outside
        will effect the classifier here.
        """

        # sanity check that the input is actually an MDM model
        if not isinstance(model,classification.MDM):
            return None
        
        # create the kernel object
        init_params = {'model' : model}
        k = cls(graph,inputA,outputA,BcipEnums.INIT_FROM_COPY,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT), \
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
