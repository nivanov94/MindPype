# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:22:02 2020

@author: Nick
"""

from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.array import Array
from classes.bcip_enums import BcipEnums
from .utils.data_extraction import extract_nested_data

import numpy as np

from sklearn.svm import SVC


class SVMClassifierKernel(Kernel):
    """
    SVM Classifier Kernel
    """
    
    def __init__(self,graph,X,y_bar,
                 init_style,init_params):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('SVM',init_style,graph)
        self._X = X
        self._y_bar = y_bar
        
        self._initialize_params = init_params
        
        if init_style == BcipEnums.INIT_FROM_DATA:
            # model will be trained using data in tensor object at later time
            self._initialized = False
            self._classifier = SVC(kernel=init_params['kernel'],
                                   shrinking=init_params['shrinking'],
                                   tol=init_params['tol'],
                                   C=init_params['C'],
                                   degree=init_params['degree'],
                                   gamma=init_params['gamma'],
                                   coef0=init_params['coef0'],
                                   probability=init_params['probability'],
                                   cache_size=init_params['cache_size'],
                                   class_weight=init_params['class_weight'],
                                   max_iter=init_params['max_iter'],
                                   decision_function_shape=init_params['decision_func_shape'],
                                   break_ties=init_params['break_ties'],
                                   random_state=init_params['random_state'])
            
        elif init_style == BcipEnums.INIT_FROM_COPY:
            # model is copy of predefined MDM model object
            self._classifier = init_params['model']
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
        
        if (not (isinstance(self._initialize_params['training_data'],Tensor) or 
                 isinstance(self._initialize_params['training_data'],Array)) or 
            not isinstance(self._initialize_params['labels'],Tensor)):
                return BcipEnums.INITIALIZATION_FAILURE
        
        if isinstance(self._initialize_params['training_data'],Tensor): 
            X = self._initialize_params['training_data'].data
        else:
            try:
                # extract the data from a potentially nested array of tensors
                X = extract_nested_data(self._initialize_params['training_data'])
            except:
                return BcipEnums.INITIALIZATION_FAILURE    
            
        y = self._initialize_params['labels'].data
        
        # ensure the shpaes are valid
        if len(X.shape) != 2 or len(y.shape) != 1:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[0] != y.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE
        
        self._classifier.fit(X,y)
        
        self._initialized = True
        
        return BcipEnums.SUCCESS
    
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self._X,Tensor)) or \
            (not (isinstance(self._y_bar,Tensor) or 
                  isinstance(self._y_bar,Scalar))):
                return BcipEnums.INVALID_PARAMETERS
        
        
        # input tensor should be two-dimensional
        if len(self._X.shape) != 2:
            return BcipEnums.INVALID_PARAMETERS
        

        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if isinstance(self._y_bar,Tensor) and self._y_bar.virtual:
            self._y_bar.shape = (self._X.shape[0],1)
        
        
        # check for dimensional alignment
        if isinstance(self._y_bar,Scalar):
            # input tensor should only be a single trial
            if self._X.shape[0] != 1 and self._X.shape[1] != 1:
                return BcipEnums.INVALID_PARAMETERS
        else:
            # check that the dimensions of the output match the dimensions of
            # input
            if self._y_bar.shape[0] != (self._X.shape[0],1):
                return BcipEnums.INVALID_PARAMETERS

        
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel by classifying the input trials
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        y_b = self._classifier.predict(self._X.data)
        
        if isinstance(self._y_bar,Scalar):
            self._y_bar.data = int(y_b)
        else:
            self._y_bar.data = y_b
            
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_untrained_SVM_node(cls,graph,X,y_bar,
                               training_data,labels,
                               C=1.0,kernel='rbf',degree=3,gamma='scale',
                               coef0=0.0,shrinking=True,probability=False,
                               tol=0.0001,cache_size=200,class_weight=None,
                               max_iter=-1,decision_func_shape='ovr',
                               break_ties=False,random_state=None):
        """
        Factory method to create a SVM classifier node and add it to a graph
        
        Note that the node will have to be initialized (i.e. trained) prior 
        to execution of the kernel.
        """
        
        # create the kernel object            
        init_params = {'training_data' : training_data, 
                       'labels'        : labels,
                       'C'             : C,
                       'kernel'        : kernel,
                       'degree'        : degree,
                       'gamma'         : gamma,
                       'coef0'         : coef0,
                       'shrinking'     : shrinking,
                       'probability'   : probability,
                       'tol'           : tol,
                       'cache_size'    : cache_size,
                       'class_weight'  : class_weight,
                       'max_iter'      : max_iter,
                       'decision_func_shape' : decision_func_shape,
                       'break_ties'    : break_ties,
                       'random_state'  : random_state}
        k = cls(graph,X,y_bar,BcipEnums.INIT_FROM_DATA,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(X,BcipEnums.INPUT),
                  Parameter(y_bar,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
    
    @classmethod
    def add_trained_SVM_node(cls,graph,X,y_bar,model):
        """
        Factory method to create a pre-trained LDA classifier node
        
        The kernel will contain a reference to the model rather than making a 
        deep-copy. Therefore any changes to the classifier object outside
        will effect the classifier here.
        """

        # sanity check that the input is actually an MDM model
        if not isinstance(model,SVC):
            return None
        
        # create the kernel object
        init_params = {'model' : model}
        k = cls(graph,X,y_bar,BcipEnums.INIT_FROM_COPY,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(X,BcipEnums.INPUT),
                  Parameter(y_bar,BcipEnums.OUTPUT))
        
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node