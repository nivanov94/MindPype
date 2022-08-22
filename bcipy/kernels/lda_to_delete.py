# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:31:32 2020

@author: Nick
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.scalar import Scalar
from classes.array import Array
from classes.bcip_enums import BcipEnums
from .utils.data_extraction import extract_nested_data

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



class LDAClassifierKernel(Kernel):
    """
    LDA Classifier Kernel
    """
    
    def __init__(self,graph,X,y_bar,conf,pred_proba,
                 init_style,initialize_params):
        """
        Kernel takes Tensor input and produces scalar label representing
        the predicted class
        """
        super().__init__('LDA',init_style,graph)
        self._X = X
        self._y_bar = y_bar
        self._conf = conf
        self._pred_proba = pred_proba

        
        
        self._init_params = initialize_params
        self._init_inA = initialize_params['initialization_data']
        self._init_outA = None
        self._initialization_data = initialize_params['initialization_data']

        if init_style == BcipEnums.INIT_FROM_DATA:
            # model will be trained using data in tensor object at later time
            self._initialized = False
            self._classifier = LinearDiscriminantAnalysis(solver=initialize_params['solver'],
                                                          shrinkage=initialize_params['shrinkage'],
                                                          priors=initialize_params['priors'],
                                                          n_components=initialize_params['n_components'],
                                                          store_covariance=initialize_params['n_components'],
                                                          tol=initialize_params['tol'])
        elif init_style == BcipEnums.INIT_FROM_COPY:
            # model is copy of predefined MDM model object
            self._classifier = initialize_params['model']
            self._initialized = True
        
    
    def initialize(self):
        """
        Set the means for the classifier
        """
        sts1, sts2 = BcipEnums.SUCCESS, BcipEnums.SUCCESS
        if self.init_style == BcipEnums.INIT_FROM_DATA:
            sts1 = self.train_classifier()
            if self.graph._missing_data:
                sts2 = self.initialization_execution()
            if sts1 != BcipEnums.SUCCESS:
                return sts1
            elif sts2 != BcipEnums.SUCCESS:
                return sts2
            else:
                return BcipEnums.SUCCESS

        else:
            # kernel contains a reference to a pre-existing MDM object, no
            # need to train here
            self._initialized = True
            if self._init_outA.__class__ != NoneType:
                return self.initialization_execution()

            return BcipEnums.SUCCESS
        
    def train_classifier(self):
        """
        Train the classifier
        
        The method will update the kernel's internal representation of the
        classifier
        """
        try:
            initialization_data = self._initialization_data
        except KeyError:
            self._initialization_data = self._init_inA

        if (not (isinstance(self._initialization_data,Tensor) or 
                 isinstance(self._initialization_data,Array)) or 
            not isinstance(self._init_params['labels'],Tensor)):
                return BcipEnums.INITIALIZATION_FAILURE
        
        if isinstance(self._initialization_data,Tensor): 
            X = self._initialization_data.data
        else:
            try:
                # extract the data from a potentially nested array of tensors
                X = extract_nested_data(self._initialization_data)
            except:
                return BcipEnums.INITIALIZATION_FAILURE    
            
        y = self._init_params['labels'].data
        
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
            
        # check that optional parameters are valid
        for opt_out in (self._conf,self._pred_proba):
            
            if opt_out != None and (not isinstance(opt_out,Tensor) and 
                                    not isinstance(opt_out,Scalar)):
                return BcipEnums.INVALID_PARAMETERS
        
        
        # input tensor should be two-dimensional
        if len(self._X.shape) != 2:
            return BcipEnums.INVALID_PARAMETERS
        

        for output in (self._y_bar,self._conf,self._pred_proba):
            # if the output is a virtual tensor and dimensionless, 
            # add the dimensions now
            if output != None and isinstance(output,Tensor) and output.virtual:
                output.shape = (self._X.shape[0],1)
        
        
            # check for dimensional alignment
            if isinstance(output,Scalar):
                # input tensor should only be a single trial
                if self._X.shape[0] != 1 and self._X.shape[1] != 1:
                    return BcipEnums.INVALID_PARAMETERS
            elif isinstance(output,Tensor):
                # check that the dimensions of the output match the dimensions of
                # input
                if output.shape[0] != (self._X.shape[0],1):
                    return BcipEnums.INVALID_PARAMETERS

        
        return BcipEnums.SUCCESS

    def initialization_execution(self):
        try:
            self._init_outA.data = Scalar.create_from_value(self._classifier.predict(self._init_inA.data))
        except:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return BcipEnums.SUCCESS

    def process_data(self):
        #TODO: fix
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        
        y_b = self._classifier.predict(self._X.data)
        
        if isinstance(self._y_bar,Scalar):
            self._y_bar.data = int(y_b)
        else:
            self._y_bar.data = y_b
        
        if self._conf != None:
            conf = self._classifier.decision_function(self._X.data)
            if isinstance(self._conf,Scalar):
                self._conf.data = float(conf)
            else:
                self._conf.data = conf
        
        if self._pred_proba != None:
            prob = self._classifier.predict_proba(self._X.data)
            if isinstance(self._pred_proba,Scalar):
                self._pred_proba = float(prob)
            else:
                self._pred_proba = prob
            
        
        return BcipEnums.SUCCESS
    
    def execute(self):
        """
        Execute the kernel by classifying the input trials
        """
        
    
    @classmethod
    def add_untrained_LDA_node(cls,graph,X,y_bar,
                               initialization_data,labels,
                               pred_proba=None,conf=None,
                               solver='eigen',shrinkage=None,
                               priors=None,n_components=None,
                               store_covariance=False,tol=0.0001):
        """
        Factory method to create a LDA classifier node and add it to a graph
        
        Note that the node will have to be initialized (i.e. trained) prior 
        to execution of the kernel.
        """
        
        # create the kernel object            
        init_params = {'initialization_data' : initialization_data, 
                       'labels'        : labels,
                       'shrinkage'     : shrinkage,
                       'solver'        : solver,
                       'priors'        : priors,
                       'n_components'  : n_components,
                       'store_covariance' : store_covariance,
                       'tol'           : tol}
        k = cls(graph,X,y_bar,conf,pred_proba,BcipEnums.INIT_FROM_DATA,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(X,BcipEnums.INPUT),
                  Parameter(y_bar,BcipEnums.OUTPUT))
        
        if conf != None:
            params += (Parameter(conf,BcipEnums.OUTPUT),)
        
        if pred_proba != None:
            params += (Parameter(pred_proba,BcipEnums.OUTPUT),)
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
    
    @classmethod
    def add_trained_LDA_node(cls,graph,X,y_bar,model,
                               pred_proba=None,conf=None):
        """
        Factory method to create a pre-trained LDA classifier node
        
        The kernel will contain a reference to the model rather than making a 
        deep-copy. Therefore any changes to the classifier object outside
        will effect the classifier here.
        """

        # sanity check that the input is actually an MDM model
        if not isinstance(model,LinearDiscriminantAnalysis):
            return None
        
        # create the kernel object
        init_params = {'model' : model}
        k = cls(graph,X,y_bar,conf,pred_proba,BcipEnums.INIT_FROM_COPY,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(X,BcipEnums.INPUT),
                  Parameter(y_bar,BcipEnums.OUTPUT))
        
        if conf != None:
            params += (Parameter(conf,BcipEnums.OUTPUT),)
        
        if pred_proba != None:
            params += (Parameter(pred_proba,BcipEnums.OUTPUT),)
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
