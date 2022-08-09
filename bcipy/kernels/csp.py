# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:03:27 2020

@author: Nick
"""

from types import NoneType
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.array import Array
from classes.bcip_enums import BcipEnums
from .utils.data_extraction import extract_nested_data

import numpy as np


class CommonSpatialPatternKernel(Kernel):
    """
    CSP Filter Kernel
    """
    
    def __init__(self,graph,inA,outA,
                 init_style,init_params,
                 num_filts):
        """
        Kernel applies a set of common spatial pattern filters to tensor of 
        covariance matrices
        """
        super().__init__('CSP',init_style,graph)
        self._inA = inA
        self._outA = outA     
        
        self._num_filts = num_filts
        self._init_params = init_params

        self._init_inA = init_params['initialization_data']
        self._init_outA = None
        
        self._initialization_data = self._init_params['initialization_data']



        self.graph = graph

        if init_style == BcipEnums.INIT_FROM_DATA:
            # model will be trained using data in tensor object at later time
            self._initialized = False
            self._W = None
            
        elif init_style == BcipEnums.INIT_FROM_COPY:
            # model is copy of predefined MDM model object
            self._W = init_params['filters']
            self._initialized = True
    

    def initialize(self):
        """
        Set the filter values
        """
        sts1, sts2 = BcipEnums.SUCCESS, BcipEnums.SUCCESS
        if self._initialization_data == None:
            self._initialization_data = self._init_inA
        
        if self.init_style == BcipEnums.INIT_FROM_DATA:
            sts1 = self.extract_filters()
            

        else:
            # kernel contains a reference to a pre-existing MDM object, no
            # need to train here
            self._initialized = True
        
        if self._init_outA.__class__ != NoneType:
            sts2 = self.initialization_execution()
        
        if sts1 != BcipEnums.SUCCESS:
            return sts1
        elif sts2 != BcipEnums.SUCCESS:
            return sts2
        else:
            return BcipEnums.SUCCESS

    
    def initialization_execution(self):

        if len(self._init_inA.shape) == 3:
            self._init_outA.shape = (self._init_inA.shape[0], self._W.shape[1], self._init_inA.shape[2])
            self._init_outA.data = np.zeros((self._init_inA.shape[0], self._W.shape[1], self._init_inA.shape[2]))
            print(f"{self._init_outA.shape}")
            #self._init_outA = Tensor.create_from_data( \
            #                                        self.session, 
            #                                        (self._init_inA.shape[0], self._init_inA.shape[1], self._W.shape[1]), 
            #                                        np.zeros((self._init_inA.shape[0], self._init_inA.shape[1], self._W.shape[1])))
            
        #try:   
        for i in range(self._init_inA.shape[0]):
            input_data = np.array(self._init_inA.data[i,:,:])

            if len(np.shape(input_data)) == 3:
                input_data = np.squeeze(input_data)

            output_data = np.matmul(self._W.T, input_data)
            self._init_outA.data[i, :, :] = output_data
        
        print(f"init_out shape: {self._init_outA.shape}")
        return BcipEnums.SUCCESS

        #except:
        #    return BcipEnums.INITIALIZATION_FAILURE
        

    def process_data(self, input_data, output_data):
        output_data.shape = (self._W.shape[1], input_data.shape[1])
        output_data.data = np.matmul(self._W.T, input_data.data) 

        return BcipEnums.SUCCESS


    def extract_filters(self):
        """
        Determine the filter values using the training data
        """

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
        if len(X.shape) == 2:
            X = X[np.newaxis, :, :]

        if len(y.shape) == 2:
            y = np.squeeze(y)

        if len(X.shape) != 3 or len(y.shape) != 1:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[0] != y.shape[0]:
            return BcipEnums.INITIALIZATION_FAILURE
        
        # y must contain 2, and only 2, unique labels
        labels = np.unique(y)
        if labels.shape[0] != 2:
            return BcipEnums.INITIALIZATION_FAILURE
        
        
        # start by calculating the covariance matrix for each class
        _ , Nc, Ns = X.shape
        C = np.zeros((2,Nc,Nc))
        for i in  range(len(labels)):
            l = labels[i]
            X_l = X[y==l,:,:]
            Nt = X_l.shape[0]
            X_l = np.transpose(X_l,(1,2,0))
            X_l = np.reshape(X_l,(Nc,Ns*Nt))
            C[i,:,:] = np.cov(X_l)
            
        
        # get the whitening matrix
        d, V = np.linalg.eig(np.mean(C,axis=0))
        
        # sort the eigenvalues in descending order
        ix = np.flip(np.argsort(d))
        d = d[ix]
        V = V[:,ix]
        
        # construct the whitening matrix
        M = np.matmul(V, np.diag(d ** (-1/2)))

        # calculate the CSP filters in the whitened space
        dC = C[0,:,:] - C[1,:,:]
        S = np.matmul(M.T,np.matmul(dC,M)) # M' * (C1 - C2) * M
        d, W = np.linalg.eig(S)
        W = np.matmul(M,W) # project filters back into channel space
        
        # sort the eigenvalues/vectors in descending over
        ix = np.flip(np.argsort(d))
        d = d[ix]
        W = W[:,ix]
        
        # extract the specified number of filters
        m = self._num_filts // 2
        f_ix = [_ for _ in range(m)] + [_ for _ in range(d.shape[0]-1,d.shape[0]-(m+1),-1)]
        self._W = W[:,f_ix]
        
        self._initialized = True
        
        return BcipEnums.SUCCESS
    
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """


        if self._initialization_data == None:
            self.graph._missing_data = True
        
        # first ensure the input and output are tensors
        if (not isinstance(self._inA,Tensor) or 
            not (isinstance(self._outA,Tensor))):
            return BcipEnums.INVALID_PARAMETERS
                
        
        # input tensor should be two- or three-dimensional
        if len(self._inA.shape) != 2 and len(self._inA.shape) != 3:
            return BcipEnums.INVALID_PARAMETERS
        

        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if len(self._inA.shape) == 2:
            out_sz = (self._inA.shape[0],self._num_filts)
        else:
            out_sz = self._inA.shape[0:2] + (self._num_filts,)
        
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = out_sz

        if self._outA.shape != out_sz:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel by classifying the input trials
        """
        #TODO: add
        return self.process_data(self._inA, self._outA)
    
    @classmethod
    def add_uninitialized_CSP_node(cls,graph,inA,outA,
                                   initialization_data,labels,
                                   num_filts):
        """
        Factory method to create a CSP filter node and add it to a graph
        
        Note that the node will have to be initialized prior 
        to execution of the kernel.
        """
        
        # create the kernel object            
        init_params = {'initialization_data' : initialization_data, 
                       'labels'        : labels}
        
        k = cls(graph,inA,outA,BcipEnums.INIT_FROM_DATA,init_params,num_filts)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
    
    @classmethod
    def add_initialized_CSP_node(cls,graph,inA,outA,filters):
        """
        Factory method to create a pre-initialized CSP filter node

        """
        
        # create the kernel object
        init_params = {'filters' : filters}
        k = cls(graph,inA,outA,BcipEnums.INIT_FROM_COPY,init_params,filters.shape[1])
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
