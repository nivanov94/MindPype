"""
Created on Wed Apr  1 10:12:10 2020

"""


from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter
from classes.tensor import Tensor
from classes.array import Array
from classes.bcip_enums import BcipEnums

from .utils.data_extraction import extract_nested_data

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class FeatureNormalizationKernel(Kernel):
    """
    Normalizes the values within a feature vector
    """
    
    def __init__(self,graph,inA,outA,init_data,method):
        """
        Kernal normalizes features for classification
        """
        super().__init__('FeatureNormalization',BcipEnums.INIT_FROM_DATA,graph)
        self._inA  = inA
        self._outA = outA
        self._method = method
        self.initialization_data = init_data
        self._translate = 0
        self._scale = 1

        self.graph = graph

        self._labels = None
        
    
    def initialize(self):
        """
        Calculate the normalization parameters using the setup data
        """
        if isinstance(self.initialization_data,Tensor):
            X = self.initialization_data.data
        elif isinstance(self.initialization_data,Array):
            try:
                X = extract_nested_data(self.initialization_data)
            except:
                return BcipEnums.INITIALIZATION_FAILURE
        else:
            return BcipEnums.INVALID_NODE
        
        if self._method == 'min-max':
            self._translate = np.min(X,axis=0)
            self._scale = np.max(X,axis=0) - np.min(X,axis=0)
        
        elif self._method == 'mean-norm':
            self._translate = np.mean(X,axis=0)
            self._scale = np.max(X,axis=0) - np.min(X,axis=0)
        
        elif self._method == 'zscore-norm':
            self._translate = np.mean(X,axis=0)
            self._scale = np.std(X,axis=0)
        
        else:
            return BcipEnums.INVALID_NODE
        
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self._inA,Tensor)) or \
            (not isinstance(self._outA,Tensor)):
                return BcipEnums.INVALID_PARAMETERS
        
        if self._method not in ('min-max','mean-norm','zscore-norm'):
            return BcipEnums.INVALID_PARAMETERS
        
        if ((not isinstance(self.initialization_data,Tensor)) and 
            (not isinstance(self.initialization_data,Array))):
            return BcipEnums.INVALID_PARAMETERS
        
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = self._inA.shape
        
        # check output shape
        if self._outA.shape != self._inA.shape:
            return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
        if isinstance(self._inA,Tensor):
            X = self._inA.data
        else:
            try:
                X = extract_nested_data(self._inA)
            except:
                return BcipEnums.EXE_FAILURE
            
        try:
            self._outA.data = (X - self._translate) / self._scale
        except:
            return BcipEnums.EXE_FAILURE
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        Xs = self._outA.data
        markers = ('o','x','^')
        m, n = Xs.shape
        mc = m // 3
        for i in range(3):
            ax.scatter(Xs[i*mc:(i+1)*mc,0],Xs[i*mc:(i+1)*mc,1],Xs[i*mc:(i+1)*mc,2],marker=markers[i])
        plt.show()
        
        return BcipEnums.SUCCESS
    
    @classmethod
    def add_feature_normalization_node(cls,graph,inA,outA,
                                       init_data,method='zscore-norm'):
        """
        Factory method to create a feature normalization kernel
        """
        
        # create the kernel object
        k = cls(graph,inA,outA,init_data,method)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
