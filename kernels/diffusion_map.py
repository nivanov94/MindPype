# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:19:56 2019

diff_map.py - Defines a diffusion mapping kernel for data visualization

@author: ivanovn
"""

from classes.tensor import Tensor
from classes.bcip_enums import BcipEnums
from classes.kernel import Kernel
from classes.node import Node
from classes.parameter import Parameter

import numpy as np

from pyriemann.utils import distance as riem_dist

class DiffusionMapKernel(Kernel):
    """
    Kernel produces projection of trial into lower dimensional space
    """
    
    def __init__(self,graph,inA,outA,n_components,init_style,\
                 initialize_params):
        """
        Create a Diffusion map spectral embedding kernel.
        The embedding is generated with manifold learning techniques
        using a set of trials as a training set.
        """
        super().__init__('DiffusionMap',init_style,graph)
        self.inA  = inA
        self.outA = outA
        self.n_components = n_components
        
        self._initialize_params = initialize_params
        
        if init_style == BcipEnums.INIT_FROM_DATA:
            # manifold will be approximated using data in tensor 
            # object at later time
            self._initialized = False
            self._embedding = None
            self._eigenvals = None
            self._training_pts = None
            self._eps = None
        
    
    def initialize(self):
        """
        Set the embedding of the kernel
        """
        
        if self.init_style == BcipEnums.INIT_FROM_DATA:
            return self.fitEmbedding()
        else:
            # kernel contains a reference to a pre-existing embedding object,
            # no need to fit here
            self._initialized = False
            return BcipEnums.NOT_SUPPORTED
    

    def fitEmbedding(self):
        """
        Fit the embedding
        
        The method will update the kernel's internal representation of the
        embedding
        """
        
        if (not isinstance(self._initialize_params['training_data'],Tensor)):
                return BcipEnums.INITIALIZATION_FAILURE
        
        X = self._initialize_params['training_data'].getData()
        
        # ensure the shpaes are valid
        if len(X.shape) != 3:
            return BcipEnums.INITIALIZATION_FAILURE
        
        if X.shape[1] != X.shape[2]:
            return BcipEnums.INITIALIZATION_FAILURE
        
        
        distmatrix = riem_dist.pairwise_distance(X)
        self._eps = np.median(distmatrix)**2 / 2
        
        kernel = np.exp(-distmatrix**2 / (4 * self._eps)) # 'heat' kernel represents similarities between trials
        D = np.diag(np.dot(kernel,np.ones(len(kernel)))) # degree matrix, sum of rows of kernel
        P = np.dot(np.inv(D),kernel) # markov chain for diffusion map
            
        # perform eigendecomposition to get mapping
        w, v = np.linalg.eig(P)

        # sort the eigenvalues
        idx = np.flip(np.argsort(w))
        w = w[idx]
        v = v[:,idx]
        
        # save the parameters of the embedding
        self._embedding = v[:,2:2+self._n_components]
        self._eigenvals = w[2:2+self._n_components]

        
        self._initialized = True
        
        return BcipEnums.SUCCESS
    

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (not isinstance(self.inA,Tensor)) or \
            (not isinstance(self.outA,Tensor)):
                return BcipEnums.INVALID_PARAMETERS
        
        input_shape = self.inA.shape
        input_rank = len(input_shape)
        
        # input tensor should not be greater than rank 3
        if input_rank > 3 or input_rank < 2:
            return BcipEnums.INVALID_PARAMETERS
        
        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if self.outA.isVirtual() and (self.outA.getShape() == ()):
            if input_rank == 2:
                self.outA.setShape((1,self.n_components.getData()))
                self.outA.setData(np.zeros((1,self.n_components.getData())))
            else:
                self.outA.setShape((1,self.n_components.getData()))
                self.outA.setData(np.zeros((input_shape[0],
                                            self.n_components.getData())))
        
        
        # check for dimensional alignment
        
        # check that the dimensions of the output match the dimensions of input
        if (input_rank == 2 and self.outA.shape[0] != 1) or \
            (input_rank == 3 and (self.inA.shape[0] != self.outA.shape[0])):
            return BcipEnums.INVALID_PARAMETERS
        
        return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel and project the input trials
        into the embedding using Nystrom's method
        """
        if not self._initialized:
            return BcipEnums.EXE_FAILURE_UNINITIALIZED
        

        n_training_pts = self._training_pts.shape[0]

        input_data = self.inA.getData()

        K_proj = np.zeros(len(input_data), n_training_pts)
        for i_trial in range(len(input_data)):
            # get the distances between the input and the embedding pts
            for i_tp in range(len(self._training_pts)):
                K_proj[i_trial,i_tp] = riem_dist.distance_riemann(input_data[i_trial,:,:],
                                                                  self._training_pts[i_tp,:,:])


        kernel = np.exp(-K_proj**2 / (4 * self._eps))
        
        D = np.diag(np.dot(kernel,np.ones(len(kernel)))) + 1 # degree matrix, sum of rows of kernel
        P = np.dot(np.inv(D),kernel) # markov chain for diffusion map of each trial

        # use to to project into the embedding
        pt_proj = np.dot(P,self._embedding) * (1/self._eigenvals)

        # set the output data
        self.outA.setData(pt_proj)
        
        return BcipEnums.SUCCESS
        
    
    @classmethod
    def addDiffusionMapKernel(cls,graph,inA,outA,
                              training_data,dimensions):
        """
        Factory method to create a diffusion map 
        visualization kernel and add it to a block
        as a generic node object.
        
        Note that the node will have to be initialized (i.e. trained) prior 
        to execution of the kernel.
        """
        
        # create the kernel object
        init_params = {'training_data' : training_data}
        k = cls(graph,inA,outA,dimensions,BcipEnums.INIT_FROM_DATA,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.addNode(node)
        
        return node
    
    

