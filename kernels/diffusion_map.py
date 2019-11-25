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

from pyriemann.embedding import Embedding
from pyriemann.utils import distance

#from scipy.sparse.csgraph import laplacian as csgraph_laplacian
#from scipy.sparse.linalg import eigsh
#from sklearn.manifold import _set_diag
#
#def _spectral_embedding(adjacency, n_components):
#    """
#    Calculate the spectral embedding
#    """
#    
#    laplacian, dd = csgraph_laplacian(adjacency,normed=True,
#                                      return_diag=True)
#    
#    laplacian = _set_diag(laplacian,1,True)
#    laplacian *= -1
#    v0 = random_state.uniform(-1,1,laplacian.shape[0])
#    lambdas, diff_map = eigsh(laplacian,k=n_components,sigma=1.0,
#                              which='LM',tol=0.0,v0=v0)
#    embedding = diff_map.T[n_components::-1]
#    embedding = embedding / dd
#    
#    return embedding[1:n_components].T, lambdas


class DiffusionMapKernel(Kernel):
    """
    Kernel produces projection of trial into lower dimensional space
    """
    
    def __init__(self,block,inA,outA,n_components,init_style,\
                 initialize_params):
        """
        Create a Diffusion map spectral embedding kernel.
        The embedding is generated with manifold learning techniques
        using a set of trials as a training set.
        """
        super().__init__('DiffusionMap',init_style,block)
        self.inA  = inA
        self.outA = outA
        self.n_components = n_components
        
        self._initialize_params = initialize_params
        
        if init_style == BcipEnums.INIT_FROM_DATA:
            # manifold will be approximated using data in tensor 
            # object at later time
            self._initialized = False
            self._embedding = None
            self._eps = None
        elif init_style == BcipEnums.INIT_FROM_COPY:
            # model is copy of predefined manifold embedding object
            self._embedding = initialize_params['embedding']
            self._eps = initialize_params['eps']
            self._initialized = True
        
    
    def initialize(self):
        """
        Set the embedding of the kernel
        """
        
        if self.init_style == BcipEnums.INIT_FROM_DATA:
            return self.fitEmbedding()
        else:
            # kernel contains a reference to a pre-existing embedding object,
            # no need to fit here
            self._initialized = True
            return BcipEnums.SUCCESS
        
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
        
        # create an embedding object from pyriemann
        embdg = Embedding(self.n_components.getData())
        

        
        # get the components of the embedded pts
        self._embedding = embdg.fit_transform(X)
        
        if self._eps == None:
            # save the value of eps used to generate the embedding
            # TODO make this cleaner, consider writing modified version of
            # embedding class to make eps accessible
            distmatrix = distance.pairwise_distance(X)
            self._eps = np.median(distmatrix)**2 / 2
            
            W = np.exp(-distmatrix**2 / (4 * self._eps))
            q = np.dot(W,np.ones(len(W)))
            W = np.divide(W,np.outer(q,q))
            
            D = np.linalg.inv(np.diag(np.sum(W,axis=1)))
            P = np.dot(D,W)
            
            
            # numpy calc
            w, v = np.linalg.eig(P)
            
        
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
        
        
        
    
    @classmethod
    def addDiffusionMapKernel(cls,block,inA,outA,
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
        k = cls(block,inA,outA,dimensions,BcipEnums.INIT_FROM_DATA,init_params)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT), \
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(block,k,2,params)
        
        # add the node to the block
        block.addNode(node)
        
        return node
    
    
#    @classmethod
#    def addTrainedRiemannMDMKernel(cls,block,inputA,outputA,\
#                                     model):
#        """
#        Factory method to create a riemann minimum distance 
#        to the mean classifier kernel containing a copy of a pre-trained
#        MDM classifier and add it to a block as a generic node object.
#        
#        The kernel will contain a reference to the model rather than making a 
#        deep-copy. Therefore any changes to the classifier object outside
#        will effect the classifier here.
#        """
#
#        # sanity check that the input is actually an MDM model
#        if not isinstance(model,classification.MDM):
#            return None
#        
#        # create the kernel object
#        init_params = {'model' : model}
#        k = cls(block,inputA,outputA,BcipEnums.INIT_FROM_COPY,init_params)
#        
#        # create parameter objects for the input and output
#        params = (Parameter(inputA,BcipEnums.INPUT), \
#                  Parameter(outputA,BcipEnums.OUTPUT))
#        
#        # add the kernel to a generic node object
#        node = Node(block,k,2,params)
#        
#        # add the node to the block
#        block.addNode(node)
#        
#        return node