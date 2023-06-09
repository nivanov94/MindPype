from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_nested_data

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class FeatureNormalizationKernel(Kernel):
    """
    Normalizes the values within a feature vector

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inA : Tensor 
        Input trial data

    outA : Tensor 
        Extracted trial data

    init_data : Tensor 
        Initialization data

    method : {'min-max', 'mean-norm', 'zscore-norm'}
        Feature normalization method
    """
    
    def __init__(self,graph,inA,outA,init_data,method,axis):
        """
        Kernal normalizes features for classification
        """
        super().__init__('FeatureNormalization',BcipEnums.INIT_FROM_DATA,graph)
        self._inA  = inA
        self._outA = outA
        self._method = method
        self._axis = axis
        self._translate = 0
        self._scale = 1
        

        self._init_inA = init_data
        self._init_labels_in = None
        self._init_outA = None
        self._init_labels_out = None


    def initialize(self):
        """
        Calculate the normalization parameters using the setup data
        """

        sts = BcipEnums.SUCCESS

        if self._init_inA._bcip_type == BcipEnums.TENSOR:
            X = self.initialization_data.data
        elif self._init_inA._bcip_type == BcipEnums.ARRAY:
            try:
                X = extract_nested_data(self._init_inA)
            except:
                return BcipEnums.INITIALIZATION_FAILURE
        else:
            return BcipEnums.INVALID_NODE


        
        if self._method == 'min-max':
            self._translate = np.min(X,axis=self._axis)
            self._scale = np.max(X,axis=self._axis) - np.min(X,axis=self._axis)
        
        elif self._method == 'mean-norm':
            self._translate = np.mean(X,axis=self._axis) #changed from init_axis, confirm it should be self._axis 
            self._scale = np.max(X,axis=self._axis) - np.min(X,axis=self._axis)
        
        elif self._method == 'zscore-norm':
            self._translate = np.mean(X,axis=self._axis)
            self._scale = np.std(X,axis=self._axis)
        
        else:
            return BcipEnums.INVALID_NODE


        # process initialization data
        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            # adjust the shape of init output tensor, as needed
            if self._init_outA.virtual:
                self._init_outA.shape = self._init_inA.shape

            sts = self._process_data(self._init_inA, self._init_outA)

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out)

        if sts == BcipEnums.SUCCESS:
            self._initialized = True

        
        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        # first ensure the input and output are tensors
        if (self._inA._bcip_type != BcipEnums.TENSOR or
            self._outA._bcip_type != BcipEnums.TENSOR):
                return BcipEnums.INVALID_PARAMETERS
        
        if self._method not in ('min-max','mean-norm','zscore-norm'):
            return BcipEnums.INVALID_PARAMETERS
        
        Nd = self._inA.shape
        if (-(Nd+1) > abs(self._axis) or
            (Nd+1) <= abs(self._axis)):
            return BcipEnums.INVALID_PARAMETERS

        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (self._outA.virtual and len(self._outA.shape) == 0):
            self._outA.shape = self._inA.shape
        
        # check output shape
        if self._outA.shape != self._inA.shape:
            return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS

    def _process_data(self, inA, outA):
        try:
            outA.data = (inA - self._translate) / self._scale
            return BcipEnums.SUCCESS
        except:
            return BcipEnums.EXE_FAILURE

    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
            
        return self._process_data(self._inA, self._outA)
    
    @classmethod
    def add_feature_normalization_node(cls,graph,inA,outA,
                                       init_data,axis=0,method='zscore-norm'):
        """
        Factory method to create a feature normalization kernel
        """

        # create the kernel object
        k = cls(graph,inA,outA,init_data,method,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
