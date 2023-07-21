from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_nested_data

import numpy as np

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
    
    def __init__(self,graph,inA,outA,init_data,labels,method,axis):
        """
        Kernal normalizes features for classification
        """
        super().__init__('FeatureNormalization',BcipEnums.INIT_FROM_DATA,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._method = method
        self._axis = axis
        self._translate = 0
        self._scale = 1

        self.initialized = False


    def initialize(self):
        """
        Calculate the normalization parameters using the setup data
        """

        sts = BcipEnums.SUCCESS
        self.initialized = False

        # get the initialization input
        init_in = self.init_inputs[0]

        if init_in.bcip_type == BcipEnums.TENSOR:
            X = init_in.data
        elif init_in.bcip_type in (BcipEnums.ARRAY, BcipEnums.CIRCULAR_BUFFER):
            try:
                X = extract_nested_data(init_in)
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

        self.initialized = True

        # process initialization data
        init_out = self.init_outputs[0]
        if sts == BcipEnums.SUCCESS and init_out is not None:
            # adjust the shape of init output tensor, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

            sts = self._process_data(init_in, init_out)

            # pass on the labels
            self.copy_init_labels_to_output()

        return BcipEnums.SUCCESS
        
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        inA = self.inputs[0]
        outA = self.outputs[0]

        # first ensure the input and output are tensors
        if (inA.bcip_type != BcipEnums.TENSOR or
            outA.bcip_type != BcipEnums.TENSOR):
                return BcipEnums.INVALID_PARAMETERS
        
        if self._method not in ('min-max','mean-norm','zscore-norm'):
            return BcipEnums.INVALID_PARAMETERS
        
        Nd = inA.shape
        if (-(Nd+1) > abs(self._axis) or
            (Nd+1) <= abs(self._axis)):
            return BcipEnums.INVALID_PARAMETERS

        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (outA.virtual and len(outA.shape) == 0):
            outA.shape = inA.shape
        
        # check output shape
        if outA.shape != inA.shape:
            return BcipEnums.INVALID_PARAMETERS
  
        return BcipEnums.SUCCESS

    def _process_data(self, inA, outA):
        try:
            outA.data = (inA.data - self._translate) / self._scale
            return BcipEnums.SUCCESS
        except:
            return BcipEnums.EXE_FAILURE

    def execute(self):
        """
        Execute the kernel and calculate the mean
        """
            
        return self._process_data(self.inputs[0], self.outputs[0])
    
    @classmethod
    def add_feature_normalization_node(cls,graph,inA,outA,
                                       init_data=None,labels=None,axis=0,method='zscore-norm'):
        """
        Factory method to create a feature normalization kernel
        """

        # create the kernel object
        k = cls(graph,inA,outA,init_data, labels,method,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
