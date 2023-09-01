from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_init_inputs

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
    
    initialization_data : Tensor
        Initialization data to train the classifier (n_trials, n_channels, n_samples)
    
    labels : Tensor
        Labels corresponding to initialization data class labels (n_trials, )

    method : {'min-max', 'mean-norm', 'zscore-norm'}
        Feature normalization method

    axis : int, default = 1
        Axis along which to apply the filter
    """
    
    def __init__(self,graph,inA,outA, initialization_data, labels ,method,axis=1):
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

        if initialization_data is not None:
            self.init_inputs = [initialization_data]

        if labels is not None:
            self.init_input_labels = labels


    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Calculate the normalization parameters using the setup data
        """
        # get the initialization input
        init_in = init_inputs[0]
        X = extract_init_inputs(init_in)
        
        if self._method == 'min-max':
            self._translate = np.min(X,axis=self._axis)
            self._scale = np.max(X,axis=self._axis) - np.min(X,axis=self._axis)
        
        elif self._method == 'mean-norm':
            self._translate = np.mean(X,axis=self._axis) #changed from init_axis, confirm it should be self._axis 
            self._scale = np.max(X,axis=self._axis) - np.min(X,axis=self._axis)
        
        elif self._method == 'zscore-norm':
            self._translate = np.mean(X,axis=self._axis)
            self._scale = np.std(X,axis=self._axis)
        
        # process initialization data
        init_out = init_outputs[0]
        if init_out is not None:
            # adjust the shape of init output tensor, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

            self._process_data(init_inputs, init_outputs)

    
    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """
        
        inA = self.inputs[0]
        outA = self.outputs[0]

        # first ensure the input and output are tensors
        if (inA.bcip_type != BcipEnums.TENSOR or
            outA.bcip_type != BcipEnums.TENSOR):
                raise TypeError('FeatureNormalization kernel requires Tensor inputs and outputs')
        
        if self._method not in ('min-max','mean-norm','zscore-norm'):
            raise ValueError('FeatureNormalization kernel: Invalid method: {}'.format(self._method))
        
        Nd = inA.shape
        if (self._axis < -len(Nd) or self._axis >= len(Nd)):
            raise ValueError('FeatureNormalization kernel: axis must be within rank of input tensor')

        # if the output is a virtual tensor and dimensionless, 
        # add the dimensions now
        if (outA.virtual and len(outA.shape) == 0):
            outA.shape = inA.shape
        
        # check output shape
        if outA.shape != inA.shape:
            raise ValueError('FeatureNormalization kernel: output shape must match input shape')
  
    def _process_data(self, inputs, outputs):
        outputs[0].data = (inputs[0].data - self._translate) / self._scale

    @classmethod
    def add_feature_normalization_node(cls,graph,inA,outA,
                                       init_data=None,labels=None,method='zscore-norm', axis=1):
        """
        Factory method to create a feature normalization kernel

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor
            Input trial data

        outA : Tensor
            Extracted trial data

        init_data : Tensor, default = None
            Initialization data

        labels : Tensor, default = None
            Initialization labels

        method : {'min-max', 'mean-norm', 'zscore-norm'}
            Feature normalization method

        axis : int, default = 1
            Axis along which to apply the filter

        Returns
        -------
        node : Node
            Node object that contains the kernel
        """

        # create the kernel object
        k = cls(graph,inA,outA,method,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
    
