from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Scalar

from scipy import signal
import numpy as np
import warnings

class Filter:
    def initialize(self):
        """
        This kernel has no internal state that must be initialized
        """
        sts = BcipEnums.SUCCESS
        
        if self._init_outA != None:
            
            if self._init_outA.virtual:
                self._init_outA.shape = self._init_inA.shape
            
            sts = self._process_data(self._init_inA, self._init_outA)

            # pass on the labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR:
                input_labels = self._init_labels_in.to_tensor()
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out)
        
        return sts
    
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # first ensure the input and output are tensors
        if (self._inA._bcip_type != BcipEnums.TENSOR or
            self._outA._bcip_type != BcipEnums.TENSOR or
            self._filt._bcip_type != BcipEnums.FILTER):
            return BcipEnums.INVALID_PARAMETERS
        
        # do not support filtering directly with zpk filter repesentation
        if self._filt.implementation == 'zpk':
            return BcipEnums.NOT_SUPPORTED
        
        # check the shape
        input_shape = self._inA.shape
        input_rank = len(input_shape)

        if self._axis < 0 or self._axis >= input_rank:
            return BcipEnums.INVALID_PARAMETERS
        
        # determine what the output shape should be
        if input_rank == 0:
            return BcipEnums.INVALID_PARAMETERS
        else:
            output_shape = input_shape
        
        # if the output is virtual and has no defined shape, set the shape now
        if self._outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = output_shape
        
        # ensure the output tensor's shape equals the expected output shape
        if self._outA.shape != output_shape:
            warnings.warn(f"The output tensor's shape for the {self._filt.ftype} filter does not match the expected output shape", RuntimeWarning, stacklevel=15)
            return BcipEnums.INVALID_PARAMETERS
        else:
            return BcipEnums.SUCCESS
        
    def execute(self):
        """
        Execute the kernel function using the scipy module function
        """
        return self._process_data(self._inA, self._outA)
 

class FilterKernel(Filter, Kernel):
    """
    Filter a tensor along the first non-singleton dimension

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inputA : Tensor or Scalar 
        Input trial data

    filt : Filter 
        BCIP Filter object outputted by bcipy.classes

    outputA : Tensor or Scalar 
        Output trial data

    axis : int
        axis along which to apply the filter
    """
    
    def __init__(self,graph,inputA,filt,outputA,axis):
        super().__init__('Filter',BcipEnums.INIT_FROM_COPY,graph)
        self._inA  = inputA
        self._filt = filt
        self._outA = outputA
        
        self._init_inA = None
        self._init_outA = None  

        self._init_labels_in = None
        self._init_labels_out = None
        self._axis = axis
    
    def _process_data(self, input_data, output_data):
        try:
            if self._filt.implementation == 'ba':
                output_data.data = signal.lfilter(self._filt.coeffs['b'],
                                                self._filt.coeffs['a'],
                                                input_data.data, 
                                                axis=self._axis)
            else:
                output_data.data = signal.sosfilt(self._filt.coeffs['sos'],
                                                input_data.data,
                                                axis=self._axis)
            return BcipEnums.SUCCESS

        except:
            return BcipEnums.EXE_FAILURE



    @classmethod
    def add_filter_node(cls,graph,inputA,filt,outputA,axis=1):
        """
        Factory method to create a filter kernel and add it to a graph
        as a generic node object.

        graph : Graph 
            Graph that the node should be added to

        inputA : Tensor or Scalar 
            Input trial data

        filt : Filter 
            BCIP Filter object outputted by bcipy.classes

        outputA : Tensor or Scalar 
            Output trial data

        axis : int
            Axis along which to apply the filter
        """
        
        # create the kernel object
        k = cls(graph,inputA,filt,outputA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT),
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

class FiltFiltKernel(Filter, Kernel):
    """
    Zero phase filter a tensor along the first non-singleton dimension

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to

    inputA : Tensor or Scalar 
        Input trial data

    filt : Filter 
        BCIP Filter object outputted by bcipy.classes

    outputA : Tensor or Scalar 
        Output trial data

    axis : int
        axis along which to apply the filter
    """
    
    def __init__(self,graph,inputA,filt,outputA,axis):
        super().__init__('FiltFilt',BcipEnums.INIT_FROM_COPY,graph)
        self._inA  = inputA
        self._filt = filt
        self._outA = outputA

        self._init_inA = None
        self._init_outA = None

        self._axis = axis
        
        self._init_labels_in = None
        self._init_labels_out = None
 
    def _process_data(self, input_data, output_data):

        try:
            if self._filt.implementation == 'ba':
                output_data.data = signal.filtfilt(self._filt.coeffs['b'],
                                                    self._filt.coeffs['a'],
                                                    input_data.data,
                                                    axis=self._axis)
            else:
                output_data.data = signal.sosfiltfilt(self._filt.coeffs['sos'],
                                                       input_data.data,
                                                       axis=self._axis)
        except Exception as e:
            warnings.warn(f"{e}", RuntimeWarning, stacklevel=5)
            return BcipEnums.EXE_FAILURE

        return BcipEnums.SUCCESS


    @classmethod
    def add_filtfilt_node(cls,graph,inputA,filt,outputA,axis=1):
        """
        Factory method to create a filtfilt kernel and add it to a graph
        as a generic node object.

        graph : Graph 
            Graph that the node should be added to

        inputA : Tensor or Scalar 
            Input trial data

        filt : Filter 
            BCIP Filter object outputted by bcipy.filters

        outputA : Tensor or Scalar 
            Output trial data

        axis : int
            axis along which to apply the filter
        """
        
        # create the kernel object
        k = cls(graph,inputA,filt,outputA,axis)
        
        # create parameter objects for the input and output
        params = (Parameter(inputA,BcipEnums.INPUT),
                  Parameter(outputA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node

