from ..core import BCIP, BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

from scipy import signal
import numpy as np
import warnings

class Filter:
    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]
        
        if init_out is not None and (init_in is not None and init_in.shape != ()):
            
            # adjust init output shape if virtual
            if init_out.virtual:
                init_out.shape = init_in.shape
            
            axis_adjusted = False
            if len(init_in.shape) == len(self.inputs[0].shape)+1 and self._axis >= 0:
                self._axis += 1
                axis_adjusted = True

            self._process_data(init_inputs, init_outputs)

            if axis_adjusted:
                self._axis -= 1

    
    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        inA = self.inputs[0]
        outA = self.outputs[0]

        # first ensure the input and output are tensors
        if (inA.bcip_type != BcipEnums.TENSOR or
            outA.bcip_type != BcipEnums.TENSOR):
            raise TypeError('Filter kernel requires Tensor inputs and outputs')
        
        if self._filt.bcip_type != BcipEnums.FILTER:
            raise TypeError('Filter kernel requires Filter object')

        # do not support filtering directly with zpk filter repesentation
        if self._filt.implementation == 'zpk':
            raise ValueError('Filter kernel: zpk filter representation not supported')
        
        # check the shape
        input_shape = inA.shape
        input_rank = len(input_shape)

        if self._axis < -input_rank or self._axis >= input_rank:
            warnings.warn(f"The axis parameter for the {self._filt.ftype} filter is out of range", RuntimeWarning, stacklevel=15)
            raise ValueError('Filter kernel: axis must be within rank of input tensor')
               
        # determine what the output shape should be
        if input_rank == 0:
            warnings.warn(f"The input tensor for the {self._filt.ftype} filter has no dimensions", RuntimeWarning, stacklevel=15)
            raise ValueError('Filter kernel: input tensor must have at least one dimension')
        else:
            output_shape = input_shape
        
        # if the output is virtual and has no defined shape, set the shape now
        if outA.virtual and len(outA.shape) == 0:
            outA.shape = output_shape
        
        # ensure the output tensor's shape equals the expected output shape
        if outA.shape != output_shape:
            warnings.warn(f"The output tensor's shape for the {self._filt.ftype} filter does not match the expected output shape", RuntimeWarning, stacklevel=15)
            raise ValueError('Filter kernel: output shape must match input shape')
        

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
    
    def __init__(self,graph,inA,filt,outA,axis):
        super().__init__('Filter',BcipEnums.INIT_FROM_COPY,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._filt = filt
        
        self._axis = axis
    
    def _process_data(self, inputs, outputs):
        if self._filt.implementation == 'ba':
            outputs[0].data = signal.lfilter(self._filt.coeffs['b'],
                                            self._filt.coeffs['a'],
                                            inputs[0].data, 
                                            axis=self._axis)
        elif self._filt.implementation == 'sos':
            outputs[0].data = signal.sosfilt(self._filt.coeffs['sos'],
                                            inputs[0].data,
                                            axis=self._axis)

        elif self._filt.implementation == 'fir':
            outputs[0].data = np.apply_along_axis(lambda x: signal.convolve(x, self._filt.coeffs['fir'], mode='same'),
                                                  arr=inputs[0].data, axis=self._axis)


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
    
    def __init__(self,graph,inA,filt,outA,axis):
        super().__init__('FiltFilt',BcipEnums.INIT_FROM_COPY,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._filt = filt

        self._axis = axis
        
    def _process_data(self, inputs, outputs):
        
        if self._filt.implementation == 'ba':
            outputs[0].data = signal.filtfilt(self._filt.coeffs['b'],
                                                self._filt.coeffs['a'],
                                                inputs[0].data,
                                                axis=self._axis)
        elif self._filt.implementation == 'sos':
            outputs[0].data = signal.sosfiltfilt(self._filt.coeffs['sos'],
                                                   inputs[0].data,
                                                   axis=self._axis)
        elif self._filt.implementation == 'fir':
            outputs[0].data = signal.lfilter(self._filt.coeffs['fir'], 
                                             1.0, inputs[0].data,
                                            axis=self._axis)
        

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

