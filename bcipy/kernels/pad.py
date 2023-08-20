from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

import numpy as np

class PadKernel(Kernel):
    """
    Kernel to conduct padding on data

    Parameters
    ----------
    graph : Graph 
        Graph that the kernel should be added to
    
    inA : Tensor
        Input trial data (n_channels, n_samples) or (n_trials, n_channels, n_samples)

    outA : Tensor
        Output trial data (n_channels, n_samples) or (n_trials, n_channels, n_samples)
    
    """

    def __init__(self, graph, inA, outA, pad_width = None, mode = 'constant', stat_length = None, constant_values = 0, end_values = 0, reflect_type = 'even', **kwargs):
        super().__init__('BaselineCorrection', BcipEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA]
        self.outputs = [outA]
        
        self._pad_width = pad_width
        self._mode = mode
        self._stat_length = stat_length
        self._constant_values = constant_values
        self._end_values = end_values
        self._reflect_type = reflect_type

        self._kwargs_dict = kwargs
        
    def _initialize(self, init_inputs, init_outputs, labels):

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # TODO need to correct padding dimensions for potentially differently shaped init inputs
            self._process_data(init_inputs, init_outputs)

    def _process_data(self, inputs, outputs):
        """
        Process the data
        """
        inp = inputs[0]
        # TODO: make this more efficient/reduce code duplication
        if self._mode in ('maximum', 'mean', 'median', 'minimum'):
            out_data = np.pad(inp.data, self._pad_width, self._mode, stat_length = self._stat_length)
        elif self._mode in ('constant'):
            out_data = np.pad(inp.data, self._pad_width, self._mode, constant_values = self._constant_values)
        elif self._mode in ('linear_ramp'):
            out_data = np.pad(inp.data, self._pad_width, self._mode, end_values = self._end_values)
        elif self._mode in ('reflect', 'symmetric'):
            out_data = np.pad(inp.data, self._pad_width, self._mode, reflect_type = self._reflect_type)
        elif self._mode in ('wrap', 'empty', 'edge'):
            out_data = np.pad(inp.data, self._pad_width, self._mode)
        else:
            out_data = np.pad(inp.data, self._pad_width, self._mode, **self._kwargs_dict)

        outputs[0].data = out_data

        return BcipEnums.SUCCESS

    @classmethod
    def add_pad_kernel(cls, graph, inA, outA, pad_width = None, mode = 'constant', stat_length = None, constant_values = 0, end_values = 0, reflect_type = 'even', **kwargs):
        """
        Add a pad kernel to the graph
        """

        k = cls(graph, inA, outA, pad_width, 
                mode, stat_length, constant_values, 
                end_values, reflect_type, **kwargs)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
