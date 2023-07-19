from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor

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
        self._inA = inA
        self._outA = outA
        self._pad_width = pad_width
        self._mode = mode
        self._stat_length = stat_length
        self._constant_values = constant_values
        self._end_values = end_values
        self._reflect_type = reflect_type

        self._kwargs_dict = kwargs
        
        self._init_inA = None
        self._init_outA = None

        self._init_labels_in = None
        self._init_labels_out = None

    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        # inA, inB, and outA must be a tensor
        for param in (self._inA, self._outA):
            if param._bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS
    
        
        # check the input dimensions are valid
        if  not 0 < len(self._inA.shape) < 4:
            return BcipEnums.INVALID_PARAMETERS
        
        # check the output dimensions are valid
        if self._outA.virtual and len(self._outA.shape) == 0:
            test_output = Tensor.create_virtual(self.session)
            self._process_data(self._inA, test_output)
            self._outA.shape = test_output.shape

        return BcipEnums.SUCCESS


    def initialize(self):
        sts = BcipEnums.SUCCESS

        if self._init_outA is not None and (self._init_inA is not None and self._init_inA.shape != ()):

            sts = self._process_data(self._init_inA, self._init_outA)

            # pass on labels
            if self._init_labels_in._bcip_type != BcipEnums.TENSOR: # type: ignore
                input_labels = self._init_labels_in.to_tensor() # type: ignore
            else:
                input_labels = self._init_labels_in
            input_labels.copy_to(self._init_labels_out) # type: ignore
        
        return sts

    def execute(self):
        """
        Execute the kernel
        """
        return self._process_data(self._inA, self._outA)
    

    def _process_data(self, inp, out):
        """
        Process the data
        """

        # TODO: make this more efficient/reduce code duplication
        if self._mode in ('maximum', 'mean', 'median', 'minimum'):
            out_data = np.pad(inp.data, self._pad_width, self._mode, stat_length = self._stat_length) # type: ignore
            out.shape = out_data.shape
            out.data = out_data

        elif self._mode in ('constant'):
            out_data = np.pad(inp.data, self._pad_width, self._mode, constant_values = self._constant_values) # type: ignore
            out.shape = out_data.shape
            out.data = out_data

        elif self._mode in ('linear_ramp'):
            out_data = np.pad(inp.data, self._pad_width, self._mode, end_values = self._end_values) # type: ignore
            out.shape = out_data.shape
            out.data = out_data

        elif self._mode in ('reflect', 'symmetric'):
            out_data = np.pad(inp.data, self._pad_width, self._mode, reflect_type = self._reflect_type) # type: ignore
            out.shape = out_data.shape
            out.data = out_data


        elif self._mode in ('wrap', 'empty', 'edge'):
            out_data = np.pad(inp.data, self._pad_width, self._mode) # type: ignore
            out.shape = out_data.shape
            out.data = out_data
        else:
            out_data = np.pad(inp.data, self._pad_width, self._mode, **self._kwargs_dict) # type: ignore
            out.shape = out_data.shape
            out.data = out_data

        return BcipEnums.SUCCESS

    @classmethod
    def add_pad_kernel(cls, graph, inA, outA, pad_width = None, mode = 'constant', stat_length = None, constant_values = 0, end_values = 0, reflect_type = 'even', **kwargs):
        """
        Add a pad kernel to the graph
        """

        k = cls(graph, inA, outA, pad_width, mode, stat_length, constant_values, end_values, reflect_type, **kwargs)
        # create parameter objects for the input and output
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)
        
        return node
