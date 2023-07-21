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
        self.inputs = [inA]
        self.outputs = [outA]
        
        self._pad_width = pad_width
        self._mode = mode
        self._stat_length = stat_length
        self._constant_values = constant_values
        self._end_values = end_values
        self._reflect_type = reflect_type

        self._kwargs_dict = kwargs
        
    def verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """
        
        inA = self.inputs[0]
        outA = self.outputs[0]

        # inA and outA must be a tensor
        for param in (inA, outA):
            if param.bcip_type != BcipEnums.TENSOR:
                return BcipEnums.INVALID_PARAMETERS
    
        
        # check the output dimensions are valid
        test_output = Tensor.create_virtual(self.session)
        self._process_data(inA, test_output)

        if outA.virtual and len(self._outA.shape) == 0:
            self._outA.shape = test_output.shape

        if outA.shape != test_output.shape:
            return BcipEnums.INVALID_PARAMETERS

        return BcipEnums.SUCCESS


    def initialize(self):
        sts = BcipEnums.SUCCESS

        init_in = self.init_inputs[0]
        init_out = self.init_outputs[0]
        labels = self.init_input_labels

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            # TODO need to correct padding dimensions for potentially differently shaped init inputs
            sts = self._process_data(init_in, init_out)

            # pass on labels
            self.copy_init_labels_to_output()
        
        return sts

    def execute(self):
        """
        Execute the kernel
        """
        return self._process_data(self.inputs[0], self.outputs[0])
    

    def _process_data(self, inp, out):
        """
        Process the data
        """

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
