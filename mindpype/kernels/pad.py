from ..core import MPEnums
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
        """
        Constructor for the PadKernel class
        """
        super().__init__('BaselineCorrection', MPEnums.INIT_FROM_NONE, graph)
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
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.tensor()

            params_adjusted = False

            # check if there's an additional dimension for init
            if len(init_in.shape) != len(self.inputs[0].shape):
                params_adjusted = True
                orig_pad_width = self._pad_width
                
                # Modify the pad width to account for the additional dimension
                if isinstance(self._pad_width, int):
                    self._pad_width = ((0,0),) + ((self._pad_width, self._pad_width),) * len(self.inputs[0].shape)
                
                elif isinstance(self._pad_width[0], int):
                    self._pad_width = ((0,0),) + ((self._pad_width[0],self._pad_width[1]),) * len(self.inputs[0].shape)
                    
                else:
                    self._pad_width = ((0,0),) + tuple([tuple(pad) for pad in self._pad_width])

                
            self._process_data([init_in], init_out)
            
            if params_adjusted:
                self._pad_width = orig_pad_width
            

        
    def _process_data(self, inputs, outputs):
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

    @classmethod
    def add_pad_node(cls, graph, inA, outA, pad_width = None, mode = 'constant', 
                     stat_length = None, constant_values = 0, end_values = 0, 
                     reflect_type = 'even', init_input=None, init_labels=None, **kwargs):
        """
        Add a pad kernel to the graph

        Parameters
        ----------

        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor
            Input trial data (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        outA : Tensor
            Output trial data (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        pad_width : int or sequence of ints, optional
            Number of values padded to the edges of each axis. ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis. ((before, after),) yields same before and after pad for each axis. (pad,) or int is a shortcut for before = after = pad width for all axes. Default is None, in which case no padding is added.
        mode : str or function, optional
            One of the following string values or a user supplied function.
            'constant' (default)
                Pads with a constant value.
            'edge'
                Pads with the edge values of array.
            'linear_ramp'
                Pads with the linear ramp between end_value and the array edge value.
            'maximum'
                Pads with the maximum value of all or part of the vector along each axis.
            'mean'
                Pads with the mean value of all or part of the vector along each axis.
            'median'
                Pads with the median value of all or part of the vector along each axis.
            'minimum'
                Pads with the minimum value of all or part of the vector along each axis.                                                                   
            'reflect'
                Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
            'symmetric'
                Pads with the reflection of the vector mirrored along the edge of the array.                            
            'wrap'                                                                              
                Pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.                                         
            'empty'
                Pads with undefined values.
            <function>                              
                Padding function, see Notes in numpy.pad documentation, linked `here <https://numpy.org/doc/stable/reference/generated/numpy.pad.html#numpy.pad>` _ .           
        stat_length : sequence or int, optional
            Number of values at edge of each axis except the concatenation axis from which the median/mean is calculated. Default is None.
        constant_values : sequence or int, optional
            Used in 'constant'. The values to set the padded values for each axis. Default is 0.
        end_values : sequence or int, optional
            Used in 'linear_ramp'. The values used for the ending value of the linear_ramp and that will form the edge of the padded array. Default is 0.
        reflect_type : str, optional
            Used in 'reflect' and 'symmetric'. The 'reflect' type is the default which reflects the values at the edge of the array. The 'symmetric' type extends the array in both directions with the reflection of the array on the nearest edge. Default is 'even'.
        kwargs : dict, optional
            Keyword arguments for other modes. See Notes linked above.
        
        Returns
        -------
        node : Node
            Node that was added to the graph containing the kernel and parameters            
        """

        k = cls(graph, inA, outA, pad_width, 
                mode, stat_length, constant_values, 
                end_values, reflect_type, **kwargs)
        
        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))
        
        # add the kernel to a generic node object
        node = Node(graph,k,params)
        
        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_input is not None:
            node.add_initialization_data([init_input], init_labels)
        
        return node
