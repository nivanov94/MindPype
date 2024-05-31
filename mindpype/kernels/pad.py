from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

import numpy as np

class PadKernel(Kernel):
    """
    Kernel to conduct padding on data

    .. note::
        This kernel utilizes the numpy function
        :func:`pad <numpy:numpy.pad>`.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor
        Input data (n_channels, n_samples) or (n_trials, n_channels, n_samples)

    outA : Tensor
        Output data (n_channels, n_samples) or (n_trials, n_channels, n_samples)

    """

    def __init__(self, graph, inA, outA, pad_width = None, mode = 'constant', stat_length = None, constant_values = 0, end_values = 0, reflect_type = 'even', **kwargs):
        """ Init """
        super().__init__('Pad', MPEnums.INIT_FROM_NONE, graph)
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
        """
        Parameters
        ----------

        init_inputs: Tensor
            Input data

        init_outputs: Tensor
            Output data

        labels : Tensor 
            Class labels for initialization data
        """
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


            self._process_data([init_in], init_outputs)

            if params_adjusted:
                self._pad_width = orig_pad_width



    def _process_data(self, inputs, outputs):
        """
        Conduct padding on the data

        Parameters
        ----------

        inputs: list of Tensors
            Input data container, list of length 1

        outputs: list of Tensors
            Output data container, list of length 1
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

    @classmethod
    def add_to_graph(cls, graph, inA, outA, pad_width = None, mode = 'constant',
                     stat_length = None, constant_values = 0, end_values = 0,
                     reflect_type = 'even', init_input=None, init_labels=None, **kwargs):
        """
        Add a pad kernel to the graph

        Parameters
        ----------

        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor
            Input data (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        outA : Tensor
            Output data (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        pad_width : int or sequence of ints, optional
            Number of values padded to the edges of each axis.
            See :func:`numpy.pad <numpy:numpy.pad>`.
        mode : str or function, optional
            String values or a user supplied function.
            See :func:`numpy.pad <numpy:numpy.pad>`.
        stat_length : sequence or int, optional
            Number of values at edge of each axis used to calculate the statistic value
            See :func:`numpy.pad <numpy:numpy.pad>`.
        constant_values : sequence or int, optional
            The values to set the padded values for each axis.
            See :func:`numpy.pad <numpy:numpy.pad>`.
        end_values : sequence or int, optional
            The values used for the ending value of the linear_ramp and that 
            will form the edge of the padded array.
            See :func:`numpy.pad <numpy:numpy.pad>`.
        reflect_type : str, optional
            See :func:`numpy.pad <numpy:numpy.pad>`.
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
