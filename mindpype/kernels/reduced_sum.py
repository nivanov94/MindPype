from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor

import numpy as np

class ReducedSumKernel(Kernel):
    """
    Kernel to compute the sum of the input tensor's
    element along the provided axis

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor
        Input data

    outA : Tensor or Scalar
        Output data

    axis : int or tuple of ints, default = None
        Axis or axes along which the sum is computed.

    keep_dims : bool, default = False
        If true, the reduced dimensions are retained with length 1

    """

    def __init__(self, graph, inA, outA,
                 axis=None, keep_dims=False):
        """ Init """
        super().__init__('ReducedSum',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._axis = axis
        self._keep_dims = keep_dims


    def _compute_output_sz(self, input_sz):
        """ 
        Determine the shape of the output

        Parameters
        ----------

        input_sz: nd.array
            Dimensions of the input data tensor
        """
        # determine the axes that will be reduced
        if self._axis is not None:
            reduced_axes = [a==self._axis for a in range(len(input_sz))]
        else:
            reduced_axes = [True] * len(input_sz)

        output_sz = []
        for i in range(len(input_sz)):
            if reduced_axes[i] and self._keep_dims:
                output_sz.append(1)
            elif not reduced_axes[i]:
                output_sz.append(input_sz[i])
        
        if len(output_sz) == 0:
            output_sz = (1,)
        else:
            output_sz = tuple(output_sz)

        return output_sz

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            tmp_axis = self._axis  # Axis may be modified during initialization, so save it temporarily
            tmp_keep_dims = self._keep_dims  # Keep dims may be modified during initialization, so save it temporarily
            override_output_shape = False
            
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            if self._axis is not None:
                # if init input is higher dimensional than input, increase axis
                if len(init_in.shape) > len(self.inputs[0].shape):
                    self._axis += 1

            else:
                # if the init input is higher dimensional than the input, then resize the input
                if len(init_in.shape) > len(self.inputs[0].shape):
                    self._axis = 1
                    init_in_data = init_in.data
                    init_in = Tensor.create_virtual(self.session)
                    init_in.data = np.reshape(init_in_data, (init_in_data.shape[0], -1))

                    if self._keep_dims:
                        output_sz = (init_in_data.shape[0],) + (1,) * (len(init_in_data.shape) - 1)
                        override_output_shape = True
                    else:
                        self._keep_dims = True

            # adjust the shape of init output tensor, as needed
            if init_out.virtual:
                if override_output_shape:
                    init_out.shape = output_sz
                else:
                    input_sz = list(init_in.shape)
                    output_sz = self._compute_output_sz(input_sz)
                init_out.shape = output_sz

            self._process_data([init_in], init_outputs)

            if override_output_shape:
                # need to manually set the output shape here since the 
                # output shape is not set correctly in the process_data method
                output_data = init_outputs[0].data
                init_outputs[0].shape = output_sz
                init_outputs[0].data = np.reshape(output_data, output_sz)

            self._axis = tmp_axis  # Restore the original axis value
            self._keep_dims = tmp_keep_dims  # Restore the original keep_dims value

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """

        inA = self.inputs[0]
        outA = self.outputs[0]

        # first ensure the inputs and outputs are the appropriate type
        if inA.mp_type != MPEnums.TENSOR:
            raise TypeError('ReducedSum kernel requires Tensor input')

        if (outA.mp_type != MPEnums.TENSOR and
            outA.mp_type != MPEnums.SCALAR):
            raise TypeError('ReducedSum kernel requires Tensor or Scalar output')

        if (outA.mp_type == MPEnums.SCALAR and
            (not outA.is_numeric)):
            raise TypeError('ReducedSum kernel requires Scalar output to be numeric')
        
        inA_shape = inA.shape

        if self._axis is not None:
            # if negative axis is provided, convert it to positive
            if self._axis < 0:
                self._axis = len(inA_shape) + self._axis

            if self._axis < 0 or self._axis >= len(inA_shape):
                raise ValueError('ReducedSum kernel: axis out of bounds')

        out_shape = self._compute_output_sz(inA_shape)

        # if the output is a virtual tensor and has no defined shape, set the shape now
        if (outA.mp_type == MPEnums.TENSOR
            and outA.virtual
            and len(outA.shape) == 0):
            outA.shape = out_shape

        # ensure the output shape equals the expected output shape
        if outA.mp_type == MPEnums.TENSOR and outA.shape != out_shape:
            raise ValueError('ReducedSum kernel: output tensor shape does not match expected shape')
        elif outA.mp_type == MPEnums.SCALAR and out_shape != (1,):
            raise ValueError('ReducedSum kernel: Multidimensional output cannot be assigned to a Scalar')

    def _process_data(self, inputs, outputs):
        """
        Compute the sum of the tensor's element along given axis.

        Parameters
        ----------

        inputs: list of Tensors 
            Input data container, list of length 1

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        outputs[0].data = np.sum(inputs[0].data,
                                 axis=self._axis,
                                 keepdims=self._keep_dims)

    @classmethod
    def add_to_graph(cls, graph, inA, outA,
                     axis=None, keep_dims=False,
                     init_input=None, init_labels=None):
        """
        Factory method to create a reduced sum kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor
            Input data

        outA : Tensor or Scalar
            Output data

        axis : int or tuple of ints, default = None
            Axis or axes along which the sum is computed.

        keep_dims : bool, default = False
            If true, the reduced dimensions are retained with length 1

        Returns
        -------
        node : Node
            Node object that contains the kernel and its parameters
        """

        # create the kernel object
        k = cls(graph, inA, outA, axis, keep_dims)

        # create parameter objects for the input and output
        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_input is not None:
            node.add_initialization_data([init_input], init_labels)

        return node
