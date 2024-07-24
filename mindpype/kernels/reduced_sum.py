from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

from ..containers import Scalar

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

    def __init__(self,graph,inA,outA,axis=None,keep_dims=False):
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
        if self._axis != None:
            axis = (self._axis,)
        else:
            axis = ()

        if self._keep_dims:
            # all reduced dimensions will be '1'
            out_shape = tuple([1 if i in axis else input_sz[i]
                                          for i in range(len(input_sz))])
        elif axis == ():
            out_shape = (1,)
        else:
            out_shape = tuple([input_sz[i] for i in range(len(input_sz))
                                                   if i not in axis])

        return out_shape


    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            # adjust the shape of init output tensor, as needed
            if init_out.virtual:
                input_sz = list(init_in.shape)
                output_sz = self._compute_output_sz(input_sz)
                init_out.shape = output_sz

            self._process_data([init_in], init_outputs)

    def verify(self):
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
            (outA.data_type not in Scalar.valid_numeric_types())):
            raise TypeError('ReducedSum kernel requires Scalar output to be numeric')

        inA_shape = inA.shape
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
    def add_to_graph(cls,graph,inA,outA,axis=None,keep_dims=False,init_input=None,init_labels=None):
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
        k = cls(graph,inA,outA,axis,keep_dims)

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
