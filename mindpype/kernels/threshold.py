from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter


class ThresholdKernel(Kernel):
    """
    Determine if scalar or tensor data elements are above or below threshold

    Parameters
    ----------

    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Scalar
        Input data

    outA : Tensor or Scalar
        Output data

    thresh : float
        Threshold value

    """

    def __init__(self,graph,inA,outA,thresh):
        """ Init """
        super().__init__('Threshold',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA, thresh]
        self.outputs = [outA]

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]
        init_thresh = init_inputs[1]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            if init_thresh.mp_type in (MPEnums.ARRAY, MPEnums.CIRCLE_BUFFER):
                init_thresh = init_thresh.to_tensor()

            # set the output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

            self._process_data([init_in, init_thresh], init_outputs)

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized
        """

        d_in = self.inputs[0]
        d_out = self.outputs[0]
        thresh = self.inputs[1]

        # input/output must be a tensor or scalar
        if not ((d_in.mp_type == MPEnums.TENSOR and d_out.mp_type == MPEnums.TENSOR) or
                (d_in.mp_type == MPEnums.SCALAR and d_out.mp_type == MPEnums.SCALAR)):
            raise TypeError("Threshold Kernel: Input and output must be either both tensors or both scalars")

        if d_in.mp_type == MPEnums.TENSOR:
            # input tensor must contain some values
            if len(d_in.shape) == 0:
                raise ValueError("Threshold Kernel: Input tensor must contain some values")

        if thresh.mp_type != MPEnums.SCALAR:
            raise TypeError("Threshold Kernel: Threshold value must be a scalar")

        if not thresh.is_numeric:
            raise TypeError("Threshold Kernel: Threshold value must be numeric")

        if d_out.mp_type == MPEnums.TENSOR:
            if d_out.virtual and len(d_out.shape) == 0:
                d_out.shape = d_in.shape

            if d_out.shape != d_in.shape:
                raise ValueError("Threshold Kernel: Input and output tensors must have the same shape")

        else:
            if not (d_in.is_numeric and d_out.is_numeric):
                raise TypeError("Threshold Kernel: Input and output scalars must be numeric")

            if d_out.data_type != d_in.data_type:
                raise TypeError("Threshold Kernel: Input and output scalars must have the same data type")

    def _process_data(self, inputs, outputs):
        """
        Determine if data elements are below or above threshold.

        Parameters
        ----------
        inputs: list of Tensors or Scalars 
            Input data container, list of length 1

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        thresh = inputs[1]
        outputs[0].data = inputs[0].data > thresh.data

    @classmethod
    def add_to_graph(cls,graph,inA,outA,thresh,init_inputs=None,init_labels=None):
        """
        Factory method to create a threshold value kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------

        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor or Scalar
            Input data

        outA : Tensor or Scalar
            Output data

        thresh : float
            Threshold value

        init_inputs : Tuple of Tensors or None
            Initialization data for the input and threshold
            Initialization data will be transformed by the kernel
            and passed to any downstream nodes within the graph during
            initialization.

        init_labels : Tensor or None
            Labels for the initialization to be passed to downstream nodes.            

        Returns
        -------
        node : Node
            Node object that was added to the graph containing the kernel
        """

        # create the kernel object
        k = cls(graph,inA,outA,thresh)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT),
                  Parameter(thresh,MPEnums.INPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node
