from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor, CircleBuffer
from ..core import MPEnums
from .kernel_utils import extract_nested_data
import numpy as np

class RunningAverageKernel(Kernel):
    """
    Kernel to calculate running average across multiple trials in a session. 
    Trials are automatically included into the next running average
    calculation.

    Parameters
    ----------
    inA : Tensor or Scalar
        Single Trial input data to the RunningAverageKernel; should be a 2D Tensor or Scalar object

    outA : Tensor or Scalar
        Output Tensor to store output of mean trial calculation; should be the same size 
        of the input tensor or a scalar.

    running_average_len : int
        Indicates the maximum number of trials that the running average kernel will be used to compute. 
        Used to preallocate tensor to store previous trial data

    axis : None or 0:
        Axis by which to calculate running average. Currently only supports mean across trials when axis = 0
        (ie. Average Tensor layer values), or single value mean, axis = None

    flush_on_init : bool, default = False
        If true, flushes the buffer on initialization.

    """
    def __init__(self, graph, inA, outA, window_len, axis = 0, flush_on_init = False):
        """ Init """
        super().__init__('RunningAverage', MPEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self._flush_on_init = flush_on_init
        self._axis = axis
        self._window_len = window_len
        self._data_buff = None


    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state to be initialized. Call initialization_execution 
        if downstream nodes are missing training data.
        
        Parameters
        ----------

        init_inputs: Tensor or Scalar
            Input data

        init_outputs: Tensor or Scalar
            Output data
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if self._flush_on_init:
            self._data_buff.flush()

        if init_in is not None:
            # check that the input is a tensor or array
            accepted_inputs = (MPEnums.ARRAY, MPEnums.TENSOR, MPEnums.CIRCLE_BUFFER)
            if init_in.mp_type not in accepted_inputs:
                raise TypeError("Running Average Kernel: Initialization input must be a Tensor or Circle Buffer")

            # extract the data fron the input and place it into the buffer
            if init_in.mp_type == MPEnums.TENSOR:
                if len(init_in.shape) != (len(self.inputs[0].shape)+1) or init_in.shape[1:] != self.inputs[0].shape:
                    raise ValueError("Running Average Kernel: Initialization input shape not compatible with input shape")
                for i in range(init_in.shape[0]):
                    data = init_in.data[i]
                    t_tmp = Tensor.create_from_data(self.session, data)
                    self._data_buff.enqueue(t_tmp)
            else:
                for i in range(init_in.num_elements):
                    data = extract_nested_data(init_in,i)
                    self.inputs[1].enqueue(data)

        if init_out is not None:
            if init_out.virtual:
                init_out.shape = self.outputs[0].shape

            # extract the data from the buffer
            X = extract_nested_data(self.inputs[1])
            init_out.data = np.mean(X,axis=self._axis)

    
    def _verify(self):
        """
        Verify the input and output types and shapes
        Create the buffer according to the input tensor shape
        """
        # check that the input is a tensor or scalar
        accepted_inputs = (MPEnums.TENSOR, MPEnums.SCALAR)
        if self.inputs[0].mp_type not in accepted_inputs:
            raise TypeError("Running Average Kernel: Input must be a Tensor or Scalar")

        # check that the output is the same type as the input
        if self.outputs[0].mp_type != self.inputs[0].mp_type:
            raise TypeError("Running Average Kernel: Output must be the same type as the input")
        
        # if the input is a tensor, check that the output has the same shape
        if self.inputs[0].mp_type == MPEnums.TENSOR:
            input_sz = self.inputs[0].shape
            if self.outputs[0].virtual:
                self.outputs[0].shape = input_sz
            
            if self.outputs[0].shape != input_sz:
                raise ValueError("Running Average Kernel: Output shape must match input")

        # need to create the buffer here because this is the first time we are guaranteed
        # to have the input size defined
        self._data_buff = CircleBuffer.create(self.session, self._window_len, self.inputs[0])


    def _reset_internal_state(self):
        """
        Reset the internal state of the kernel
        """
        self._data_buff.flush()


    def _process_data(self, inputs, outputs):
        """
        Add input data to data object storing previous trials and process the stacked data

        Parameters
        ----------

        inputs: list of Tensors or Scalars
            Input data container, list of length 1

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        # enqueue the input data
        self._data_buff.enqueue(inputs[0])

        # extract the data from the buffer
        X = extract_nested_data(self._data_buff)
        outputs[0].data = np.mean(X, axis=self._axis)


    @classmethod
    def add_to_graph(cls, graph, inA, outA, window_len, axis=0, flush_on_init=False, init_input=None, init_labels=None):
        """
        Factory method to create running average node and add it to the specified graph

        Parameters
        ----------
        graph : Graph
            The graph where the node object should be added

        inA : Tensor or Scalar
            Single input data to the RunningAverageKernel, should be a Tensor or Scalar object.
        
        buffer : CircleBuffer
            CircleBuffer object to store previous data. Defines the maximum number of trials
            that the running average kernel will be used to compute.

        outA : Tensor or Scalar
            Output Tensor to store output of mean trial calculation; should be the same size 
            of the input tensor or a scalar.

        axis : None or 0:
            Axis by which to calculate running average. Currently only supports mean across trials when axis = 0 
            (ie. Average Tensor layer values), or single value mean, axis = None

        flush_on_init : bool
            If true, flushes the buffer on initialization.

        Returns
        -------
        node : Node
            The node object that was added to the graph containing the running average kernel

        """
        kernel = cls(graph, inA, outA, window_len, axis, flush_on_init)

        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        node = Node(graph, kernel, params)

        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_input is not None:
            node.add_initialization_data([init_input], init_labels)

        return node
