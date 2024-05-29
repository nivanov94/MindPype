from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

from scipy import signal
import numpy as np

class ResampleKernel(Kernel):
    """
    Kernel to resample timeseries data

    .. note::
        This kernel utilizes the numpy function
        :func:`ceil <numpy:numpy.ceil>`.
    
    .. note::
        This kernel utilizes the sscipy module
        :mod:`signal <scipy:scipy.signal>`.


    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Array
        Input data

    factor: float
        Resample factor

    outA : Tensor
        Resampled timeseries data

    axis :
        The axis that is to be resampled
    """

    def __init__(self,graph,inA,factor,outA,axis = 1):
        """ Init """
        super().__init__('Resample',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._factor = factor
        self._axis = axis

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            axis_adjusted = False
            if len(init_in.shape) == len(self.inputs[0].shape)+1 and self._axis >= 0:
                self._axis += 1
                axis_adjusted = True

            self._process_data([init_in], init_outputs)

            if axis_adjusted:
                self._axis -= 1


    def _process_data(self, inputs, outputs):
        """
        Process trial data according to the scipy function

        Parameters
        ----------

        inputs: list of Tensors or Arrays
            Input data container, list of length 1

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        outputs[0].data = signal.resample(inputs[0].data,
                                          np.ceil(inputs[0].shape[self._axis] * self._factor).astype(int),
                                          axis=self._axis)


    @classmethod
    def add_to_graph(cls,graph,inA,factor,outA,axis=1,init_input=None,init_labels=None):
        """
        Factory method to create an extract kernel
        and add it to a graph as a generic node object.

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor or Array
            Input data

        factor: float
            Resample factor

        outA : Tensor
            Resampled timeseries data

        axis : int, default = 1
            The axis that is to be resampled

        Returns
        -------
        node : Node
            Node object that contains the kernel and its parameters
        """

        # create the kernel object
        k = cls(graph,inA,factor,outA,axis)

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
