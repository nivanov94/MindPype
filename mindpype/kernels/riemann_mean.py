from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

from pyriemann.utils.mean import mean_riemann

class RiemannMeanKernel(Kernel):
    """
    Calculates the Riemann mean of covariances contained in a tensor.
    Kernel takes 3D Tensor input and produces 2D Tensor representing mean

    .. note::
        This kernel utilizes the numpy function
        :func:`mean_riemann <pyriemann:pyriemann.utils.mean.mean_riemann>`,
    
    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor
        Input data

    outA : Tensor
        Output data

    axis : int
        Axis over which the mean should be calculated (see np.mean for more info)

    weights : array_like
        Weights for each sample
    """

    def __init__(self,graph,inA,outA,weights):
        """ Init """
        super().__init__('RiemannMean',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA]
        self.outputs = [outA]
        self._covariance_inputs = (0,)

        self._w = weights

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized
        """
        init_in = init_inputs[0]
        init_out = init_outputs[0]

        if init_out is not None and (init_in is not None and init_in.shape != ()):
            if init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

            # update output size, as needed
            if init_out.virtual:
                init_out.shape = init_in.shape[-2:] # TODO what are the expected inputs? will we ever compute more than one mean here?

            self._process_data([init_in], init_outputs)


    def _process_data(self, inputs, outputs):
        """
        Calculate the Riemann mean of covariances contained in tensor.

        Parameters
        ----------

        inputs: list of Tensors
            Input data container, list of length 1
        
        outputs: list of Tensors
            Output data container, list of length 1
        """
        outputs[0].data = mean_riemann(inputs[0].data, sample_weight=self._w)

    @classmethod
    def add_to_graph(cls,graph,inA,outA,weights=None,init_input=None,init_labels=None):
        """
        Factory method to create a Riemann mean calculating kernel

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor
            Input data

        outA : Tensor
            Output data

        weights : array_like, default=None
        """

        # create the kernel object
        k = cls(graph,inA,outA,weights)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_input is not None:
            node.add_initialization_data([init_input],init_labels)

        return node

