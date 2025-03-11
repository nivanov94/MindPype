from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor
from .kernel_utils import extract_init_inputs

import numpy as np

class MatrixMultKernel(Kernel):
    """
    Kernel to perform matrix multiplication of two tensors.

    Parameters
    ----------
    graph: Graph
        The graph to which the kernel's node belongs.
    inA: Tensor
        The first tensor to be multiplied.
    inB: Tensor
        The second tensor to be multiplied.
    outA: Tensor
        The output tensor.

    See Also
    --------
    Kernel: Base class for all kernels.
    """

    def __init__(self, graph, inA, inB, outA):
        super().__init__("MatrixMult", MPEnums.INIT_FROM_NONE, graph)
        self.inputs = [inA, inB]
        self.outputs = [outA]

        self._covariance_inputs = (0, 1) # HACK for now probably need to remove verification pass... too restrictive

    def _compute_output_sz(self, inA, inB):
        """
        Compute the output tensor size based on the input tensors.

        Parameters
        ----------
        inA: Tensor
            The first input tensor.
        
        inB: Tensor
            The second input tensor.

        Returns
        -------
        tuple
            The output tensor size.
        """
        # case 1: both inputs are 2D tensors
        if len(inA.shape) == 2 and len(inB.shape) == 2:
            # check that the inner dimensions match
            if inA.shape[1] != inB.shape[0]:
                raise ValueError("Inner dimensions of input tensors must match.")
            return (inA.shape[0], inB.shape[1])
        
        # case 2: inA is a 3D tensor and inB is a 2D tensor
        elif len(inA.shape) == 3 and len(inB.shape) == 2:
            # check that the inner dimensions match
            if inA.shape[2] != inB.shape[0]:
                raise ValueError("Inner dimensions of input tensors must match.")
            return (inA.shape[0], inA.shape[1], inB.shape[1])
        
        # case 3: inA is a 2D tensor and inB is a 3D tensor
        elif len(inA.shape) == 2 and len(inB.shape) == 3:
            # check that the inner dimensions match
            if inA.shape[1] != inB.shape[0]:
                raise ValueError("Inner dimensions of input tensors must match.")
            return (inB.shape[0], inA.shape[0], inB.shape[2])
        
        # case 4: both inputs are 3D tensors
        elif len(inA.shape) == 3 and len(inB.shape) == 3:
            # check that the inner dimensions match
            if inA.shape[2] != inB.shape[1]:
                raise ValueError("Inner dimensions of input tensors must match.")
            return (inA.shape[0], inA.shape[1], inB.shape[2])
        
        else:
            return ()

    def _initalize(self, init_inputs, init_outputs, labels=None):
        """
        Compute any initialization data outputs needed to 
        initialize other nodes within the graph.
        
        Parameters
        ----------
        init_inputs: list
            List of input data to be used for initialization. List of length 2.
        init_outputs: list 
            List of output data to be used for initialization. List of length 1.
        labels: None
            Not used, only included for compatibility with Kernel class.
        """

        init_inA, init_inB = init_inputs
        init_outA = init_outputs[0]

        # check if any initalization is needed
        if init_inA is None or init_inB is None or init_outA is None:
            return None
        
        # check that the init inputs are valid tensors
        if init_inA.mp_type != MPEnums.TENSOR or init_inB.mp_type != MPEnums.TENSOR:
            raise TypeError("Invalid initialization input type. Expected Tensor.")
        
        # set the output tensor shape, as needed
        output_shape = self._compute_output_sz(init_inA, init_inB)
        if init_outA.virtual:
            init_outA.shape = output_shape

        # process the initialization data
        self._process_data([init_inA, init_inB], init_outputs)

    def _verify(self):
        """
        Verify the kernel parameters.
        """
        # check that the input and output tensors are valid
        inA, inB = self.inputs
        outA = self.outputs[0]

        # input and output must be a tensor
        for param in self.inputs + self.outputs:
            if param.mp_type != MPEnums.TENSOR:
                raise TypeError("Input and output must be a tensor.")
    
        # check input tensor dimensions
        out_sz = self._compute_output_sz(inA, inB)

        if len(out_sz) == 0:
            raise ValueError("Invalid input tensor dimensions.")

        if outA.virtual:
            outA.shape = out_sz

        if outA.shape != out_sz:
            raise ValueError("Output tensor shape does not match expected shape.")
        
    def _process_data(self, inputs, outputs):
        """
        Perform the matrix multiplication operation.

        Parameters
        ----------
        inputs: list
            List of input data. List of length 2.
        outputs: list
            List of output data. List of length 1.
        """
        inA, inB = inputs
        outA = outputs[0]

        # case 1: both inputs are 2D tensors
        if len(inA.shape) == 2 and len(inB.shape) == 2:
            outA.data = np.dot(inA.data, inB.data)

        # case 2: inA is a 3D tensor and inB is a 2D tensor
        elif len(inA.shape) == 3 and len(inB.shape) == 2:
            outA.data = np.einsum('ijk,kl->ijl', inA.data, inB.data)

        # case 3: inA is a 2D tensor and inB is a 3D tensor
        elif len(inA.shape) == 2 and len(inB.shape) == 3:
            outA.data = np.einsum('ij,mjk->mik', inA.data, inB.data)

        # case 4: both inputs are 3D tensors
        else:
            outa_data = np.zeros(self._compute_output_sz(inA, inB))
            for i in range(inA.shape[0]):
                outa_data[i] = np.matmul(inA.data[i], inB.data[i])
            outA.data = outa_data

    @classmethod
    def add_to_graph(cls, graph, inA, inB, outA, init_inputs=None, init_labels=None):
        """
        Factory method to create a MatrixMultKernel node and add it to a graph.

        Parameters
        ----------
        graph: Graph
            The graph to which the kernel's node belongs.
        inA: Tensor
            The first tensor to be multiplied.
        inB: Tensor
            The second tensor to be multiplied.
        outA: Tensor
            The output tensor.
        init_inputs: list of Tensor
            The initialization input tensors. Default is None.
        init_labels: Tensor
            The initialization labels tensor. Default is None.

        Returns
        -------
        Node
            The node representing the MatrixMultKernel operation.
        """
        # create the kernel node
        k = cls(graph, inA, inB, outA)

        params = (
            Parameter(inA, MPEnums.INPUT),
            Parameter(inB, MPEnums.INPUT),
            Parameter(outA, MPEnums.OUTPUT)
        )

        # add the kernel to a generic node object
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        if init_inputs is not None or init_labels is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node