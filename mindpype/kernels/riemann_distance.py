from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor

import numpy as np

from pyriemann.utils.distance import distance_riemann

class RiemannDistanceKernel(Kernel):
    """
    Calculates the Riemann mean of covariances contained in a tensor. 
    Kernel computes pairwise distances between 2D tensors

    .. note::
        This kernel utilizes the numpy functions
        :func:`squeeze <numpy:numpy.squeeze>`,
        :func:`asarray <numpy:numpy.asarray>`.

    .. note::
        This kernel utilizes the pyriemann function
        :func:`distance_riemann <pyriemann:pyriemann.utils.distance.distance_riemann>`,

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor
        Input 1 data

    inB : Tensor
        Input 2 data

    outA : Tensor or Scalar
        Output data

    """

    def __init__(self,graph,inA,inB,outA):
        """ Init """
        super().__init__('RiemannDistance',MPEnums.INIT_FROM_NONE,graph)
        self.inputs = [inA,inB]
        self.outputs = [outA]

        self._covariance_inputs = (0,1)

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        This kernel has no internal state that must be initialized. Call initialization_execution if downstream nodes are missing training data
        """

        init_inA, init_inB = init_inputs
        init_out = init_outputs[0]

        for init_in in (init_inA, init_inB):
            if init_in is not None and init_in.mp_type != MPEnums.TENSOR:
                init_in = init_in.to_tensor()

        if init_out is not None and (init_inA is not None and init_inA.shape != ()):
            # update output size, as needed
            if init_out.virtual:
                output_sz = self._compute_output_shape(init_inA, init_inB)
                init_out.shape = output_sz

            self._process_data([init_inA, init_inB], init_outputs)


    def _compute_output_shape(self, inA, inB):
        """
        Determine the shape of the tensor that contains the calculated Riemann mean of covariances

        Parameters
        ----------

        inA : Tensor 
            Input 1 data

        inB : Tensor
            Input 2 data
        """
        out_sz = []
        mat_sz = None
        for param in (inA,inB):
            if param._mp_type == MPEnums.TENSOR:
                # ensure it is 3D or 2D
                param_rank = len(param.shape)
                if param_rank != 2 and param_rank != 3:
                    print("Both inputs must be either 2D or 3D")
                    return ()

                if mat_sz == None:
                    mat_sz = param.shape[-2:]
                elif param.shape[-2:] != mat_sz:
                    return ()

                if param_rank == 3:
                    out_sz.append(param.shape[0])
                else:
                    out_sz.append(1)
            else:
                raise TypeError("RiemannianDistance kernel: Input should be tensor")

        return tuple(out_sz)

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """

        inA, inB = self.inputs
        outA = self.outputs[0]

        # first ensure the input and output are tensors or Arrays of Tensors
        for param in (inA, inB, outA):
            if (param.mp_type != MPEnums.TENSOR):
                raise TypeError("RiemannianDistance kernel: All inputs and outputs must be Tensors")

        out_sz = self._compute_output_shape(inA, inB)
        num_combos = out_sz[0]*out_sz[1]

        # if the output is a virtual tensor and dimensionless,
        # add the dimensions now
        if (outA.virtual and len(outA.shape) == 0):
            outA.shape = out_sz


        if (outA.mp_type == MPEnums.TENSOR and
            outA.shape != out_sz):
            raise ValueError("RiemannianDistance kernel: Output shape does not match expected shape")
    
    def _process_data(self, inputs, outputs):
        """
        Execute the kernel and calculate the mean

        Parameters
        ----------

        inputs: Tensor
            Input data container

        outputs: Tensor or Scalar
            Output data container
        """
        def get_obj_data_at_index(obj,index,rank):
            """
            Get value at specified index from tensor or array

            Parameters
            ----------

            obj: Tensor 
                Data container to extract data from
            
            index: Int
                Index to extract data from

            rank: Int
                Number of dimensions of data container

            Returns
            -------

            data: numpy.ndarray
                Data at specified index of tensor or array
            """
            if obj.mp_type == MPEnums.TENSOR:
                if rank == 1 and len(obj.shape) == 2:
                    return obj.data
                else:
                    return obj.data[index,:,:]

        def set_obj_data_at_index(obj,index,data):
            """
            Set value at specified index from tensor 

            Parameters
            ----------

            obj: Tensor 
                Data container to extract data from
            
            index: Int
                Index to extract data from

            rank: Int
                Number of dimensions of data container
            """
            if obj.mp_type == MPEnums.TENSOR:
                tensor_data = obj.data # need to extract and edit numpy array b/c tensor currently does not allow sliced modifications
                tensor_data[index] = data
                obj.data = tensor_data

        out_sz = []
        for in_param in inputs:
            if in_param.mp_type == MPEnums.TENSOR:
                if len(in_param.shape) == 3:
                    m = in_param.shape[0]
                else:
                    m = 1

            out_sz.append(m)


        for i in range(out_sz[0]):
            # extract the ith element from inA
            x = get_obj_data_at_index(inputs[0],i,out_sz[0])

            for j in range(out_sz[1]):
                # extract the jth element from inB
                y = get_obj_data_at_index(inputs[1],j,out_sz[1])

                set_obj_data_at_index(outputs[0],(i,j),
                                      distance_riemann(x,y))

    @classmethod
    def add_to_graph(cls,graph,inA,inB,outA,init_inputs=None,init_labels=None):
        """
        Factory method to create a Riemann mean calculating kernel

        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to

        inA : Tensor
            Input 1 data

        inB : Tensor
            Input 2 data

        outA : Tensor or Scalar
            Output data

        Returns
        -------
        node : Node
            Node object that contains the kernel and parameters
        """

        # create the kernel object
        k = cls(graph,inA,inB,outA)

        # create parameter objects for the input and output
        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(inB,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        # add the kernel to a generic node object
        node = Node(graph,k,params)

        # add the node to the graph
        graph.add_node(node)

        # if initialization data is provided, add it to the node
        if init_inputs is not None:
            node.add_initialization_data(init_inputs, init_labels)

        return node

