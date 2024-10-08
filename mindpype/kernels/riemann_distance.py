from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor

import numpy as np

from pyriemann.utils.distance import distance_riemann

class RiemannDistanceKernel(Kernel):
    """
    Kernel computes pairwise distances between 2D tensors according to the riemann metric

    .. note::
        This kernel utilizes the pyriemann function
        :func:`distance_riemann <pyriemann:pyriemann.utils.distance.distance_riemann>`,

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to

    inA : Tensor or Array
        Input 1 data

    inB : Tensor or Array
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

        inA : Tensor or Array
            Input 1 data

        inB : Tensor or Array
            Input 2 data
        """
        out_sz = []
        mat_sz = None
        for param in (inA,inB):
            if param.mp_type == MPEnums.TENSOR:
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
                #ensure it is an array of 2D tensors
                param_rank = param.capacity
                for i in range(param_rank):
                    e = param.get_element(i)
                    if not isinstance(e,Tensor) or len(e.shape) != 2:
                        raise TypeError("RiemannianDistance kernel: All elements of input array must be 2D tensors")

                    if mat_sz == None:
                        mat_sz = e.shape
                    elif mat_sz != e.shape:
                        return ()

                out_sz.append(param_rank)

        return tuple(out_sz)

    def _verify(self):
        """
        Verify the inputs and outputs are appropriately sized and typed
        """

        inA, inB = self.inputs
        outA = self.outputs[0]

        # first ensure the input and output are tensors or Arrays of Tensors
        for param in (inA, inB, outA):
            if (param.mp_type != MPEnums.TENSOR and
                param.mp_type != MPEnums.ARRAY):
                raise TypeError("RiemannianDistance kernel: All inputs and outputs must be Tensors or Arrays")

        out_sz = self._compute_output_shape(inA, inB)
        num_combos = out_sz[0]*out_sz[1]

        # if the output is a virtual tensor and dimensionless,
        # add the dimensions now
        if (outA.virtual and len(outA.shape) == 0):
            outA.shape = out_sz


        if (outA.mp_type != MPEnums.TENSOR and
            outA.shape != out_sz):
            raise ValueError("RiemannianDistance kernel: Output shape does not match expected shape")
        elif outA.mp_type == MPEnums.ARRAY:
            if outA.capacity != num_combos:
                raise ValueError("RiemannianDistance kernel: Output array capacity does not match expected capacity")

            for i in range(outA.capacity):
                e = outA.get_element(i)
                if ((e.mp_type != MPEnums.TENSOR and
                     e.mp_type != MPEnums.SCALAR) or
                    (e.mp_type == MPEnums.TENSOR and np.squeeze(e.shape) != ()) or
                    (e.mp_type == MPEnums.SCALAR and e.data_type != float)):
                    raise TypeError("RiemannianDistance kernel: All elements of output array must be single-valued Tensors or Scalars")

    def _process_data(self, inputs, outputs):
        """
        Execute the kernel and calculate the mean

        Parameters
        ----------

        inputs: list of Tensors or Arrays
            Input data container, list of length 2

        outputs: list of Tensors or Scalars
            Output data container, list of length 1
        """
        def get_obj_data_at_index(obj,index,rank):
            """
            Get value at specified index from tensor or array

            Parameters
            ----------

            obj: Tensor or Array
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
            else:
                return obj.get_element(index).data

        def set_obj_data_at_index(obj,index,data):
            """
            Set value at specified index from tensor or array

            Parameters
            ----------

            obj: Tensor or Array
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
            else:
                e = obj.get_element(index[0]*index[1])
                if e.mp_type == MPEnums.TENSOR:
                    e.data = np.asarray([[data]])
                else:
                    e.data = data

        out_sz = []
        for in_param in inputs:
            if in_param.mp_type == MPEnums.TENSOR:
                if len(in_param.shape) == 3:
                    m = in_param.shape[0]
                else:
                    m = 1
            else:
                m = in_param.capacity

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

        inA : Tensor or Array
            Input 1 data

        inB : Tensor or Array
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

