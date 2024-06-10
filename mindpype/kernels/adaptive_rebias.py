from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from ..containers import Tensor

import numpy as np
import pyriemann


class AdaptiveRebiasKernel(Kernel):
    """
    Kernel for applying adaptive rebiasing to the data.
    
    Parameters
    ----------
    graph : Graph
        The graph object to which the kernel is associated.
    inA : Tensor
        The input tensor to the kernel.
    outA : Tensor
        The output tensor from the kernel.
    update_rate : float, default=None
        The update rate for the adaptive rebiasing. If None,
        the value will be dynmically set according to the 
        number of samples used to compute the bias.
    init_data : Tensor or Array, default=None
        The initial data for the adaptive rebiasing.
    init_labels : Tensor or Array, default=None
        The initial labels for the adaptive rebiasing.
    
    See Also
    --------
    Kernel : Base class for all kernels.
    """

    def __init__(self, graph, inA, outA, 
                 update_rate=0.1, 
                 init_data=None, 
                 init_labels=None):
        """Init."""
        super().__init__('AdaptiveRebias', MPEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self._update_rate = update_rate
        self._bias = None
        self._Nt = 0

        if init_data is not None:
            self._init_data = [init_data]
        
        if init_labels is not None:
            self._init_input_labels = init_labels


    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize the adaptive rebiasing kernel.
        
        Parameters
        ----------
        init_inputs : list
            The list of input tensors for the initialization.
        init_outputs : list
            The list of output tensors for the initialization.
        labels : list
            The list of labels for the initialization.
        """
        init_in = init_inputs[0]
        block_labels = labels[0]  # labels of the block/run when the samples were recorded

        # check that the input init data is in the correct type
        accepted_inputs = (MPEnums.TENSOR, MPEnums.ARRAY, MPEnums.CIRCLE_BUFFER)
        for init_obj in (init_in, block_labels):
            if init_obj.mp_type not in accepted_inputs:
                raise TypeError('Initialization data must be a tensor or array of tensors')
            

        # convert inputs to Tensors if they are not already
        if init_in.mp_type != MPEnums.TENSOR:
            init_in = init_in.to_tensor()

        if block_labels.mp_type != MPEnums.TENSOR:
            block_labels = block_labels.to_tensor()
        
        if init_outputs[0] is not None:
            init_out = init_outputs[0]
            # compute the mean covariance matrix for each block
            blocks = np.unique(block_labels.data)
            means = {b : pyriemann.utils.mean.mean_riemann(init_in.data[block_labels.data == b]) for b in blocks}

            # adjust output shape as needed
            if init_out.virtual:
                init_out.shape = init_in.shape

            # compute the rebiased covariance matrices for each block
            for b in blocks:
                t_in_tmp = Tensor.create_from_data(self.session, 
                                                   init_in.data[block_labels.data == b].data)
                t_out_tmp = Tensor.create(self.session,
                                          t_in_tmp.shape)
                self._bias = means[b]
                self._inv_sqrt_bias = pyriemann.utils.base.invsqrtm(self._bias)
                self._process_data([t_in_tmp], [t_out_tmp])
                init_out.data[block_labels.data == b] = t_out_tmp.data

        # compute the mean covariance for the entire dataset
        self._bias = pyriemann.utils.mean.mean_riemann(init_in.data)
        self._inv_sqrt_bias = pyriemann.utils.base.invsqrtm(self._bias)
        if self._update_rate is None:
            self._Nt = init_in.data.shape[0]



    def _update(self, init_inputs, init_outputs, labels):
        """
        Update the initalization of the adaptive rebiasing kernel.

        Parameters
        ----------
        init_inputs : list of Tensor or Array
            The list of input tensors for the initialization, list of 1
        init_outputs : list of Tensor or Array
            The list of output tensors for the initialization, list of 1
        labels : list of Tensor or Array
            The list of labels for the initialization, list of 1
        """
        self._initialize(init_inputs, init_outputs, labels)

    def _process_data(self, inputs, outputs):
        """
        Rebias the data by computing the geodesic between between the
        data and the bias covariance matrix.
        
        Parameters
        ----------
        inputs : list of Tensor
            Input data to be processed, list of 1
        outputs : list
            Output data from the processing, list of 1
        """
        inA = inputs[0]
        outA = outputs[0]
        
        # remove the bias from the data
        outA.data = self.inv_sqrt_bias @ inA.data @ self.inv_sqrt_bias
    
        # update the bias covariance matrix
        if self._update_rate is None:
            self._Nt += inA.data.shape[0]
            alpha = 1 / self._Nt
        else:
            alpha = self._update_rate

        self._bias = pyriemann.utils.geodesic.geodesic_riemann(self._bias, 
                                                               inA.data,
                                                               alpha)
        self._inv_sqrt_bias = pyriemann.utils.base.invsqrtm(self._bias)

    
    @classmethod
    def add_to_graph(cls, graph, inA, outA, 
                     update_rate=0.1, 
                     init_data=None, 
                     init_labels=None):
        """
        Factory method to create a rebias kernel and add it to the graph.
        
        Parameters
        ----------
        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor
            Input data to be rebias
        outA : Tensor
            Output data from the rebiasing process
        update_rate : float, default=None
            The update rate for the adaptive rebiasing. If None,
            the value will be dynmically set according to the 
            number of samples used to compute the bias.
        init_data : Tensor or Array, default=None
            The initial data for the adaptive rebiasing.
        init_labels : Tensor or Array, default=None
            The initial labels for the adaptive rebiasing.
        
        Returns
        -------
        Node
            The node containing the rebias kernel
            that was added to the graph
        """

        # create the kernel object
        k = AdaptiveRebiasKernel(graph, inA, outA,
                                 update_rate=update_rate,
                                 init_data=init_data,
                                 init_labels=init_labels)
        
        # create the parameters
        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))
        
        # create the node
        node = Node(graph, k, params)

        # add the node to the graph
        graph.add_node(node)

        # add the initialization data
        if init_data is not None:
            node.add_initialization_data(init_data, init_labels)
        
        return node
        