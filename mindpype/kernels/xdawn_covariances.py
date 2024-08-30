from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

from pyriemann.estimation import XdawnCovariances
import numpy as np

class XDawnCovarianceKernel(Kernel):
    """
    Kernel to perform XDawn spatial filtering and covariance estimation.

    .. note:: 
        This kernel utilizes the 
        :class:`XdawnCovariances <pyriemann:pyriemann.estimation.XdawnCovariances>` 
        class from the pyriemann package.

    Parameters
    ----------
    graph : Graph
        Graph that the kernel should be added to
    inA : Tensor
        Input data container
    outA : Tensor
        Output data container
    initialization_data : Tensor 
        Data to initialize the estimator with (n_trials, n_channels, n_samples)
    labels : Tensor
        Class labels for initialization data
    n_filters : int, default=4
        Number of Xdawn filters per class.
    classes : list of int | None, default=None
        list of classes to use for prototype estimation.
        If None, all classes will be used.
    n_classes : int, default=2
        Number of classes to use for prototype estimation

    See Also
    --------
    :class:`Kernel` : Base class for all kernels

    :class:`XdawnCovariances <pyriemann:pyriemann.estimation.XdawnCovariances>` : XDawn Covariance Estimator
    """

    def __init__(self, graph, inA, outA, initialization_data=None, labels=None, 
                 n_filters=4, classes=None):
        """Init"""
        super().__init__("XDawnCovarianceKernel", MPEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        self.add_initialization_data([initialization_data], labels)

        self._initialized = False
        self.n_filters = n_filters
        self._xdawn_estimator = XdawnCovariances(nfilter=n_filters, classes=classes)


    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize the internal state of the kernel. Fit the xdawn_estimator classifier
        using the provided initialization data.

        Parameters
        ----------
        init_inputs : list of Tensors
            Initialization input data container, list of length 1

        init_outputs : list of Tensors
           Initialization output data container, list of length 1

        labels : Tensor
            Labels corresponding to the initialization data class labels (n_trials,)
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        # check if the initialization data is in a Tensor, if not convert it
        if init_in.mp_type != MPEnums.TENSOR:
            init_in = init_in.to_tensor()

        # check if the labels are in a tensor
        if labels.mp_type != MPEnums.TENSOR:
            labels = labels.to_tensor()

        n_cls = np.unique(labels.data).shape[0]

        # fit the xdawn estimator
        self._xdawn_estimator.fit(init_in.data, np.squeeze(labels.data))

        # compute the initialization output
        if init_in is not None and init_out is not None:
            # update the init output shape as needed
            Nt = init_in.shape[0]
            Nc = self._xdawn_estimator.nfilter*(n_cls**2)
            if init_out.shape != (Nt,Nc,Nc):
                init_out.shape = (Nt,Nc,Nc)
            # process the initialization data
            self._process_data([init_in], init_outputs)

    def _process_data(self, inputs, outputs):
        """
        Perform XDawn spatial filtering on the input data 
        and store the result in the output data.

        Parameters
        ----------

        inputs : list of Tensors
            Input data container, list of length 1
        outputs : list Tensors
            Output data container, list of length 1
        """
        input_data = inputs[0].data

        if len(inputs[0].shape) == 2:
            input_data = input_data[np.newaxis, :, :] # input must be 3D
            print(input_data.shape)
            print(self._xdawn_estimator.P_.shape)

        outputs[0].data = self._xdawn_estimator.transform(input_data.data)

    @classmethod
    def add_to_graph(cls, graph, inA, outA, initialization_data=None, labels=None,
                     num_filters=4, classes=None):
        """
        Factory method to create xdawn_covariance kernel, add it to a node,
        and add the node to the specified graph.

        Parameters
        ----------

        graph : Graph
            Graph that the kernel should be added to
        inA : Tensor
            Input data container
        outA : Tensor
            Output data container
        initialization_data : Tensor
            Data to initialize the estimator with (n_trials, n_channels, n_samples)
        labels : Tensor
            Class labels for initialization data
        n_filters : int, default=4
            Number of Xdawn filters per class.
        classes : list of int | None, default=None
            list of classes to use for prototype estimation.
            If None, all classes will be used.
        n_classes : int, default=2
            Number of classes to use for prototype estimation

        Returns
        -------
        node : Node
            Node containing the kernel
        """
        kernel = cls(graph, inA, outA, initialization_data, labels, 
                     num_filters, classes)

        params = (Parameter(inA, MPEnums.INPUT),
                  Parameter(outA, MPEnums.OUTPUT))

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node
