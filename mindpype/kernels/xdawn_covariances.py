from ..core import MPEnums
from ..kernel import Kernel
from ..graph import Node, Parameter

from pyriemann.estimation import XdawnCovariances
import numpy as np

class XDawnCovarianceKernel(Kernel):
    """
    Kernel to estimate special form covariance matrices for ERP combined with Xdawn

    Parameters
    ----------
    inA : Tensor
        Input data
    outA : Tensor
        Output data
    initialization_data : Tensor
        Data to initialize the estimator with (n_trials, n_channels, n_samples)
    labels : Tensor
        Class labels for initialization data
    nfilter : int, default=4
        Number of Xdawn filters per class.
    applyfilters : bool, default=True
        If true, spatial filter are applied to the prototypes and the signals. If False, filters are applied only to the ERP prototypes allowing for a better generalization across subject and session at the expense of dimensionality increase. In that case, the estimation is similar to pyriemann.estimation.ERPCovariances with svd=nfilter but with more compact prototype reduction.
    classeslist of int | None, default=None
        list of classes to take into account for prototype estimation. If None, all classes will be accounted.
    estimatorstring, default=’scm’
        Covariance matrix estimator, see pyriemann.utils.covariance.covariances().
    xdawn_estimatorstring, default=’scm’
        Covariance matrix estimator for Xdawn spatial filtering. Should be regularized using ‘lwf’ or ‘oas’, see pyriemann.utils.covariance.covariances().
    baseline_covarray, shape (n_chan, n_chan) | None, default=None
        Baseline covariance for Xdawn spatial filtering, see pyriemann.spatialfilters.Xdawn

    Examples
    --------
    """


    def __init__(self, graph, inA, outA, initialization_data=None, labels=None, num_filters=4, applyfilters=True,
                 classes=None, estimator='scm', xdawn_estimator='scm', baseline_cov=None, num_classes=2):
        """
        Constructor for the XDawnCovarianceKernel class
        """
        super().__init__("XDawnCovarianceKernel", MPEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        if initialization_data is not None:
            self.init_inputs = [initialization_data]

        if labels is not None:
            self.init_input_labels = labels

        self._initialized = False
        self._num_filters = num_filters
        self._num_classes = num_classes
        self._xdawn_estimator = XdawnCovariances(num_filters, applyfilters, classes, estimator, xdawn_estimator, baseline_cov)

    def _initialize(self, init_inputs, init_outputs, labels):
        """
        Initialize the internal state of the kernel. Fit the xdawn_estimator classifier, etc.
        """

        init_in = init_inputs[0]
        init_out = init_outputs[0]

        # check if the initialization data is in a Tensor, if not convert it
        if init_in.mp_type != MPEnums.TENSOR:
            init_in = init_in.to_tensor()

        # check if the labels are in a tensor
        if labels.mp_type != MPEnums.TENSOR:
            labels = labels.to_tensor()

        if np.unique(labels.data).shape[0] != self._num_classes:
            raise ValueError("Number of unique labels must match number of classes")

        self._xdawn_estimator.fit(init_in.data, np.squeeze(labels.data))

        if init_in is not None and init_out is not None:
            # update the init output shape as needed
            Nt = init_in.shape[0]
            Nc = self._xdawn_estimator.nfilter*(self._num_classes**2)
            if init_out.shape != (Nt,Nc,Nc):
                init_out.shape = (Nt,Nc,Nc)
            # process the initialization data
            self._process_data([init_in], init_outputs)

    def _process_data(self, inputs, outputs):
        """
        Process input data according to outlined kernel function

        Parameters
        ----------
        input_data : Tensor
            Input data to be processed
        output_data : Tensor
            Output data to be processed

        Returns
        -------
        sts : MPEnums
            Status of the processing
        """
        input_data = inputs[0].data

        if len(inputs[0].shape) == 2:
            input_data = input_data[np.newaxis, :, :] # input must be 3D

        outputs[0].data = self._xdawn_estimator.transform(input_data.data)

    @classmethod
    def add_to_graph(cls, graph, inA, outA, initialization_data=None, labels=None,
                     num_filters=4, applyfilters=True, classes=None,
                     estimator='scm', xdawn_estimator='scm', baseline_cov=None, num_classes=2):
        """
        Factory method to create xdawn_covariance kernel, add it to a node, and add the node to the specified graph.

        Parameters
        ----------

        graph : Graph
            Graph to add the node to
        inA : Tensor
            Input data
        outA : Tensor
            Output data
        initialization_data : Tensor
            Data to initialize the estimator with (n_trials, n_channels, n_samples)
        labels : Tensor
            Class labels for initialization data
        nfilter : int, default=4
            Number of Xdawn filters per class.
        applyfilters : bool, default=True
            If true, spatial filter are applied to the prototypes and the signals. If False, filters are applied only to the ERP prototypes allowing for a better generalization across subject and session at the expense of dimensionality increase. In that case, the estimation is similar to pyriemann.estimation.ERPCovariances with svd=nfilter but with more compact prototype reduction.
        classeslist of int | None, default=None
            list of classes to take into account for prototype estimation. If None, all classes will be accounted.
        estimatorstring, default=’scm’
            Covariance matrix estimator, see pyriemann.utils.covariance.covariances().
        xdawn_estimatorstring, default=’scm’
            Covariance matrix estimator for Xdawn spatial filtering. Should be regularized using ‘lwf’ or ‘oas’, see pyriemann.utils.covariance.covariances().
        baseline_covarray, shape (n_chan, n_chan) | None, default=None
            Baseline covariance for Xdawn spatial filtering, see pyriemann.spatialfilters.Xdawn

        Returns
        -------
        node : Node
            Node containing the kernel
        """
        kernel = cls(graph, inA, outA, initialization_data, labels, num_filters,
                     applyfilters, classes, estimator, xdawn_estimator, baseline_cov, num_classes)

        params = (Parameter(inA,MPEnums.INPUT),
                  Parameter(outA,MPEnums.OUTPUT))

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node

