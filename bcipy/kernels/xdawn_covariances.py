from ..core import BcipEnums
from ..kernel import Kernel
from ..graph import Node, Parameter
from .kernel_utils import extract_nested_data
from ..containers import Tensor

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
                 classes=None, estimator='scm', xdawn_estimator='scm', baseline_cov=None):
        """
        Constructor for the XDawnCovarianceKernel class
        """
        super().__init__("XDawnCovarianceKernel", BcipEnums.INIT_FROM_DATA, graph)
        self.inputs = [inA]
        self.outputs = [outA]

        if initialization_data is not None:
            self.init_inputs = [initialization_data]
            
        if labels is not None:
            self.init_input_labels = labels

        self._itialized = False
        self._xdawn_estimator = XdawnCovariances(num_filters, applyfilters, classes, estimator, xdawn_estimator, baseline_cov)

    def verify(self):
        """
        Verify that the input and output data are in the correct format and size
        """
        d_in = self.inputs[0]
        d_out = self.outputs[0]

        if (d_in.bcip_type != BcipEnums.TENSOR or
            d_out.bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS

        if len(d_in.shape) != 2 and len(d_in.shape) != 3:
            return BcipEnums.INVALID_PARAMETERS

        n_classes = 2 # TODO may need to make this a required input
        Nc = self._xdawn_estimator.nfilter*(n_classes**2)

        if len(d_in.shape) == 2:
            d_out.shape = (Nc, Nc)

        elif len(d_in.shape) == 3:
            d_out.shape = (d_in.shape[0], Nc, Nc)

        return BcipEnums.SUCCESS

    def initialize(self):
        """
        Initialize the internal state of the kernel. Fit the xdawn_estimator classifier, etc.
        """

        sts = BcipEnums.SUCCESS

        init_in = self.init_inputs[0]
        labels = self.init_input_labels
        init_out = self.init_outputs[0]

        
        # check if the initialization data is in a Tensor, if not convert it
        if init_in.bcip_type != BcipEnums.TENSOR:
            init_in = init_in.to_tensor()
 
        # check if the labels are in a tensor
        if labels.bcip_type != BcipEnums.TENSOR:
            labels = self._init_labels_in.to_tensor()

        if len(labels.shape) == 2:
            y = np.squeeze(labels.data)
            
        try:
            self._xdawn_estimator.fit(init_in.data, y)
        except Exception as e:
            print("XDawnCovarianceKernel could not be properly fitted. Please check the shape of your initialization data and labels. See the following exception:")
            print(e)
            sts = BcipEnums.INITIALIZATION_FAILURE
        
        if sts == BcipEnums.SUCCESS and init_in is not None and init_out is not None:
            # update the init output shape as needed
            n_classes = np.unique(y).shape[0]
            Nt = init_in.shape[0]
            Nc = self._xdawn_estimator.nfilter*(n_classes**2)
            if init_out.shape != (Nt,Nc,Nc):
                init_out.shape = (Nt,Nc,Nc)
            # process the initialization data
            sts = self._process_data(init_in, init_out)

            # pass on the labels
            self.copy_init_labels_to_output()
        
        if sts == BcipEnums.SUCCESS:
            self._initialzed = True

        return sts
    
    def execute(self):
        """
        Execute processing of trial data
        """
        tmp_data = self.inputs[0].data
        if len(self.inputs[0].shape) == 2:
            tmp_data = tmp_data[np.newaxis, :, :] # input must be 3D
        tmp_tensor = Tensor.create_from_data(self._session, tmp_data.shape, tmp_data)
        return self._process_data(tmp_tensor, self.outputs[0])

    def _process_data(self, input_data, output_data):
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
        sts : BcipEnums
            Status of the processing
        """
        if output_data.bcip_type == BcipEnums.TENSOR:
            result = self._xdawn_estimator.transform(input_data.data)
            
            if len(output_data.shape) == 2 and result.shape[0] == 1:
                result = np.squeeze(result)
            
            output_data.data = result
            return BcipEnums.SUCCESS
        else:
            return BcipEnums.EXE_FAILURE


    @classmethod
    def add_xdawn_covariance_node(cls, graph, inA, outA, initialization_data=None, labels=None,
                                  num_filters=4, applyfilters=True, classes=None, 
                                  estimator='scm', xdawn_estimator='scm', baseline_cov=None):
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
                     applyfilters, classes, estimator, xdawn_estimator, baseline_cov)
        
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT)
                  )

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node

