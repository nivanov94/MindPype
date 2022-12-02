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
    

    Paramters
    ---------
    inA : Tensor object
        - Input data

    outA : Tensor object
        - Output data

    initialization_data : Tensor object
        - Data to initialize the estimator with (n_trials, n_channels, n_samples)

    labels : Tensor object
        - Class labels for initialization data

    nfilter : int, default=4
        - Number of Xdawn filters per class.

    applyfilters : bool, default=True
        - If true, spatial filter are applied to the prototypes and the signals. If False, filters are applied only to the ERP prototypes allowing for a better generalization across subject and session at the expense of dimensionality increase. In that case, the estimation is similar to pyriemann.estimation.ERPCovariances with svd=nfilter but with more compact prototype reduction.

    classeslist of int | None, default=None
        - list of classes to take into account for prototype estimation. If None, all classes will be accounted.

    estimatorstring, default=’scm’
        - Covariance matrix estimator, see pyriemann.utils.covariance.covariances().

    xdawn_estimatorstring, default=’scm’
        - Covariance matrix estimator for Xdawn spatial filtering. Should be regularized using ‘lwf’ or ‘oas’, see pyriemann.utils.covariance.covariances().

    baseline_covarray, shape (n_chan, n_chan) | None, default=None
        - Baseline covariance for Xdawn spatial filtering, see pyriemann.spatialfilters.Xdawn

    Examples
    --------
    """


    def __init__(self, graph, inA, outA, initialization_data, labels, num_filters=4, applyfilters=True, 
                 classes=None, estimator='scm', xdawn_estimator='scm', baseline_cov=None):
        super().__init__("XDawnCovarianceKernel", BcipEnums.INIT_FROM_DATA, graph)
        self._inA = inA
        self._outA = outA

        self._init_inA = initialization_data
        self._labels = labels
        self._init_outA = None
 
        self._xdawn_estimator = XdawnCovariances(num_filters, applyfilters, classes, estimator, xdawn_estimator, baseline_cov)

    def verify(self):
        if (self._inA._bcip_type != BcipEnums.TENSOR or
            self._outA._bcip_type != BcipEnums.TENSOR):
            return BcipEnums.INVALID_PARAMETERS

        if len(self._inA.shape) != 2 and len(self._inA.shape) != 3:
            return BcipEnums.INVALID_PARAMETERS

        n_classes = 2 # TODO may need to make this a required input
        Nc = self._xdawn_estimator.nfilter*(n_classes**2)

        if len(self._inA.shape) == 2:
            self._outA.shape = (Nc, Nc)

        elif len(self._inA.shape) == 3:
            self._outA.shape = (self._inA[0].shape, Nc, Nc)

        return BcipEnums.SUCCESS

    def initialize(self):
        """
        Initialize the internal state of the kernel. Fit the xdawn_estimator classifier, etc.
        """

        sts = BcipEnums.SUCCESS
        
        # check if the initialization data is in a Tensor, if not convert it
        if self._init_inA._bcip_type != BcipEnums.TENSOR:
            local_init_tensor = self._init_inA.to_tensor()
        else:
            local_init_tensor = self._init_inA
 
        # check if the labels are in a tensor
        if self._labels._bcip_type != BcipEnums.TENSOR:
            local_labels = self._labels.to_tensor()
        else:
            local_labels = self._labels

        if len(local_labels.shape) == 2:
            local_labels.data = np.squeeze(local_labels.data)
            
        try:
            self._xdawn_estimator = self._xdawn_estimator.fit(local_init_tensor.data, local_labels.data)
        except:
            #print("XDawnCovarianceKernel could not be properly fitted. Please check the shape of your initialization data and labels")
            sts = BcipEnums.INITIALIZATION_FAILURE
        
        if sts == BcipEnums.SUCCESS and self._init_outA != None:
            # update the init output shape as needed
            n_classes = np.unique(local_labels.data).shape[0]
            Nt = local_init_tensor.shape[0]
            Nc = self._xdawn_estimator.nfilter*(n_classes**2)
            if self._init_outA.shape != (Nt,Nc,Nc):
                self._init_outA.shape = (Nt,Nc,Nc)
            # process the initialization data
            sts = self._process_data(local_init_tensor, self._init_outA)
        
        return sts
    
    def execute(self):
        """
        Execute processing of trial data
        """
        temp_data = self._inA.data
        if len(self._inA.shape) == 2:
            temp_data = temp_data[np.newaxis, :, :] # input must be 3D
        temp_tensor = Tensor.create_from_data(self._session, temp_data.shape, temp_data)
        return self._process_data(temp_tensor, self._outA)

    def _process_data(self, input_data, output_data):
        """
        Process input data according to outlined kernel function
        """
        if output_data._bcip_type == BcipEnums.TENSOR:
            result = self._xdawn_estimator.transform(input_data.data)
            
            if len(output_data.shape) == 2 and result.shape[0] == 1:
                result = np.squeeze(result)
            
            output_data.data = result
            return BcipEnums.SUCCESS
        else:
            return BcipEnums.EXE_FAILURE


    @classmethod
    def add_xdawn_covariance_node(cls, graph, inA, outA, initialization_data, labels,
                                  num_filters=4, applyfilters=True, classes=None, 
                                  estimator='scm', xdawn_estimator='scm', baseline_cov=None):
        """
        Factory method to create xdawn_covariance kernel, add it to a node, and add the node to the specified graph.

        inA : Tensor object
            - Input data

        outA : Tensor object
            - Output data

        initialization_data : Tensor object
            - Data to initialize the estimator with (n_trials, n_channels, n_samples)

        labels : Tensor object
            - Class labels for initialization data

        nfilter : int, default=4
            - Number of Xdawn filters per class.

        applyfilters : bool, default=True
            - If true, spatial filter are applied to the prototypes and the signals. If False, filters are applied only to the ERP prototypes allowing for a better generalization across subject and session at the expense of dimensionality increase. In that case, the estimation is similar to pyriemann.estimation.ERPCovariances with svd=nfilter but with more compact prototype reduction.

        classeslist of int | None, default=None
            - list of classes to take into account for prototype estimation. If None, all classes will be accounted.

        estimatorstring, default=’scm’
            - Covariance matrix estimator, see pyriemann.utils.covariance.covariances().

        xdawn_estimatorstring, default=’scm’
            - Covariance matrix estimator for Xdawn spatial filtering. Should be regularized using ‘lwf’ or ‘oas’, see pyriemann.utils.covariance.covariances().

        baseline_covarray, shape (n_chan, n_chan) | None, default=None
            - Baseline covariance for Xdawn spatial filtering, see pyriemann.spatialfilters.Xdawn
        """
        kernel = cls(graph, inA, outA, initialization_data, labels, num_filters,
                     applyfilters, classes, estimator, xdawn_estimator, baseline_cov)
        
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT)
                  )

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node

