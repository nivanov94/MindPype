"""
Created on Fri Aug 12 14:53:29 2022

xdawn_covariances.py - Calculates the Xdawn covariances 

@author: aaronlio
"""

from pickle import NONE
from types import NoneType

from requests import session
from classes.kernel import Kernel
from classes.bcip_enums import BcipEnums
from classes.parameter import Parameter
from classes.node import Node
from classes.tensor import Tensor

from pyriemann.estimation import XdawnCovariances
import numpy as np

class XDawnCovarianceKernel(Kernel):
    """
    Kernel to estimate special form covariance matrices for ERP combined with Xdawn
    

    Paramters
    ---------

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


    def __init__(self, graph, inA, outA, initialization_data, labels, num_filters=4, applyfilters=True, classes=None, estimator='scm', xdawn_estimator='scm', baseline_cov=None):
        super().__init__("XDawnCovarianceKernel", BcipEnums.INIT_FROM_DATA, graph)
        self._inA = inA
        self._outA = outA

        self._initialization_data = initialization_data
        self._labels = labels
        self._init_inA = None
        self._init_outA = None

        self._xdawn_estimator = XdawnCovariances(num_filters, applyfilters, classes, estimator, xdawn_estimator, baseline_cov)

    def verify(self):
        if not isinstance(self._inA, Tensor):
            return BcipEnums.INVALID_PARAMETERS

        if len(self._inA.shape) != 2 and len(self._inA.shape) != 3:
            return BcipEnums.INVALID_PARAMETERS

        if len(self._inA.shape) == 2:
            self._outA.shape = (self._inA.shape[0], self._inA.shape[0])

        elif len(self._inA.shape) == 3:
            self._outA.shape = (self._inA[0].shape, self._inA[1].shape, self._inA.shape[1])

        return BcipEnums.SUCCESS

    def initialize(self):
        sts1, sts2 = BcipEnums.SUCCESS, BcipEnums.SUCCESS
        if self._initialization_data == None:
            self._initialization_data = self._init_inA
        
        if len(self._labels.shape) == 2:
            temp_labels = np.squeeze(self._labels.data)
            
        try:
            self._xdawn_estimator = self._xdawn_estimator.fit(self._initialization_data.data, temp_labels)
        except:
            print("XDawnCovarianceKernel could not be properly fitted. Please check the shape of your initialization data and labels")
            sts1 = BcipEnums.INITIALIZATION_FAILURE
        
        if self._init_outA.__class__ != NoneType:
            sts2 = self.initilization_execution()
        
        if sts1 != BcipEnums.SUCCESS:
            return sts1
        elif sts2 != BcipEnums.SUCCESS:
            return sts2
        else:
            return BcipEnums.SUCCESS
    
    def execute(self):
        temp_data = self._inA.data
        if len(self._inA.shape) == 2:
            temp_data = temp_data[np.newaxis, :, :]
        temp_tensor = Tensor.create_from_data(self._session, temp_data.shape, temp_data)
        print(temp_tensor.shape)
        return self.process_data(temp_tensor, self._outA)

    def process_data(self, input_data, output_data):
        if isinstance(output_data, Tensor):
            result = self._xdawn_estimator.transform(input_data.data)
            if output_data.shape != result.shape:
                output_data.shape = result.shape
            output_data.data = result
        else:
            return BcipEnums.EXE_FAILURE

        
        return BcipEnums.SUCCESS

    def initilization_execution(self):
        sts = self.process_data(self._initialization_data, self._init_outA)
        
        if sts != BcipEnums.SUCCESS:
            return BcipEnums.INITIALIZATION_FAILURE
        
        return sts

    @classmethod
    def add_xdawn_covariance_kernel(cls, graph, inA, outA, initialization_data, labels, num_filters=4, applyfilters=True, classes=None, estimator='scm', xdawn_estimator='scm', baseline_cov=None):

        kernel = cls(graph, inA, outA, initialization_data, labels, num_filters, applyfilters, classes, estimator, xdawn_estimator, baseline_cov)
        
        params = (Parameter(inA,BcipEnums.INPUT),
                  Parameter(outA,BcipEnums.OUTPUT)
                  )

        node = Node(graph, kernel, params)

        graph.add_node(node)

        return node

