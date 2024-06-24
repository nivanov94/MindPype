import mindpype as mp
import sys, os
import numpy as np
from pyriemann.estimation import XdawnCovariances

class XDawnKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestXDawnKernelExecution(self, raw_data, initialization_data, init_label_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        init_inputs = mp.Tensor.create_from_data(self.__session, initialization_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)

        outTensor = mp.Tensor.create(self.__session, (50,16,16))
        tensor_test_node = mp.kernels.XDawnCovarianceKernel.add_to_graph(self.__graph, inTensor, outTensor, initialization_data=init_inputs, labels=init_labels)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return (initialization_data, init_label_data, raw_data, outTensor.data)

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(50, 26, 260)
    initialization_data = np.random.randn(50,26,260)
    init_label_data = np.concatenate((np.zeros((25,)), np.ones((25,))))
    
    KernelExecutionUnitTest_Object = XDawnKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestXDawnKernelExecution(raw_data, initialization_data, init_label_data)

    xdawn_estimator = XdawnCovariances(nfilter=4, classes=None)
    xdawn_estimator.fit(res[0], res[1])
    expected_output = xdawn_estimator.transform(res[2])

    assert (res[3] == expected_output).all()

    del KernelExecutionUnitTest_Object
    
