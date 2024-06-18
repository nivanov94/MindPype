import mindpype as mp
import sys, os
import numpy as np
from pyriemann.estimation import XdawnCovariances

class XDawnKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestXDawnKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        labels = mp.Tensor.create(self.__session, (2,))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        node = mp.kernels.XDawnCovarianceKernel.add_to_graph(self.__graph,inTensor,outTensor,labels)
        return node.mp_type
    
class XDawnKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestXDawnKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        label_data = np.random.randint(-10,10, size=(2,))
        labels = mp.Tensor.create_from_data(self.__session, label_data)
        outTensor = mp.Tensor.create(self.__session, (2,2,2))
        tensor_test_node = mp.kernels.XDawnCovarianceKernel.add_to_graph(self.__graph,inTensor, outTensor, labels)

        initialization_data = np.random.randint(-10,10, size=(2,2,2))
        init_inputs = mp.Tensor.create_from_data(self.__session, initialization_data)
        init_outputs = mp.Tensor.create(self.__session, (2,2,2))
        
        init_label_data = np.random.randint(-10,10, size=(2,))
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize(init_inputs,init_outputs,init_labels)
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, labels, outTensor.data)

def test_create():
    KernelUnitTest_Object = XDawnKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestXDawnKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = XDawnKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestXDawnKernelExecution()

    xdawn_estimator = XdawnCovariances(nfilter=4, classes=None)
    xdawn_estimator.fit(res[0], res[1])
    output = xdawn_estimator.transform(res[0])

    assert (res[2] == output).all()

    del KernelExecutionUnitTest_Object