import mindpype as mp
import sys, os
import numpy as np
from pyriemann.estimation import XdawnCovariances

class XDawnKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestXDawnKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (50,26,26))
        labels = mp.Tensor.create(self.__session, (50,))
        outTensor = mp.Tensor.create(self.__session, (50,26,26))
        node = mp.kernels.XDawnCovarianceKernel.add_to_graph(self.__graph,inTensor,outTensor,labels)
        return node.mp_type
    
class XDawnKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestXDawnKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(50, 26, 260)) # TODO use randn to generate inputs to get floats
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        
        initialization_data = np.random.randint(-10,10, size=(50,26,260))
        init_inputs = mp.Tensor.create_from_data(self.__session, initialization_data)
        #init_outputs = mp.Tensor.create(self.__session, (50,26,26))
        
        init_label_data = np.concatenate((np.zeros((25,)), np.ones((25,))))
        init_labels = mp.Tensor.create_from_data(self.__session, init_label_data)

        
        outTensor = mp.Tensor.create(self.__session, (50,16,16))
        tensor_test_node = mp.kernels.XDawnCovarianceKernel.add_to_graph(self.__graph, inTensor, outTensor, initialization_data=init_inputs, labels=init_labels)


        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (initialization_data, init_label_data, raw_data, outTensor.data)

def test_create():
    KernelUnitTest_Object = XDawnKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestXDawnKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = XDawnKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestXDawnKernelExecution()

    xdawn_estimator = XdawnCovariances(nfilter=4, classes=None)
    xdawn_estimator.fit(res[0], res[1])
    output = xdawn_estimator.transform(res[2])

    assert (res[3] == output).all()

    del KernelExecutionUnitTest_Object
    
test_execute()