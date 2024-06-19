import mindpype as mp
import sys, os
import numpy as np

class TransposeKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTransposeKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (2,2,2))
        outTensor = mp.Tensor.create(self.__session, (2,2,2))
        node = mp.kernels.TransposeKernel.add_to_graph(self.__graph,inTensor,outTensor)
        return node.mp_type
    
class TransposeKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTransposeKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2,2))
        tensor_test_node = mp.kernels.TransposeKernel.add_to_graph(self.__graph,inTensor,outTensor)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)

def test_create():
    KernelUnitTest_Object = TransposeKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestTransposeKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = TransposeKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestTransposeKernelExecution()
    assert (res[1] == np.transpose(res[0])).all()
    del KernelExecutionUnitTest_Object