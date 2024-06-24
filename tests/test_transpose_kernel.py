import mindpype as mp
import sys, os
import numpy as np

class TransposeKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestTransposeKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2,2))
        tensor_test_node = mp.kernels.TransposeKernel.add_to_graph(self.__graph,inTensor,outTensor)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return outTensor.data


def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    
    KernelExecutionUnitTest_Object = TransposeKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestTransposeKernelExecution(raw_data)
    assert (res == np.transpose(raw_data)).all()
    del KernelExecutionUnitTest_Object