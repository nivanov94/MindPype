import mindpype as mp
import sys, os
import numpy as np

class ThresholdKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def ThresholdKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2,2))
        thresh = mp.Scalar.create_from_value(self.__session, 1)
        tensor_test_node = mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor, outTensor, thresh)

        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()

        return (inTensor.data, thresh.data, outTensor.data)

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(2,2,2)
    
    KernelExecutionUnitTest_Object = ThresholdKernelUnitTest()
    res = KernelExecutionUnitTest_Object.ThresholdKernelExecution(raw_data)

    output = res[0] > res[1]
    assert (res[2] == output).all()

    del KernelExecutionUnitTest_Object