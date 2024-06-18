import mindpype as mp
import sys, os
import numpy as np

class ThresholdKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestThresholdKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (1,1))
        outTensor = mp.Tensor.create(self.__session, (1,1))
        thresh = mp.Scalar.create_from_value(self.__session, 1)
        node = mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor,outTensor,thresh)
        return node.mp_type
    
class ThresholdKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def ThresholdKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(2,2,2))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session, (2,2,2))
        thresh = mp.Scalar.create_from_value(self.__session, 1)
        tensor_test_node = mp.kernels.ThresholdKernel.add_to_graph(self.__graph,inTensor, outTensor, thresh)

        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (inTensor.data, thresh.data, outTensor.data)

def test_create():
    KernelUnitTest_Object = ThresholdKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestThresholdKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = ThresholdKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.ThresholdKernelExecution()

    output = res[0] > res[1]
    assert (res[2] == output).all()

    del KernelExecutionUnitTest_Object