import mindpype as mp
import sys, os
import numpy as np

class BaselineCorrectionKernelCreationUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestBaselineCorrectionKernelCreation(self):
        inTensor = mp.Tensor.create(self.__session, (10,10))
        outTensor = mp.Tensor.create(self.__session,(10,10))
        node = mp.kernels.BaselineCorrectionKernel.add_to_graph(self.__graph,inTensor,outTensor,baseline_period=[0,10])
        return node.mp_type
    
class BaselineCorrectionKernelExecutionUnitTest:

    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestBaselineCorrectionKernelExecution(self):
        np.random.seed(44)
        raw_data = np.random.randint(-10,10, size=(10,10))
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session,(10,10))
        node = mp.kernels.BaselineCorrectionKernel.add_to_graph(self.__graph,inTensor,outTensor,baseline_period=[0,10])
        
        sys.stdout = open(os.devnull, 'w')
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        sys.stdout = sys.__stdout__

        return (raw_data, outTensor.data)

def test_create():
    KernelUnitTest_Object = BaselineCorrectionKernelCreationUnitTest()
    assert KernelUnitTest_Object.TestBaselineCorrectionKernelCreation() == mp.MPEnums.NODE
    del KernelUnitTest_Object


def test_execute():
    KernelExecutionUnitTest_Object = BaselineCorrectionKernelExecutionUnitTest()
    res = KernelExecutionUnitTest_Object.TestBaselineCorrectionKernelExecution()
    expected_output = res[0] - np.mean(res[0], axis=-1, keepdims=True)
    del KernelExecutionUnitTest_Object
    