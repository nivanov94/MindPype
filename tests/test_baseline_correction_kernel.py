import mindpype as mp
import numpy as np

class BaselineCorrectionKernelUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)

    def TestBaselineCorrectionKernelExecution(self, raw_data):
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        outTensor = mp.Tensor.create(self.__session,(10,10))
        node = mp.kernels.BaselineCorrectionKernel.add_to_graph(self.__graph,inTensor,outTensor,baseline_period=[0,10])
        self.__graph.verify()
        self.__graph.initialize()
        self.__graph.execute()
        return outTensor.data

def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(10,10)
    KernelExecutionUnitTest_Object = BaselineCorrectionKernelUnitTest()
    res = KernelExecutionUnitTest_Object.TestBaselineCorrectionKernelExecution(raw_data)
    expected_output = raw_data - np.mean(raw_data, axis=-1, keepdims=True)
    assert (res == expected_output).all()
    del KernelExecutionUnitTest_Object
    